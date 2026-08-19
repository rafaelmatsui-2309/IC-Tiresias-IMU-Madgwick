"""
Visualizador de Orientação Espacial - Adaptado do Projeto João
--------------------------------------------------------------
Usa apenas a parte de visualização 3D do projeto João,
sem áudio, sem BLE — recebe quaternion via Serial USB
no mesmo formato do seu firmware atual: Q:w,x,y,z

Dependências:
    pip install pyqtgraph PyQt6 numpy numpy-stl pyserial

Controles:
    ESPAÇO → Tare (calibra posição atual como "frente")
    R       → Reset do tare
    ESC     → Fechar
"""

import sys
import math
import struct
import threading
import numpy as np
import serial
import serial.tools.list_ports

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QComboBox, QSpinBox, QFrame)
from PyQt6.QtCore    import QTimer, Qt
from PyQt6.QtGui     import QMatrix4x4, QFont, QColor

import pyqtgraph.opengl as gl

# ── Tenta carregar numpy-stl (opcional) ───────────────────────────────────────
try:
    from stl import mesh as stl_mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False
    print("[aviso] numpy-stl não instalado — usando modelo de placa simples.")


# ============================================================
# MATEMÁTICA DE QUATERNION  (portada do aar_core.py do João)
# ============================================================

def quaternion_conjugate(q):
    """
    Inverte a rotação representada pelo quaternion.
    Se q representa "girar 30° à direita",
    o conjugado representa "girar 30° à esquerda".
    Usado para o tare: subtrai matematicamente a posição de referência.
    """
    w, x, y, z = q
    return (w, -x, -y, -z)


def quaternion_multiply(q1, q2):
    """
    Compõe duas rotações em sequência.
    NÃO é comutativo: q1*q2 ≠ q2*q1
    Usado para aplicar o tare:
        q_relativo = tare_conjugado × q_atual
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def quaternion_to_matrix(q):
    """
    Converte quaternion → matriz de rotação 4×4 (formato OpenGL).

    VANTAGEM sobre converter para Euler:
    - Sem gimbal lock
    - Sem problema de "volta completa" no roll
    - Vai direto para a GPU sem passar por ângulos intermediários

    Esta é a razão pela qual o projeto João não tem os problemas
    de singularidade que apareceram no filtro Kalman com Euler.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w,      0],
        [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w,      0],
        [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y,  0],
        [0,                  0,                  0,                   1],
    ], dtype=np.float32)


def quat_rotate(q, v):
    """
    Rotaciona o vetor v pelo quaternion q.
    Equivalente a q * v * q⁻¹, mas otimizado para um único vetor.
    Usado para calcular o vetor "frente" da placa.
    """
    w, x, y, z = q
    qvec = np.array([x, y, z])
    uv   = np.cross(qvec, v)
    uuv  = np.cross(qvec, uv)
    return v + 2 * (w * uv + uuv)


def quaternion_to_euler(q):
    """Converte quaternion → (roll, pitch, yaw) em graus — apenas para o HUD."""
    w, x, y, z = q

    sinr = 2 * (w*x + y*z)
    cosr = 1 - 2 * (x*x + y*y)
    roll = math.degrees(math.atan2(sinr, cosr))

    sinp = 2 * (w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))

    siny = 2 * (w*z + x*y)
    cosy = 1 - 2 * (y*y + z*z)
    yaw  = math.degrees(math.atan2(siny, cosy))

    return roll, pitch, yaw


# ============================================================
# LEITURA SERIAL  (mesmo protocolo do seu firmware)
# ============================================================

class SerialReader:
    """
    Lê quaternions da serial em background.
    Protocolo esperado (mesmo do seu firmware):
        Q:w,x,y,z\n
    """

    def __init__(self):
        self.quaternion = (1.0, 0.0, 0.0, 0.0)
        self._serial    = None
        self._thread    = None
        self._running   = False

    def connect(self, port: str, baudrate: int = 115200) -> bool:
        try:
            self._serial  = serial.Serial(port, baudrate, timeout=0.1)
            self._running = True
            self._thread  = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"Erro ao abrir serial: {e}")
            return False

    def disconnect(self):
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()

    def _read_loop(self):
        latest = None
        while self._running:
            try:
                # Drena o buffer e guarda só o Q: mais recente
                while self._serial.in_waiting > 0:
                    line = self._serial.readline().decode(errors='ignore').strip()
                    if line.startswith("Q:"):
                        latest = line

                if latest:
                    parts = latest[2:].split(',')
                    if len(parts) == 4:
                        q = tuple(float(v) for v in parts)
                        # Normaliza
                        norm = math.sqrt(sum(v*v for v in q))
                        if norm > 0:
                            self.quaternion = tuple(v/norm for v in q)
                    latest = None

            except Exception:
                pass

    @staticmethod
    def list_ports():
        return [p.device for p in serial.tools.list_ports.comports()]


# ============================================================
# MODELO 3D  (placa simples — substitua por STL se quiser)
# ============================================================

def make_board_mesh():
    """
    Cria um modelo simplificado do nRF5340 Audio DK:
    um paralelepípedo achatado com faces coloridas por eixo.
    """
    # Dimensões proporcionais ao DK (largura × altura × espessura)
    w, h, d = 1.4, 1.0, 0.06

    vertices = np.array([
        # frente (Z+)
        [-w, -h,  d], [ w, -h,  d], [ w,  h,  d], [-w,  h,  d],
        # trás  (Z-)
        [-w, -h, -d], [ w, -h, -d], [ w,  h, -d], [-w,  h, -d],
    ], dtype=np.float32)

    faces = np.array([
        [0,1,2],[0,2,3],   # frente
        [4,6,5],[4,7,6],   # trás
        [0,4,5],[0,5,1],   # baixo
        [2,6,7],[2,7,3],   # cima
        [0,3,7],[0,7,4],   # esquerda
        [1,5,6],[1,6,2],   # direita
    ])

    colors = np.array([
        [0.2, 0.6, 1.0, 1.0],  # frente   — azul (eixo Z+)
        [0.2, 0.6, 1.0, 1.0],
        [0.1, 0.3, 0.6, 1.0],  # trás     — azul escuro
        [0.1, 0.3, 0.6, 1.0],
        [0.2, 0.8, 0.3, 1.0],  # baixo    — verde
        [0.2, 0.8, 0.3, 1.0],
        [0.1, 0.5, 0.2, 1.0],  # cima     — verde escuro
        [0.1, 0.5, 0.2, 1.0],
        [0.8, 0.3, 0.2, 1.0],  # esquerda — vermelho
        [0.8, 0.3, 0.2, 1.0],
        [0.5, 0.1, 0.1, 1.0],  # direita  — vermelho escuro
        [0.5, 0.1, 0.1, 1.0],
    ], dtype=np.float32)

    return gl.MeshData(vertexes=vertices, faces=faces, faceColors=colors)


def load_stl_mesh(path: str):
    """Carrega modelo STL externo (opcional)."""
    if not HAS_STL:
        return None
    try:
        m        = stl_mesh.Mesh.from_file(path)
        vertices = m.vectors.reshape(-1, 3).astype(np.float32)
        faces    = np.arange(vertices.shape[0]).reshape(-1, 3)
        # Centraliza e normaliza escala
        vertices -= vertices.mean(axis=0)
        vertices /= np.max(np.linalg.norm(vertices, axis=1))
        return gl.MeshData(vertexes=vertices, faces=faces)
    except Exception as e:
        print(f"Erro ao carregar STL: {e}")
        return None


# ============================================================
# JANELA PRINCIPAL
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Visualizer — nRF5340 + BMI270")
        self.resize(1100, 700)
        self.setStyleSheet("background:#1e1e2e; color:#cdd6f4;")

        self.reader        = SerialReader()
        self.tare_quat     = None   # quaternion de referência (tare)
        self.connected     = False

        self._build_ui()
        self._build_3d()

        # Timer de render: ~60 FPS
        self._timer = QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(16)

    # ── Interface ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Painel lateral esquerdo
        panel = QFrame()
        panel.setFixedWidth(240)
        panel.setStyleSheet("background:#181825; border-right:1px solid #313244;")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(16, 20, 16, 20)
        pv.setSpacing(12)

        # Título
        title = QLabel("IMU Visualizer")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color:#cba6f7;")
        pv.addWidget(title)

        sub = QLabel("nRF5340 Audio DK + BMI270")
        sub.setStyleSheet("color:#6c7086; font-size:11px;")
        pv.addWidget(sub)

        pv.addWidget(self._separator())

        # Porta serial
        pv.addWidget(self._label("Porta Serial"))
        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet(self._combo_style())
        self._refresh_ports()
        pv.addWidget(self.port_combo)

        pv.addWidget(self._label("Baudrate"))
        self.baud_spin = QSpinBox()
        self.baud_spin.setRange(9600, 921600)
        self.baud_spin.setValue(115200)
        self.baud_spin.setSingleStep(9600)
        self.baud_spin.setStyleSheet(self._combo_style())
        pv.addWidget(self.baud_spin)

        self.btn_connect = QPushButton("Conectar")
        self.btn_connect.setStyleSheet(self._btn_style("#89b4fa"))
        self.btn_connect.clicked.connect(self._toggle_connect)
        pv.addWidget(self.btn_connect)

        pv.addWidget(self._separator())

        # Tare
        pv.addWidget(self._label("Calibração"))

        self.btn_tare = QPushButton("Tare  [ESPAÇO]")
        self.btn_tare.setStyleSheet(self._btn_style("#a6e3a1"))
        self.btn_tare.clicked.connect(self._do_tare)
        pv.addWidget(self.btn_tare)

        self.btn_reset = QPushButton("Reset Tare  [R]")
        self.btn_reset.setStyleSheet(self._btn_style("#f38ba8"))
        self.btn_reset.clicked.connect(self._reset_tare)
        pv.addWidget(self.btn_reset)

        pv.addWidget(self._separator())

        # HUD de valores
        pv.addWidget(self._label("Orientação"))
        self.lbl_roll  = self._value_label("Roll:   —")
        self.lbl_pitch = self._value_label("Pitch:  —")
        self.lbl_yaw   = self._value_label("Yaw:    —")
        self.lbl_w     = self._value_label("W: —")
        self.lbl_x     = self._value_label("X: —")
        self.lbl_y     = self._value_label("Y: —")
        self.lbl_z     = self._value_label("Z: —")
        for lbl in [self.lbl_roll, self.lbl_pitch, self.lbl_yaw,
                    self._separator(),
                    self.lbl_w, self.lbl_x, self.lbl_y, self.lbl_z]:
            pv.addWidget(lbl)

        pv.addStretch()

        # Status
        self.lbl_status = QLabel("● Desconectado")
        self.lbl_status.setStyleSheet("color:#f38ba8; font-size:11px;")
        pv.addWidget(self.lbl_status)

        root.addWidget(panel)

        # Container 3D
        self.gl_container = QWidget()
        root.addWidget(self.gl_container, stretch=1)
        gl_layout = QVBoxLayout(self.gl_container)
        gl_layout.setContentsMargins(0, 0, 0, 0)

    def _build_3d(self):
        """Monta a cena 3D igual ao projeto João."""
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('#11111b')
        self.view.setCameraPosition(distance=5, elevation=20, azimuth=30)

        # Grade e eixos
        grid = gl.GLGridItem()
        grid.setColor((50, 50, 70, 80))
        self.view.addItem(grid)
        self.view.addItem(gl.GLAxisItem(size=QVector3D_compat(1.5, 1.5, 1.5)))

        # Modelo da placa
        mesh_data = make_board_mesh()
        self.board = gl.GLMeshItem(
            meshdata  = mesh_data,
            smooth    = False,
            drawEdges = True,
            edgeColor = (0.3, 0.3, 0.4, 1.0),
        )
        self.view.addItem(self.board)

        # Linha de direção "frente" (eixo X+ da placa — igual ao projeto João)
        self.forward_line = gl.GLLinePlotItem(
            pos       = np.array([[0,0,0],[1,0,0]]),
            color     = (1.0, 0.8, 0.0, 1.0),
            width     = 3,
            antialias = True,
        )
        self.view.addItem(self.forward_line)

        # Adiciona ao layout
        self.gl_container.layout().addWidget(self.view)

    # ── Atualização ────────────────────────────────────────────────────────────

    def _update(self):
        q_raw = self.reader.quaternion  # quaternion cru do sensor

        # Aplica tare (igual ao projeto João)
        if self.tare_quat is not None:
            q = quaternion_multiply(self.tare_quat, q_raw)
        else:
            q = q_raw

        # Quaternion → matriz 4×4 → aplica no modelo
        # (sem passar por Euler — elimina gimbal lock e volta completa)
        mat = quaternion_to_matrix(q)
        self.board.setTransform(QMatrix4x4(*mat.flatten()))

        # Calcula vetor "frente" e atualiza linha direcional
        forward = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        self.forward_line.setData(
            pos = np.array([[0.0, 0.0, 0.0], forward * 2.0])
        )

        # Atualiza HUD
        roll, pitch, yaw = quaternion_to_euler(q)
        self.lbl_roll.setText( f"Roll:   {roll:+.1f}°")
        self.lbl_pitch.setText(f"Pitch:  {pitch:+.1f}°")
        self.lbl_yaw.setText(  f"Yaw:    {yaw:+.1f}°")
        self.lbl_w.setText(f"W:  {q[0]:+.4f}")
        self.lbl_x.setText(f"X:  {q[1]:+.4f}")
        self.lbl_y.setText(f"Y:  {q[2]:+.4f}")
        self.lbl_z.setText(f"Z:  {q[3]:+.4f}")

    # ── Ações ──────────────────────────────────────────────────────────────────

    def _toggle_connect(self):
        if not self.connected:
            port = self.port_combo.currentText()
            baud = self.baud_spin.value()
            if self.reader.connect(port, baud):
                self.connected = True
                self.btn_connect.setText("Desconectar")
                self.lbl_status.setText("● Conectado")
                self.lbl_status.setStyleSheet("color:#a6e3a1; font-size:11px;")
        else:
            self.reader.disconnect()
            self.connected = False
            self.btn_connect.setText("Conectar")
            self.lbl_status.setText("● Desconectado")
            self.lbl_status.setStyleSheet("color:#f38ba8; font-size:11px;")

    def _do_tare(self):
        """
        Captura a orientação atual como referência (posição "zero").
        Igual ao tare do projeto João:
            tare = conjugado do quaternion atual
            q_relativo = tare × q_futuro
        Resolve o problema de yaw não voltar à posição inicial!
        """
        q = self.reader.quaternion
        self.tare_quat = quaternion_conjugate(q)
        print(f"Tare aplicado: {q}")

    def _reset_tare(self):
        self.tare_quat = None
        print("Tare resetado")

    def _refresh_ports(self):
        self.port_combo.clear()
        for p in SerialReader.list_ports():
            self.port_combo.addItem(p)

    # ── Teclado ────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._do_tare()
        elif event.key() == Qt.Key.Key_R:
            self._reset_tare()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        self.reader.disconnect()
        event.accept()

    # ── Helpers de estilo ──────────────────────────────────────────────────────

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#6c7086; font-size:11px; font-weight:600;")
        return lbl

    def _value_label(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Courier New", 11))
        lbl.setStyleSheet("color:#cdd6f4;")
        return lbl

    def _separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#313244;")
        return line

    def _btn_style(self, color):
        return f"""
            QPushButton {{
                background: transparent;
                border: 1px solid {color};
                border-radius: 6px;
                color: {color};
                padding: 6px;
                font-size: 12px;
            }}
            QPushButton:hover {{ background: {color}22; }}
            QPushButton:pressed {{ background: {color}44; }}
        """

    def _combo_style(self):
        return """
            QComboBox, QSpinBox {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                color: #cdd6f4;
                padding: 4px 8px;
            }
        """


# ── Compatibilidade GLAxisItem ─────────────────────────────────────────────────

def QVector3D_compat(x, y, z):
    """GLAxisItem aceita QtGui.QVector3D ou pyqtgraph.Vector."""
    try:
        from PyQt6.QtGui import QVector3D
        return QVector3D(x, y, z)
    except Exception:
        import pyqtgraph as pg
        return pg.Vector(x, y, z)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())