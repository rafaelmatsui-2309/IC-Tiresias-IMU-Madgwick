"""
Visualizador BLE — IMU nRF5340 + BMI270
----------------------------------------
Versão do visualizer_joao.py adaptada para receber
quaternion via BLE em vez de serial USB.

Protocolo BLE:
  - Nome do dispositivo: "Tiresias_DK"  (CONFIG_BT_DEVICE_NAME)
  - CHAR_UUID: "12345678-1234-5678-1234-56789abcdef1"
  - Dados: 16 bytes little-endian (4 floats: w, x, y, z)

Dependências:
    pip install pyqtgraph PyQt6 PyOpenGL numpy bleak qasync
"""

import sys
import math
import struct
import asyncio
import numpy as np

from bleak import BleakScanner, BleakClient
import qasync

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QFrame)
from PyQt6.QtCore    import QTimer, Qt
from PyQt6.QtGui     import QMatrix4x4, QFont

import pyqtgraph.opengl as gl

# ── Tenta carregar numpy-stl (opcional) ───────────────────────────────────────
try:
    from stl import mesh as stl_mesh
    HAS_STL = True
except ImportError:
    HAS_STL = False

# ===========================================================
# CONFIG BLE — deve bater com o firmware
# ===========================================================
DEVICE_NAME = "Tiresias_DK"
CHAR_UUID   = "12345678-1234-5678-1234-56789abcdef1"

# ===========================================================
# MATEMÁTICA DE QUATERNION (portada do aar_core.py do João)
# ===========================================================

def quaternion_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )

def quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,  2*x*y-2*z*w,    2*x*z+2*y*w,   0],
        [2*x*y+2*z*w,    1-2*x*x-2*z*z,  2*y*z-2*x*w,   0],
        [2*x*z-2*y*w,    2*y*z+2*x*w,    1-2*x*x-2*y*y, 0],
        [0,              0,              0,              1],
    ], dtype=np.float32)

def quat_rotate(q, v):
    w, x, y, z = q
    qvec = np.array([x, y, z])
    uv   = np.cross(qvec, v)
    uuv  = np.cross(qvec, uv)
    return v + 2 * (w * uv + uuv)

def quaternion_to_euler(q):
    w, x, y, z = q
    sinr = 2*(w*x + y*z);  cosr = 1 - 2*(x*x + y*y)
    roll  = math.degrees(math.atan2(sinr, cosr))
    sinp  = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.degrees(math.asin(sinp))
    siny  = 2*(w*z + x*y);  cosy = 1 - 2*(y*y + z*z)
    yaw   = math.degrees(math.atan2(siny, cosy))
    return roll, pitch, yaw

# ===========================================================
# CLIENTE BLE
# ===========================================================

class BLEReader:
    """
    Gerencia a conexão BLE em background.
    Recebe notificações do firmware e atualiza self.quaternion.
    """

    def __init__(self):
        self.quaternion  = (1.0, 0.0, 0.0, 0.0)
        self.connected   = False
        self._client     = None
        self._status_cb  = None   # callback para atualizar UI

    def set_status_callback(self, cb):
        self._status_cb = cb

    def _notify(self, status):
        if self._status_cb:
            self._status_cb(status)

    def notification_handler(self, sender, data):
        """
        Chamada pelo bleak a cada notificação BLE.
        Desempacota 16 bytes little-endian → 4 floats (w, x, y, z)
        Mesmo formato que o projeto João usa.
        """
        if len(data) != 16:
            return

        q_raw = struct.unpack("<ffff", data)  # little-endian, 4 floats

        # Normaliza (segurança extra)
        norm = math.sqrt(sum(v*v for v in q_raw))
        if norm > 0:
            self.quaternion = tuple(v/norm for v in q_raw)

    async def connect(self):
        self._notify("Escaneando...")
        print(f"Procurando '{DEVICE_NAME}'...")

        device = None
        devices = await BleakScanner.discover(timeout=10.0)
        for d in devices:
            if d.name and DEVICE_NAME in d.name:
                device = d
                break

        if device is None:
            self._notify("Dispositivo não encontrado")
            print(f"'{DEVICE_NAME}' não encontrado. Verifique se o firmware está rodando.")
            return

        print(f"Encontrado: {device.name} ({device.address})")
        self._notify("Conectando...")

        try:
            self._client = BleakClient(device,
                                       disconnected_callback=self._on_disconnect)
            await self._client.connect()
            await self._client.start_notify(CHAR_UUID, self.notification_handler)
            self.connected = True
            self._notify("Conectado")
            print("BLE conectado! Recebendo quaternion...\n")

        except Exception as e:
            self._notify(f"Erro: {e}")
            print(f"Erro BLE: {e}")

    def _on_disconnect(self, client):
        self.connected = False
        self._notify("Desconectado")
        print("\nBLE desconectado.")

    async def disconnect(self):
        if self._client and self._client.is_connected:
            await self._client.disconnect()
        self.connected = False


# ===========================================================
# MODELO 3D
# ===========================================================

def make_board_mesh():
    w, h, d = 1.4, 1.0, 0.06
    vertices = np.array([
        [-w,-h, d],[ w,-h, d],[ w, h, d],[-w, h, d],
        [-w,-h,-d],[ w,-h,-d],[ w, h,-d],[-w, h,-d],
    ], dtype=np.float32)
    faces = np.array([
        [0,1,2],[0,2,3],[4,6,5],[4,7,6],
        [0,4,5],[0,5,1],[2,6,7],[2,7,3],
        [0,3,7],[0,7,4],[1,5,6],[1,6,2],
    ])
    colors = np.array([
        [0.2,0.6,1.0,1.0],[0.2,0.6,1.0,1.0],
        [0.1,0.3,0.6,1.0],[0.1,0.3,0.6,1.0],
        [0.2,0.8,0.3,1.0],[0.2,0.8,0.3,1.0],
        [0.1,0.5,0.2,1.0],[0.1,0.5,0.2,1.0],
        [0.8,0.3,0.2,1.0],[0.8,0.3,0.2,1.0],
        [0.5,0.1,0.1,1.0],[0.5,0.1,0.1,1.0],
    ], dtype=np.float32)
    return gl.MeshData(vertexes=vertices, faces=faces, faceColors=colors)


# ===========================================================
# JANELA PRINCIPAL
# ===========================================================

class MainWindow(QMainWindow):

    def __init__(self, loop):
        super().__init__()
        self._loop   = loop
        self.reader  = BLEReader()
        self.reader.set_status_callback(self._set_status)
        self.tare_quat = None

        self.setWindowTitle("IMU BLE Visualizer — nRF5340 + BMI270")
        self.resize(1100, 700)
        self.setStyleSheet("background:#1e1e2e; color:#cdd6f4;")

        self._build_ui()
        self._build_3d()

        self._timer = QTimer()
        self._timer.timeout.connect(self._update)
        self._timer.start(16)   # ~60 FPS

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(0)

        panel = QFrame()
        panel.setFixedWidth(240)
        panel.setStyleSheet("background:#181825; border-right:1px solid #313244;")
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(16,20,16,20)
        pv.setSpacing(12)

        title = QLabel("IMU BLE Visualizer")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet("color:#cba6f7;")
        pv.addWidget(title)

        sub = QLabel("nRF5340 Audio DK + BMI270")
        sub.setStyleSheet("color:#6c7086; font-size:11px;")
        pv.addWidget(sub)

        pv.addWidget(self._sep())

        # Botão conectar
        self.btn_connect = QPushButton("Conectar BLE")
        self.btn_connect.setStyleSheet(self._btn("#89b4fa"))
        self.btn_connect.clicked.connect(self._toggle_connect)
        pv.addWidget(self.btn_connect)

        pv.addWidget(self._sep())

        # Tare
        pv.addWidget(self._lbl("Calibração"))
        self.btn_tare = QPushButton("Tare  [ESPAÇO]")
        self.btn_tare.setStyleSheet(self._btn("#a6e3a1"))
        self.btn_tare.clicked.connect(self._do_tare)
        pv.addWidget(self.btn_tare)

        self.btn_reset = QPushButton("Reset Tare  [R]")
        self.btn_reset.setStyleSheet(self._btn("#f38ba8"))
        self.btn_reset.clicked.connect(self._reset_tare)
        pv.addWidget(self.btn_reset)

        pv.addWidget(self._sep())

        # HUD
        pv.addWidget(self._lbl("Orientação"))
        self.lbl_roll  = self._val("Roll:   —")
        self.lbl_pitch = self._val("Pitch:  —")
        self.lbl_yaw   = self._val("Yaw:    —")
        self.lbl_w     = self._val("W: —")
        self.lbl_x     = self._val("X: —")
        self.lbl_y     = self._val("Y: —")
        self.lbl_z     = self._val("Z: —")
        for w in [self.lbl_roll, self.lbl_pitch, self.lbl_yaw,
                  self._sep(),
                  self.lbl_w, self.lbl_x, self.lbl_y, self.lbl_z]:
            pv.addWidget(w)

        pv.addStretch()

        self.lbl_status = QLabel("● Desconectado")
        self.lbl_status.setStyleSheet("color:#f38ba8; font-size:11px;")
        pv.addWidget(self.lbl_status)

        root.addWidget(panel)

        self.gl_container = QWidget()
        root.addWidget(self.gl_container, stretch=1)
        QVBoxLayout(self.gl_container).setContentsMargins(0,0,0,0)

    def _build_3d(self):
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('#11111b')
        self.view.setCameraPosition(distance=5, elevation=20, azimuth=30)

        grid = gl.GLGridItem()
        grid.setColor((50,50,70,80))
        self.view.addItem(grid)
        self.view.addItem(gl.GLAxisItem())

        self.board = gl.GLMeshItem(
            meshdata=make_board_mesh(),
            smooth=False, drawEdges=True,
            edgeColor=(0.3,0.3,0.4,1.0),
        )
        self.view.addItem(self.board)

        self.forward_line = gl.GLLinePlotItem(
            pos=np.array([[0,0,0],[1,0,0]]),
            color=(1.0,0.8,0.0,1.0), width=3, antialias=True,
        )
        self.view.addItem(self.forward_line)

        self.gl_container.layout().addWidget(self.view)

    # ── Render ────────────────────────────────────────────────────────────────

    def _update(self):
        q_raw = self.reader.quaternion

        # Aplica tare (mesmo sistema do projeto João)
        q = quaternion_multiply(self.tare_quat, q_raw) \
            if self.tare_quat else q_raw

        # Quaternion → matriz 4×4 (sem passar por Euler → sem gimbal lock)
        mat = quaternion_to_matrix(q)
        self.board.setTransform(QMatrix4x4(*mat.flatten()))

        forward = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        self.forward_line.setData(pos=np.array([[0,0,0], forward*2]))

        roll, pitch, yaw = quaternion_to_euler(q)
        self.lbl_roll.setText( f"Roll:   {roll:+.1f}°")
        self.lbl_pitch.setText(f"Pitch:  {pitch:+.1f}°")
        self.lbl_yaw.setText(  f"Yaw:    {yaw:+.1f}°")
        self.lbl_w.setText(f"W:  {q[0]:+.4f}")
        self.lbl_x.setText(f"X:  {q[1]:+.4f}")
        self.lbl_y.setText(f"Y:  {q[2]:+.4f}")
        self.lbl_z.setText(f"Z:  {q[3]:+.4f}")

    # ── Ações ─────────────────────────────────────────────────────────────────

    def _toggle_connect(self):
        if not self.reader.connected:
            self.btn_connect.setText("Conectando...")
            self.btn_connect.setEnabled(False)
            # Dispara corrotina BLE no loop asyncio
            asyncio.ensure_future(self._connect_ble())
        else:
            asyncio.ensure_future(self.reader.disconnect())
            self.btn_connect.setText("Conectar BLE")

    async def _connect_ble(self):
        await self.reader.connect()
        self.btn_connect.setEnabled(True)
        if self.reader.connected:
            self.btn_connect.setText("Desconectar")
        else:
            self.btn_connect.setText("Conectar BLE")

    def _do_tare(self):
        self.tare_quat = quaternion_conjugate(self.reader.quaternion)
        print("Tare aplicado")

    def _reset_tare(self):
        self.tare_quat = None
        print("Tare resetado")

    def _set_status(self, text):
        is_ok = "Conectado" in text
        color = "#a6e3a1" if is_ok else "#f38ba8"
        self.lbl_status.setText(f"● {text}")
        self.lbl_status.setStyleSheet(f"color:{color}; font-size:11px;")

    # ── Teclado ───────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._do_tare()
        elif event.key() == Qt.Key.Key_R:
            self._reset_tare()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        asyncio.ensure_future(self.reader.disconnect())
        event.accept()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _lbl(self, t):
        l = QLabel(t)
        l.setStyleSheet("color:#6c7086; font-size:11px; font-weight:600;")
        return l

    def _val(self, t):
        l = QLabel(t)
        l.setFont(QFont("Courier New", 11))
        l.setStyleSheet("color:#cdd6f4;")
        return l

    def _sep(self):
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("color:#313244;")
        return f

    def _btn(self, c):
        return f"""
            QPushButton {{
                background:transparent; border:1px solid {c};
                border-radius:6px; color:{c}; padding:6px; font-size:12px;
            }}
            QPushButton:hover   {{ background:{c}22; }}
            QPushButton:pressed {{ background:{c}44; }}
            QPushButton:disabled {{ opacity:0.4; }}
        """


# ===========================================================
# ENTRY POINT
# ===========================================================

if __name__ == "__main__":
    app  = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    win = MainWindow(loop)
    win.show()

    with loop:
        loop.run_forever()
