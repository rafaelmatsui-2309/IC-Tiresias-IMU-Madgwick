from vpython import *
import serial
import numpy as np

# === AJUSTE AQUI SUA PORTA ===
PORT = 'COM7'        # Windows
# PORT = '/dev/ttyACM0'  # Linux

BAUDRATE = 115200

ser = serial.Serial(PORT, BAUDRATE)

scene = canvas(title="IMU Orientation")
scene.width = 800
scene.height = 600

cube = box(length=2, height=0.4, width=1)

def quaternion_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

while True:
    rate(100)

    try:
        line = ser.readline().decode(errors='ignore').strip()


        if line.startswith("Q:"):
            data = line[2:].split(',')
            print(line)

            if len(data) == 4:
                q = np.array([float(v) for v in data])
                q = q / np.linalg.norm(q)

                R = quaternion_to_matrix(q)

                forward = vector(R[0,2], R[1,2], R[2,2])
                up = vector(R[0,1], R[1,1], R[2,1])

                cube.axis = forward
                cube.up = up

    except Exception as e:
        print("Erro:", e)


