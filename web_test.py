import threading
import time
import uvicorn

from hardware.accelerometer import BMI160
from hardware.drive_module import DriveModule
from web.web_server import create_app


motor_power = 0.0
lock = threading.Lock()


def set_motor_power(value: float):
    global motor_power

    value = max(-1.0, min(1.0, value))

    with lock:
        motor_power = value

    print(f"Motor power changed: {motor_power:.2f}")


def get_motor_power():
    with lock:
        return motor_power


def robot_loop():
    imu = BMI160(bus_id=1)
    drive = DriveModule()
    drive.reset_encoders()

    try:
        while True:
            power = get_motor_power()

            if power > 0:
                drive.forward(power)
            elif power < 0:
                drive.backward(abs(power))
            else:
                drive.stop()

            ax, ay, az = imu.read_acc()
            gx, gy, gz = imu.read_gyro()
            e1, e2 = drive.get_encoder_steps()

            print(
                f"POWER:{power:.2f} | "
                f"ENC M1:{e1} M2:{e2} | "
                f"ACC:{ax},{ay},{az} | "
                f"GYRO:{gx},{gy},{gz}"
            )

            time.sleep(0.1)

    finally:
        drive.close()


def main():
    robot_thread = threading.Thread(target=robot_loop, daemon=True)
    robot_thread.start()

    app = create_app(on_motor_power_change=set_motor_power)

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
