from hardware.accelerometer import BMI160
from hardware.drive_module import DriveModule

import threading
import time
import sys
import tty
import termios


motor_power = 0.0
running = True


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return key


def keyboard_loop():
    global motor_power, running

    print("\nSterowanie:")
    print("+ -> zwiększ moc")
    print("- -> zmniejsz moc")
    print("0 -> stop")
    print("q -> wyjście\n")

    while running:
        key = get_key()

        if key == "=":
            motor_power = min(motor_power + 0.1, 1.0)
        elif key == "-":
            motor_power = max(motor_power - 0.1, -1.0)
        elif key == "o":
            motor_power = -0.5
        elif key == "p":
            motor_power = 0.5
        elif key == "0":
            motor_power = 0.0

        elif key == "q":
            running = False

        print(f"\rMotor power: {motor_power:.1f}      ", end="", flush=True)


def main():
    global running, motor_power

    imu = BMI160(bus_id=1)
    drive = DriveModule()

    drive.reset_encoders()

    keyboard_thread = threading.Thread(target=keyboard_loop)
    keyboard_thread.daemon = True
    keyboard_thread.start()

    try:
        while running:

            # --- SILNIKI ---
            if motor_power > 0:
                drive.forward(abs(motor_power))

            elif motor_power < 0:
                drive.backward(abs(motor_power))

            else:
                drive.stop()

            # --- ENKODERY ---
            e1, e2 = drive.get_encoder_steps()

            # --- IMU ---
            ax, ay, az = imu.read_acc()
            gx, gy, gz = imu.read_gyro()

            print(
                f"\n\n\rPOWER: {motor_power:.1f}"
            #    f"\nENC -> M1:{e1:6} M2:{e2:6}"
            #    f"\nACC -> X:{ax:6} Y:{ay:6} Z:{az:6}"
            #    f"\nGYR -> X:{gx:6} Y:{gy:6} Z:{gz:6}"
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nPrzerwano program.")

    finally:
        running = False
        drive.close()


if __name__ == "__main__":
    main()
