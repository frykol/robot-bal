from hardware.accelerometer import BMI160
from hardware.drive_module import DriveModule
import time

def main():
    imu = BMI160(bus_id=1)
    drive = DriveModule()



    try:
        drive.reset_encoders()

        print("--- Jazda do przodu ---")
        drive.forward(0.5)

        for _ in range(20):
            e1, e2 = drive.get_encoder_steps()
            print(f"Silnik 1: {e1} impulsów | Silnik 2: {e2} impulsów")
            time.sleep(0.1)

        print("--- Stop ---")
        drive.stop()
        time.sleep(1)

        print("--- Jazda do tyłu ---")
        drive.backward(0.5)

        for _ in range(20):
            e1, e2 = drive.get_encoder_steps()
            print(f"Silnik 1: {e1} impulsów | Silnik 2: {e2} impulsów")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Przerwano program.")

    finally:
        drive.close()

    try:
        while True:
            ax, ay, az = imu.read_acc()
            gx, gy, gz = imu.read_gyro()

            print(
                f"ACC -> X:{ax:6} Y:{ay:6} Z:{az:6} | "
                f"GYRO -> X:{gx:6} Y:{gy:6} Z:{gz:6}"
            )

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Przerwano program.")

if __name__ == "__main__":
    main()
