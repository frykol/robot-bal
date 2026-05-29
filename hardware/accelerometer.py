from smbus2 import SMBus
import sys
import time

BMI160_ADDR = 0x69

REG_CHIP_ID = 0x00
CHIP_ID_EXPECTED = 0xD1

REG_CMD = 0x7E
REG_ACC_X_L = 0x12
REG_GYR_X_L = 0x0C
REG_GYR_RANGE = 0x43

# ±250 °/s — matches Bosch sensitivity used in runtime (see rl.imu_obs.GYR_LSB_PER_DPS).
from rl.imu_obs import GYR_LSB_PER_DPS

GYR_RANGE_250DPS = 0x03


def twos_complement(low, high):
    value = (high << 8) | low
    if value & 0x8000:
        value -= 1 << 16
    return value


class BMI160:
    def __init__(self, bus_id):
        self.bus = SMBus(bus_id)
        self.addr = BMI160_ADDR
        self.init_sensor()

    def init_sensor(self):
        self.bus.write_byte_data(self.addr, REG_CMD, 0xB6)
        time.sleep(0.1)

        self.bus.write_byte_data(self.addr, REG_CMD, 0x11)
        time.sleep(0.05)
        self.bus.write_byte_data(self.addr, REG_CMD, 0x15)
        time.sleep(0.05)

        self.bus.write_byte_data(self.addr, REG_GYR_RANGE, GYR_RANGE_250DPS)
        time.sleep(0.01)

        chip_id = self.bus.read_byte_data(self.addr, REG_CHIP_ID)
        if chip_id != CHIP_ID_EXPECTED:
            print(
                f"BMI160: unexpected CHIP_ID 0x{chip_id:02X} "
                f"(expected 0x{CHIP_ID_EXPECTED:02X})",
                file=sys.stderr,
            )

    def read_acc(self):
        data = self.bus.read_i2c_block_data(self.addr, REG_ACC_X_L, 6)
        ax = twos_complement(data[0], data[1])
        ay = twos_complement(data[2], data[3])
        az = twos_complement(data[4], data[5])
        return ax, ay, az

    def read_gyro(self):
        data = self.bus.read_i2c_block_data(self.addr, REG_GYR_X_L, 6)
        gx = twos_complement(data[0], data[1])
        gy = twos_complement(data[2], data[3])
        gz = twos_complement(data[4], data[5])
        return gx, gy, gz
