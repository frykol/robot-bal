from smbus2 import SMBus
import time

BMI160_ADDR = 0x69

REG_CMD = 0x7E
REG_ACC_X_L = 0x12
REG_GYR_X_L = 0x0C

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

