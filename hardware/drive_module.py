from gpiozero import Motor, RotaryEncoder


class DriveModule:
    def __init__(
        self,
        motor1_pins=(20, 21),
        motor2_pins=(23, 24),
        encoder1_pins=(5, 6),
        encoder2_pins=(13, 19),
    ):
        self.motor1 = Motor(forward=motor1_pins[0], backward=motor1_pins[1])
        self.motor2 = Motor(forward=motor2_pins[0], backward=motor2_pins[1])

        self.encoder1 = RotaryEncoder(
            a=encoder1_pins[0],
            b=encoder1_pins[1],
            max_steps=0,
        )
        self.encoder2 = RotaryEncoder(
            a=encoder2_pins[0],
            b=encoder2_pins[1],
            max_steps=0,
        )

    def reset_encoders(self):
        self.encoder1.steps = 0
        self.encoder2.steps = 0

    def forward(self, speed=0.5):
        self.motor1.forward(speed)
        self.motor2.forward(speed)

    def backward(self, speed=0.5):
        self.motor1.backward(speed)
        self.motor2.backward(speed)

    def stop(self):
        self.motor1.stop()
        self.motor2.stop()

    def get_encoder_steps(self):
        return self.encoder1.steps, self.encoder2.steps

    def close(self):
        self.stop()

        self.motor1.close()
        self.motor2.close()

        self.encoder1.close()
        self.encoder2.close()
