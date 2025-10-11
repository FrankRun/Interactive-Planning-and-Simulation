from collections import deque
import numpy as np
from config import cfg


class PID:
    """
    PID controller
    """

    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0):
        """
        :param vehicle: actor to apply to local planner logic onto
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self.dt = 0.1
        self._e_buffer = deque(maxlen=10)

    def reset(self):
        self._e_buffer = deque(maxlen=10)

    def run_step(self, current_value, target_value):

        return self._pid_control(target_value, current_value)

    def _pid_control(self, target_value, current_value):
        """
        Estimate the throttle of the vehicle based on the PID equations

        :return: throttle control in the range [0, 1]
        """
        _e = (target_value - current_value)
        self._e_buffer.append(_e)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.dt
            _ie = sum(self._e_buffer) * self.dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _e) + (self._K_D * _de ) + (self._K_I * _ie ), -1.0, 1.0)