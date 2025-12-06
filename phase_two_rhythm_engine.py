

class RhythmEngine:
    """
    Simple discretized theta/gamma rhythm generator.
    """
    def __init__(self, sim_steps_per_second: int = 60,
                 theta_freq: float = 4.0, gamma_per_theta: int = 8,
                 theta_phase_window: tuple = (0.1, 0.2)):
        self.sim_steps_per_second = sim_steps_per_second
        self.theta_freq = theta_freq
        self.gamma_per_theta = gamma_per_theta
        self.steps_per_theta = max(1, int(sim_steps_per_second / theta_freq))
        self.gamma_period = max(1, self.steps_per_theta // gamma_per_theta)
        self.theta_phase_window = theta_phase_window
        self.t = 0

    def tick(self, steps: int = 1):
        self.t += steps

    def theta_in_window(self):
        pos = (self.t % self.steps_per_theta) / self.steps_per_theta
        return (pos >= self.theta_phase_window[0] and 
                pos <= self.theta_phase_window[1])

    def get_gates_for_column(self, preferred_bin: int):
        gb = self.gamma_bin()
        gamma_gate = (gb == preferred_bin)
        theta_gate = self.theta_in_window()
        return gb, gamma_gate, theta_gate

    def gamma_bin(self):
        pos = self.t % self.steps_per_theta
        return int(pos // self.gamma_period)