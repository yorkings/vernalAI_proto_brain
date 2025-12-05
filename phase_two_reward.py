class RewardModule:
    """
    Computes reward, value estimate, and dopamine-like RPE signal.
    Very small, local estimator for Phase 2.4.
    """
    def __init__(self, value_lr:float=0.01, baseline:float=0.0):
        self.V = float(baseline)        # expected value estimate
        self.value_lr = float(value_lr) # learning rate for V


    def step(self, reward: float) -> float:
        """
        Given observed reward, compute RPE delta and update V.
        Returns dopamine signal (RPE).
        t​=rt​−r^t​
        """
        r = float(reward)
        delta = r - self.V  # RPE
        self.V += self.value_lr * delta  # update value estimate
        return float(delta)
    def reset(self):
        """Reset value estimate to baseline."""
        self.V = 0.0 
        