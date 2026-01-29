import torch
from collections import deque

class ActionMomentum:
    def __init__(self, num_actions=3, lambda_m=0.85, m_scale=0.5):
        self.momentum = torch.zeros(num_actions)
        self.lambda_m = lambda_m
        self.m_scale = m_scale
        
    def update(self, chosen_action):
        # Decay and update
        self.momentum = self.lambda_m * self.momentum
        self.momentum[chosen_action] += (1 - self.lambda_m)
        
    def get_momentum_bonus(self, action_idx):
        return self.m_scale * self.momentum[action_idx]


class LoopDetector:
    def __init__(self, registry_size=8, loop_penalty=0.5):
        self.registry = deque(maxlen=registry_size)
        self.loop_penalty = loop_penalty
        
    def compute_fingerprint(self, predicted_states, ndigits=4):
        # Simple mean-based fingerprint
        if not predicted_states:
            return 0
        mean_vec = torch.mean(torch.stack(predicted_states), dim=0)
        # Round and hash
        rounded = torch.round(mean_vec * (10**ndigits)) / (10**ndigits)
        fp_hash = hash(tuple(rounded.numpy().flatten()))
        return fp_hash
    
    def check_loop(self, predicted_states):
        fp = self.compute_fingerprint(predicted_states)
        is_loop = fp in self.registry
        self.registry.append(fp)
        return is_loop
