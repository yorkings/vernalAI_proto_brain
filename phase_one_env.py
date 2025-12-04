# phase1_life_loop.py
"""
Phase 1.3 - Life Loop v0
Minimal organism that senses a 1D world, acts, receives reward and learns locally.
Dependencies: torch
Run: python phase1_life_loop.py
"""

import torch
import random
import math
import csv
from typing import List

# -------------------------
# ProtoNeuron (Phase 1.1)
# -------------------------
class ProtoNeuron:
    def __init__(self, input_size:int, learning_rate:float=0.01, decay:float=0.9, init_scale:float=0.1):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.gamma = decay

        # weights and bias (1-D tensors)
        self.weights = ((torch.randn(input_size) * 2.0) - 1.0) * init_scale
        self.bias = ((torch.randn(1) * 2.0) - 1.0) * init_scale

        # internal state
        self.trace = torch.tensor(0.0)   # scalar trace
        self.activation_fn = torch.tanh  # phi(z)
        self.last_input = torch.zeros(input_size)
        self.last_activation = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        z = sum(w_i * x_i) + b
        a = phi(z)
        trace = gamma * trace + a
        Returns scalar activation a (tensor shape [])
        """
        if x.shape[0] != self.input_size:
            raise ValueError("Input size mismatch")

        # store last input for learning
        self.last_input = x.clone()

        z = torch.dot(self.weights, x) + self.bias.squeeze()
        a = self.activation_fn(z)

        # update trace (scalar)
        self.trace = self.gamma * self.trace + a

        # store for learn()
        self.last_activation = a
        return a

    def learn(self, target: torch.Tensor):
        """
        Local learning rule (Hebbian + error-driven):
        Δw_i = alpha * a * (t - a) * x_i
        Δb   = alpha * (t - a)
        target can be scalar tensor or float
        """
        a = self.last_activation
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(float(target))

        error = target - a  # local error (scalar)
        # corrected rule includes 'a'
        delta_w = self.learning_rate * a * error * self.last_input
        delta_b = self.learning_rate * error

        # apply updates in-place
        self.weights += delta_w
        self.bias += delta_b

# -------------------------
# ProtoLayer (Phase 1.2)
# -------------------------
class ProtoLayer:
    def __init__(self, input_size:int, num_neurons:int=5, learning_rate:float=0.01, feedback_strength:float=0.1):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.feedback_strength = feedback_strength

        self.neurons: List[ProtoNeuron] = [ProtoNeuron(input_size, learning_rate) for _ in range(num_neurons)]
        self.last_outputs = torch.zeros(num_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1. run all neurons in parallel on same x
        2. compute feedback = beta * mean(outputs)
        3. outputs += feedback
        4. stabilize: out = out / (1 + |out|)
        returns outputs tensor shape (num_neurons,)
        """
        if x.shape[0] != self.input_size:
            raise ValueError("Input size mismatch for layer")

        outputs = torch.zeros(self.num_neurons)
        for i, n in enumerate(self.neurons):
            outputs[i] = n.forward(x)

        mean_out = torch.mean(outputs)
        feedback = self.feedback_strength * mean_out
        outputs = outputs + feedback

        # stabilization (shunting-like)
        outputs = outputs / (1.0 + torch.abs(outputs))

        self.last_outputs = outputs.clone()
        return outputs

    def learn(self, target):
        # same scalar target for all neurons for this simple phase
        for n in self.neurons:
            n.learn(target)

# -------------------------
# Simple 1D Environment
# -------------------------
class World1D:
    """
    Simple 1D world with position p.
    Action a in {-1, 0, +1} moves position by step_size * a + small_noise.
    Goal at 0.0 (by default). Reward = -abs(distance_to_goal).
    """
    def __init__(self, start_pos:float=5.0, goal:float=0.0, step_size:float=0.5, noise_scale:float=0.02):
        self.pos = float(start_pos)
        self.goal = float(goal)
        self.step_size = float(step_size)
        self.noise_scale = float(noise_scale)

    def sense(self) -> torch.Tensor:
        # sensory vector: scalar position normalized (optional). Using 1D raw.
        return torch.tensor([self.pos], dtype=torch.float32)

    def step(self, action:int):
        # action should be -1, 0, or 1
        noise = random.gauss(0.0, self.noise_scale)
        self.pos += action * self.step_size + noise
        # return new state (for convenience)
        return self.pos

    def reward(self):
        # negative distance to goal (higher is better since less negative)
        return -abs(self.pos - self.goal)

    def reset(self, start_pos:float=None):
        if start_pos is None:
            # randomize start
            self.pos = random.uniform(-5.0, 5.0)
        else:
            self.pos = float(start_pos)

# -------------------------
# Agent (action mapping + life loop helpers)
# -------------------------
class SimpleAgent:
    """
    Uses ProtoLayer to pick actions.
    Action mapping: take first neuron output sign -> action in {-1,0,+1}
    """
    def __init__(self, layer:ProtoLayer):
        self.layer = layer

    def choose_action(self, layer_outputs: torch.Tensor) -> int:
        # primitive policy: use first neuron's output sign
        if layer_outputs.shape[0] == 0:
            return 0
        v = float(layer_outputs[0].item())
        if v > 0.2:
            return 1
        elif v < -0.2:
            return -1
        else:
            return 0

# -------------------------
# Training / Life loop
# -------------------------
def run_life_loop(num_steps:int = 2000, log_every:int = 50):
    # components
    env = World1D(start_pos=5.0, goal=0.0, step_size=0.3, noise_scale=0.02)
    layer = ProtoLayer(input_size=1, num_neurons=6, learning_rate=0.02, feedback_strength=0.08)
    agent = SimpleAgent(layer)

    # logging
    logs = []
    for t in range(num_steps):
        x = env.sense()                 # sense
        h = layer.forward(x)            # process
        a = agent.choose_action(h)      # act
        env.step(a)                     # world update
        r = env.reward()                # reward

        # Use reward as local target (very simple)
        # Option: scale reward into [-1,1] for stability
        target = torch.tensor([float(r)], dtype=torch.float32)
        layer.learn(target)             # local learning

        # logging
        logs.append((t, env.pos, float(r), a, h.numpy().tolist()))

        if (t % log_every) == 0 or t == num_steps - 1:
            print(f"t={t:04d} pos={env.pos:+.3f} reward={r:+.3f} action={a} mean_h={float(h.mean()):+.4f}")

    # save logs to CSV for inspection
    with open("phase1_life_loop_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t","pos","reward","action","layer_outputs"])
        for row in logs:
            writer.writerow([row[0], row[1], row[2], row[3], row[4]])

    print("Run finished. Logs written to phase1_life_loop_log.csv")

if __name__ == "__main__":
    run_life_loop(num_steps=2000, log_every=100)
