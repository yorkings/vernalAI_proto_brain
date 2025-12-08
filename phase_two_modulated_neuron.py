import torch
torch.set_default_dtype(torch.float32)
class ProtoNeuronMod:
    def __init__(self, input_size: int, learning_rate: float = 0.01,decay: float = 0.9, init_scale: float = 0.1,elig_decay: float = 0.9, local_lr: float = 0.01):
        self.input_size = input_size
        self.learning_rate = learning_rate  # for DA updates
        self.local_lr = local_lr            # for local Hebbian updates
        self.gamma = decay
        # Weights & bias
        self.weights = ((torch.randn(input_size,dtype=torch.float32) * 2.0) - 1.0) * init_scale
        self.bias = ((torch.randn(1,dtype=torch.float32) * 2.0) - 1.0) * init_scale 
        # State
        self.trace = torch.tensor(0.0)
        self.last_input = torch.zeros(input_size)
        self.last_activation = torch.tensor(0.0)
        
        # Eligibility trace (same shape as weights)
        self.eligibility = torch.zeros_like(self.weights)
        self.elig_decay = elig_decay
        
        # Plasticity gate (can be modulated)
        self.plasticity_gate = 1.0

    def forward(self,x:torch.tensor):
        if x.shape[0] != self.input_size:
            raise ValueError("Input size mismatch")
        self.last_input = x.clone()
        z=torch.dot(self.weights, x) + self.bias.squeeze()
        a=torch.tanh(z)
        self.trace=self.gamma *self.trace + a
        self.last_activation = a
        # Update eligibility trace: e ← λ_e * e + a * x
        self.eligibility = self.elig_decay * self.eligibility + (a * x) 
        return a

    def learn(self, target=None, delta=None, gate: float = 1.0, mix_local: float = 0.5): 
        """
        Apply learning with dopamine modulation.
        
        Args:
            target: Local target for supervised learning
            delta: Dopamine signal (RPE)
            gate: Plasticity gate multiplier
            mix_local: Blend between local (1) and DA (0) learning
        """
        if target is not None:
            if not isinstance(target, torch.Tensor):
                target = torch.tensor([float(target)])
            a = self.last_activation
            error = target - a
            delta_w_local = self.local_lr * a * error * self.last_input
            delta_b_local = self.local_lr * error
        else:
            delta_w_local = torch.zeros_like(self.weights)
            delta_b_local = torch.tensor([0.0])
        if delta is not None:
            # Three-factor rule: α * gate * delta * eligibility
            delta_w_da = self.learning_rate * (delta * self.eligibility) * (gate * self.plasticity_gate)
            delta_b_da = self.learning_rate * delta * (gate * self.plasticity_gate)
        else:
            delta_w_da = torch.zeros_like(self.weights)
            delta_b_da = torch.tensor([0.0])
        
        # Blend updates
        w_update = (mix_local * delta_w_local) + ((1 - mix_local) * delta_w_da)
        b_update = (mix_local * delta_b_local) + ((1 - mix_local) * delta_b_da)
        
        # Apply updates
        self.weights += w_update
        self.bias += b_update
    
    def reset_eligibility(self):
        """Reset eligibility trace."""
        self.eligibility = torch.zeros_like(self.weights)     

# Quick test
if __name__ == "__main__":
    print("Quick test of ProtoNeuronMod...")
    
    # Create neuron
    neuron = ProtoNeuronMod(input_size=5)
    
    # Test forward
    x = torch.randn(5)
    activation = neuron.forward(x)
    
    print(f"Activation type: {type(activation)}")
    print(f"Activation value: {activation}")
    print(f"Is tensor: {isinstance(activation, torch.Tensor)}")
    
    # Test learning
    neuron.learn(target=0.5, delta=0.2)
    print("Learning successful!")
