import torch
from phase_two_column import ProtoColumn
from phase_two_cortex import CorticalBlock  # Import the CorticalBlock
from typing import List,Callable

class AttractorCortex(CorticalBlock):  # Extend CorticalBlock
    def __init__(self, columns: List[ProtoColumn],input_proj: bool = False,rho: float = 0.3,eta: float = 0.01, weight_decay: float = 1e-4,init_scale: float = 0.02, 
                 reduce_fn: Callable = lambda v: v.mean(), settling_steps: int = 5, energy_weight: float = 0.01, convergence_thresh: float = 1e-4):
        # Initialize parent CorticalBlock
        super().__init__(columns, input_proj, rho, eta, weight_decay,init_scale, reduce_fn)
        
        # Attractor-specific parameters
        self.settling_steps = settling_steps  # Max settling iterations
        self.energy_weight = energy_weight    # Lambda for energy computation
        self.convergence_thresh = convergence_thresh
        
        # State tracking for attractor dynamics
        self.stable_state = None  # Last converged state
        self.energy_value = None  # Last computed energy
        self.converged = False    # Whether last forward converged
        self.settling_history = []  # Track convergence per step
        
    def compute_energy(self, o: torch.Tensor) -> float:
        """
        Compute energy of a state o.
        Energy = -0.5 * (o^T @ W @ o) + energy_weight * sum(o^2)
        Lower energy = more stable state.
        """
        recurrent_energy = -0.5 * torch.dot(o, self.W @ o)
        regularization = self.energy_weight * torch.sum(o**2)
        return (recurrent_energy + regularization).item()
    
    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with attractor dynamics.  
        Steps:
        1. Update columns with input x (if provided)
        2. Start with initial column summaries
        3. Iteratively settle to attractor state
        4. Return converged state
        
        Returns converged state or best state after settling_steps.
        """
        # 1. Update columns if x provided
        if x is not None:
            for col in self.columns:
                col.forward(x)
        
        # 2. Get initial column summaries
        o = torch.zeros(self.C)
        for i, col in enumerate(self.columns):
            if col.last_output is None:
                # Need to compute column output first
                if x is None:
                    raise ValueError("Columns need input x or pre-computed outputs")
                col.forward(x)
            o[i] = float(self.reduce_fn(col.last_output))
        
        # Initialize recurrent state
        z = self.z.clone()
        prev_o = o.clone()
        
        # 3. Settling loop
        self.converged = False
        self.settling_history = []
        
        for step in range(self.settling_steps):
            # Compute recurrent input
            if self.use_input_proj and x is not None:
                input_proj = self.B.matmul(x)
            else:
                input_proj = torch.zeros(self.C)
            
            # Update recurrent state
            z = (1.0 - self.rho) * z + self.rho * (self.W @ o + input_proj)
            
            # Normalize
            new_o = z / (1 + torch.abs(z))
            
            # Check convergence
            diff = torch.norm(new_o - prev_o).item()
            self.settling_history.append(diff)
            
            if diff < self.convergence_thresh:
                self.converged = True
                o = new_o
                break
            
            prev_o = o
            o = new_o
        
        # 4. Save final state
        self.z = z
        self.stable_state = o.clone()
        self.energy_value = self.compute_energy(o)
        self.last_o = o.clone()
        self.last_raw_o = torch.tensor([self.reduce_fn(col.last_output) 
                                       for col in self.columns])
        
        return o
    
    def learn(self) -> None:
        """
        Learn using Oja's rule with energy regularization.
        
        Update rule: ΔW = η * (o o^T - energy_weight * W)
        """
        if self.stable_state is None:
            raise ValueError("No stable state to learn from. Call forward() first.")
        
        o = self.stable_state
        
        # Outer product for Hebbian term
        outer = o.unsqueeze(1) * o.unsqueeze(0)  # C x C
        
        # Oja update with energy regularization
        deltaW = self.eta * (outer - self.energy_weight * self.W)
        
        # Update weights
        self.W += deltaW
        
        # Normalize rows for stability
        row_norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        self.W = self.W / row_norms
    
    def get_convergence_info(self) -> dict:
        """Get information about last settling process."""
        if not self.settling_history:
            return {"converged": False, "steps": 0, "final_diff": None}
        
        return {
            "converged": self.converged,
            "steps": len(self.settling_history),
            "final_diff": self.settling_history[-1],
            "energy": self.energy_value
        }


# Test the AttractorCortex
if __name__ == "__main__":
    print("Testing AttractorCortex...")
    
    # Create test columns
    in_dim = 10
    n_cols = 4
    n_neurons = 5
    
    cols = []
    for i in range(n_cols):
        col = ProtoColumn(input_size=in_dim, num_neurons=n_neurons)
        cols.append(col)
    
    # Create attractor cortex
    cortex = AttractorCortex(
        columns=cols,
        input_proj=True,
        rho=0.3,
        eta=0.01,
        weight_decay=0.001,
        settling_steps=10,
        energy_weight=0.01
    )
    
    print(f"Cortex has {cortex.C} columns")
    print(f"W shape: {cortex.W.shape}")
    print(f"B shape: {cortex.B.shape if cortex.B is not None else 'None'}")
    
    # Test forward with settling
    x = torch.randn(in_dim)
    output = cortex.forward(x)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output: {output}")
    
    # Check convergence
    info = cortex.get_convergence_info()
    print(f"\nConvergence info:")
    print(f"  Converged: {info['converged']}")
    print(f"  Steps taken: {info['steps']}")
    print(f"  Final difference: {info['final_diff']:.6f}")
    print(f"  Energy: {info['energy']:.6f}")
    
    # Test learning
    old_W = cortex.W.clone()
    cortex.learn()
    new_W = cortex.W
    
    diff = torch.mean(torch.abs(new_W - old_W)).item()
    print(f"\nWeight change after learning: {diff:.6f}")
    
    # Test without input projection
    print("\n--- Testing without input projection ---")
    cortex2 = AttractorCortex(
        columns=cols,
        input_proj=False,
        settling_steps=5
    )
    
    # Need to activate columns first
    for col in cols:
        col.forward(x)
    
    output2 = cortex2.forward()
    info2 = cortex2.get_convergence_info()
    print(f"Output: {output2}")
    print(f"Converged in {info2['steps']} steps")
    
    # Test energy computation
    print("\n--- Testing energy computation ---")
    test_state = torch.randn(n_cols)
    energy = cortex.compute_energy(test_state)
    print(f"Energy of random state: {energy:.6f}")
    print(f"Energy of stable state: {cortex.energy_value:.6f}")
    
    print("\nAll tests passed!")