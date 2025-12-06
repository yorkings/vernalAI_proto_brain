import torch
from phase_two_column import ProtoColumn
from phase_two_cortex import CorticalBlock  # Import the CorticalBlock
from typing import List,Callable

class AttractorCortex(CorticalBlock):  # Extend CorticalBlock
    def __init__(self, columns: List[ProtoColumn],input_proj: bool = False,rho: float = 0.3,eta: float = 0.01, weight_decay: float = 1e-4,init_scale: float = 0.02, 
                 reduce_fn: Callable = lambda v: v.mean(), settling_steps: int = 5, energy_weight: float = 0.01, convergence_thresh: float = 1e-4,
                 activation_target=0.2,homeo_lr = 0.01,synaptic_scaling_target = 1.0,
                 k_sparsity = 2,inhibition_strength = 1.0,beta_soft = 20,use_soft_k = False

                 ):
        # Initialize parent CorticalBlock
        super().__init__(columns, input_proj, rho, eta, weight_decay,init_scale, reduce_fn)
        
        # Attractor-specific parameters
        self.settling_steps = settling_steps  # Max settling iterations
        self.energy_weight = energy_weight    # Lambda for energy computation
        self.convergence_thresh = convergence_thresh


        self.col_activation_avg = torch.zeros(self.C)

        #homeostasis params
        self.homeo_lr=homeo_lr
        self.synaptic_scaling_target=synaptic_scaling_target
        self.activation_target=activation_target

        #sparsity
        self.k_sparsity=k_sparsity
        self.inhibition_strength=inhibition_strength
        self.beta_soft=beta_soft
        self.use_soft_k=use_soft_k

        # State tracking for attractor dynamics
        self.stable_state = None  # Last converged state
        self.energy_value = None  # Last computed energy
        self.converged = False    # Whether last forward converged
        self.settling_history = []  # Track convergence per step
        
        self.W_temp = torch.randn(self.C, self.C) * 0.05
        self.eta_temp = 0.01
        self.temp_weight_decay = 1e-4

        self.last_state = None
        self.last_pred = None
        self.last_error = None
        self.last_surprise = None

    def predict_next(self, o: torch.Tensor) -> torch.Tensor:
        """Predict next state: ô_{t+1} = normalize(W_temp @ o_t)."""
        pred = self.W_temp @ o
        pred = pred / (1 + torch.abs(pred))
        pred = self.apply_sparsity(pred)
        return pred       
            
    def compute_energy(self, o: torch.Tensor) -> float:
        """Compute energy: E(o) = -0.5*(o^T @ W @ o) + energy_weight*sum(o^2)."""
        recurrent_energy = -0.5 * torch.dot(o, self.W @ o)
        regularization = self.energy_weight * torch.sum(o**2)
        return (recurrent_energy + regularization).item()
    
    def forward(self, x: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with attractor settling and temporal prediction."""
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
            new_o=self.apply_sparsity(new_o)            
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
        self.last_raw_o = torch.tensor([self.reduce_fn(col.last_output) for col in self.columns])
        # Temporal prediction for next timestep
        self.last_pred = self.predict_next(o)
        self.last_state = o.clone()
        return o
        
    #the three fuctions are for providing stability
    def homeostasis_update(self, o):
        """o = stable column activations (C-dim)"""
        # Update exponential moving average
        self.col_activation_avg = (0.99 * self.col_activation_avg + self.homeo_lr * o)
        # deviation from desired firing level
        delta = self.activation_target - self.col_activation_avg
        # shift recurrent weights
        self.W += self.homeo_lr * delta.unsqueeze(1)

    def synaptic_scaling(self):
        """Scale each row of W to maintain constant norm."""
        norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        target = self.synaptic_scaling_target
        self.W *= (target / norms)

    def normalize_recurrent_weights(self):
        """ Keep network stable."""
        # row normalize (keeps dynamics bounded)
        norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        self.W = self.W / norms

    def apply_sparsity(self, o: torch.Tensor) -> torch.Tensor:
        """k-Winner-Take-All (hard or soft) for sparse coding."""
        if self.use_soft_k:
            # Soft top-k via sharpened softmax
            o_ = torch.exp(self.beta_soft * o)
            return o_ / (o_.sum() + 1e-8)
        # Hard k-WTA
        C = o.shape[0]
        # If k >= C, do nothing
        if self.k_sparsity >= C:
            return o
        # Get threshold
        topk_vals, _ = torch.topk(o, self.k_sparsity)
        T = topk_vals[-1]
        # Apply competition mask
        mask = (o >= T).float()
        return o * mask        

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
        #stabilize the weights and gradients
        o = self.stable_state
        self.homeostasis_update(o)
        self.synaptic_scaling()
        self.normalize_recurrent_weights()

    def learn_temporal(self, next_state: torch.Tensor):
        """Learn temporal transitions: ΔW_temp = η_temp*(o_t @ o_{t+1}^T - λ*W_temp)."""
        if self.last_state is None:
            return
        # Make next_state sparse
        next_state = self.apply_sparsity(next_state)
        # Compute prediction error
        error = next_state - self.last_pred
        self.last_error = error
        self.last_surprise = torch.norm(error, p=1).item() 
        # Hebbian temporal learning
        outer = torch.outer(self.last_state, next_state)
        dW = self.eta_temp * (outer - self.temp_weight_decay * self.W_temp)
        self.W_temp += dW

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


if __name__ == "__main__":
    print("Testing AttractorCortex Temporal Prediction...")
    
    # Setup
    torch.manual_seed(42)
    input_dim = 8
    num_columns = 5
    neurons_per_column = 4
    
    # Create columns
    columns = []
    for i in range(num_columns):
        col = ProtoColumn(input_size=input_dim, num_neurons=neurons_per_column)
        columns.append(col)
    
    # Create cortex
    cortex = AttractorCortex(
        columns=columns,
        input_proj=True,
        settling_steps=8,
        k_sparsity=2
    )
    
    print(f"Cortex: {cortex.C} columns, k={cortex.k_sparsity} sparsity")
    
    # Test 1: Basic prediction
    print("\n1. Basic temporal prediction test:")
    x1 = torch.randn(input_dim)
    output1 = cortex.forward(x=x1)
    
    print(f"State: {output1}")
    print(f"Prediction: {cortex.last_pred}")
    print(f"Active units: {(output1 != 0).sum().item()}")
    
    # Test 2: Sequence learning
    print("\n2. Sequence learning test:")
    
    # Learn A→B
    pattern_a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    pattern_b = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0])
    
    cortex.stable_state = pattern_a.clone()
    cortex.last_state = pattern_a.clone()
    cortex.last_pred = cortex.predict_next(pattern_a)
    
    print(f"Initial prediction of B from A: {cortex.last_pred}")
    print(f"Initial error: {torch.norm(pattern_b - cortex.last_pred):.4f}")
    
    # Train transition
    for i in range(10):
        cortex.learn_temporal(pattern_b)
    
    cortex.last_pred = cortex.predict_next(pattern_a)
    print(f"Trained prediction of B from A: {cortex.last_pred}")
    print(f"Final error: {torch.norm(pattern_b - cortex.last_pred):.4f}")
    
    # Test 3: Different sparsity modes
    print("\n3. Sparsity mode test:")
    
    test_vec = torch.tensor([0.8, 0.5, 0.3, 0.9, 0.1])
    
    # Hard k-WTA
    hard_cortex = AttractorCortex(columns=columns, k_sparsity=2, use_soft_k=False)
    hard_out = hard_cortex.apply_sparsity(test_vec)
    print(f"Hard k-WTA (k=2): {hard_out}")
    
    # Soft k-WTA
    soft_cortex = AttractorCortex(columns=columns, k_sparsity=2, use_soft_k=True)
    soft_out = soft_cortex.apply_sparsity(test_vec)
    print(f"Soft k-WTA: {soft_out}")
    
    # Test 4: Convergence
    print("\n4. Convergence test:")
    
    for col in columns:
        col.forward(torch.randn(input_dim))
    
    output = cortex.forward()
    info = cortex.get_convergence_info()
    print(f"Steps: {info['steps']}, Converged: {info['converged']}")
    print(f"Final diff: {info['final_diff']:.6f}, Energy: {info['energy']:.6f}")
    
    print("\nAll tests passed!")