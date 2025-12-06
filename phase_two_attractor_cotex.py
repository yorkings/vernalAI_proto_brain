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
        self.last_raw_o = torch.tensor([self.reduce_fn(col.last_output) 
                                       for col in self.columns])
        
        return o
    
    #the three fuctions are for providing stability
    def homeostasis_update(self, o):
        """
        o = stable column activations (C-dim)
        """
        # Update exponential moving average
        self.col_activation_avg = (
            0.99 * self.col_activation_avg + self.homeo_lr * o
        )

        # deviation from desired firing level
        delta = self.activation_target - self.col_activation_avg

        # shift recurrent weights
        self.W += self.homeo_lr * delta.unsqueeze(1)

    def synaptic_scaling(self):
        """
        Scale each row of W to maintain constant norm.
        """
        norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        target = self.synaptic_scaling_target
        self.W *= (target / norms)

    def normalize_recurrent_weights(self):
        """
        Keep network stable.
        """
        # row normalize (keeps dynamics bounded)
        norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        self.W = self.W / norms

    def apply_sparsity(self, o: torch.Tensor) -> torch.Tensor:
        """
        k-Winner-Take-All (hard or soft) for sparse coding.
        """
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
    print("Testing AttractorCortex...")
    print("-" * 50)
    
    # Create test columns
    input_dim = 8
    num_columns = 4
    neurons_per_column = 5
    
    columns = []
    for i in range(num_columns):
        col = ProtoColumn(input_size=input_dim, num_neurons=neurons_per_column)
        columns.append(col)
    
    # Test 1: Basic initialization
    print("\n1. Testing initialization...")
    cortex = AttractorCortex(
        columns=columns,
        input_proj=True,
        settling_steps=10,
        energy_weight=0.01,
        activation_target=0.2,
        homeo_lr=0.01,
        synaptic_scaling_target=1.0
    )
    
    print(f"Created cortex with {cortex.C} columns")
    print(f"W shape: {cortex.W.shape}")
    print(f"B shape: {cortex.B.shape if cortex.B is not None else 'None'}")
    print(f"Settling steps: {cortex.settling_steps}")
    print(f"Activation target: {cortex.activation_target}")
    
    # Test 2: Forward pass with attractor dynamics
    print("\n2. Testing forward pass with settling...")
    x = torch.randn(input_dim)
    
    # First activate columns
    for col in columns:
        col.forward(x)
    
    # Attractor forward
    output = cortex.forward(x=x)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    # Check convergence info
    info = cortex.get_convergence_info()
    print(f"Converged: {info['converged']}")
    print(f"Steps taken: {info['steps']}")
    print(f"Final difference: {info['final_diff']:.6f}")
    print(f"Energy: {info['energy']:.6f}")
    
    # Test 3: Learning with stabilization
    print("\n3. Testing learning with stabilization...")
    
    # Store initial weights
    W_before = cortex.W.clone()
    activation_avg_before = cortex.col_activation_avg.clone()
    
    # Learn
    cortex.learn()
    
    # Check changes
    W_after = cortex.W.clone()
    activation_avg_after = cortex.col_activation_avg.clone()
    
    W_change = torch.mean(torch.abs(W_after - W_before)).item()
    activation_change = torch.mean(torch.abs(activation_avg_after - activation_avg_before)).item()
    
    print(f"Weight change: {W_change:.6f}")
    print(f"Activation avg change: {activation_change:.6f}")
    
    # Check weight norms
    norms = torch.norm(cortex.W, dim=1)
    print(f"Row norms after learning: {norms}")
    
    # Test 4: Multiple forward passes to test homeostasis
    print("\n4. Testing homeostasis over multiple steps...")
    
    cortex2 = AttractorCortex(
        columns=columns,
        input_proj=False,
        activation_target=0.3,
        homeo_lr=0.05
    )
    
    activations_over_time = []
    for step in range(5):
        x_step = torch.randn(input_dim)
        
        # Activate columns
        for col in columns:
            col.forward(x_step)
        
        # Forward through cortex
        output = cortex2.forward()
        activations_over_time.append(output.detach().numpy())
        
        # Learn every step
        cortex2.learn()
        
        # Check activation average
        if step == 0:
            print(f"Step {step+1}: Activation avg = {cortex2.col_activation_avg}")
    
    print(f"Final activation average: {cortex2.col_activation_avg}")
    
    # Test 5: Energy computation
    print("\n5. Testing energy computation...")
    
    # Random state energy
    random_state = torch.randn(num_columns)
    random_energy = cortex.compute_energy(random_state)
    print(f"Random state energy: {random_energy:.6f}")
    
    # Stable state energy
    if cortex.energy_value is not None:
        print(f"Stable state energy: {cortex.energy_value:.6f}")
        print(f"Stable state lower energy: {cortex.energy_value < random_energy}")
    
    # Test 6: Different settling parameters
    print("\n6. Testing different settling parameters...")
    
    # Fast settling
    fast_cortex = AttractorCortex(
        columns=columns,
        input_proj=False,
        settling_steps=3,
        convergence_thresh=0.1
    )
    
    # Slow settling
    slow_cortex = AttractorCortex(
        columns=columns,
        input_proj=False,
        settling_steps=20,
        convergence_thresh=1e-6
    )
    
    # Test both
    for col in columns:
        col.forward(x)
    
    fast_output = fast_cortex.forward()
    fast_info = fast_cortex.get_convergence_info()
    
    for col in columns:
        col.forward(x)
    
    slow_output = slow_cortex.forward()
    slow_info = slow_cortex.get_convergence_info()
    
    print(f"Fast cortex: {fast_info['steps']} steps, converged={fast_info['converged']}")
    print(f"Slow cortex: {slow_info['steps']} steps, converged={slow_info['converged']}")
    
    # Test 7: Synaptic scaling
    print("\n7. Testing synaptic scaling...")
    
    # Create cortex with different scaling target
    scaled_cortex = AttractorCortex(
        columns=columns,
        input_proj=False,
        synaptic_scaling_target=2.0
    )
    
    # Do a forward and learn
    for col in columns:
        col.forward(x)
    
    scaled_output = scaled_cortex.forward()
    scaled_cortex.learn()
    
    # Check norms
    scaled_norms = torch.norm(scaled_cortex.W, dim=1)
    print(f"Target norm: {scaled_cortex.synaptic_scaling_target}")
    print(f"Actual row norms: {scaled_norms}")
    print(f"Close to target: {torch.allclose(scaled_norms, torch.ones_like(scaled_norms)*2.0, rtol=0.1)}")
    
    # Test 8: Error handling
    print("\n8. Testing error handling...")
    
    try:
        # Try to learn without forward pass
        bad_cortex = AttractorCortex(columns=columns, input_proj=False)
        bad_cortex.learn()
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    try:
        # Try forward without activating columns
        bad_cortex2 = AttractorCortex(columns=columns, input_proj=False)
        bad_cortex2.forward()
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("\n" + "=" * 50)
    print("All AttractorCortex tests completed!")
    print("=" * 50)
