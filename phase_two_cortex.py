import torch
from phase_two_column import ProtoColumn
from typing import List,Callable
from phase_two_GWI import GlobalWorkspace

class CorticalBlock:
    def __init__(self ,columns: List[ProtoColumn],input_proj: bool = False,rho: float = 0.2, eta: float = 1e-3,weight_decay: float = 1e-4, init_scale: float = 0.02,
        reduce_fn: Callable = lambda v: v.mean(),elig_decay: float = 0.9, gwi_threshold: float = 0.6):
        self.columns = columns
        self.C = len(columns)
        self.reduce_fn = reduce_fn  # how to reduce column vector -> scalar
        self.rho = rho
        self.eta = eta
        self.weight_decay = weight_decay

        # Get hidden dimension from first column
        self.hidden_dim = columns[0].layer.num_neurons if hasattr(columns[0].layer, 'num_neurons') else 5
         # Initialize Global Workspace
        self.workspace = GlobalWorkspace(hidden_dim=self.hidden_dim,num_columns=self.C,threshold=gwi_threshold)

        # recurrent state (pre-activation)
        self.z = torch.zeros(self.C)
        # inter-column weights W (C x C)
        self.W = (torch.randn(self.C, self.C) * init_scale)
        # optional input projection (B matrix)
        self.use_input_proj = input_proj
        if self.use_input_proj:
            # assume all columns share same input dim; use first column's layer input size
            in_dim = columns[0].layer.input_size
            self.B = (torch.randn(self.C, in_dim) * init_scale)
        else:
            self.B = None
         # Block-level eligibility
        self.eligibility = torch.zeros_like(self.W)
        self.elig_decay = elig_decay
        
        # Block-level plasticity gate
        self.plasticity_gate = 1.0
         # State cache
        self.last_o = None
        self.last_raw_o = None    

    def forward(self,x=None):
        """
            1. collect column scalar outputs o_t
            2. update z_t = (1-rho) z_{t-1} + rho (W o_t + B x)
            3. output = normalize(z_t)
            4. return block outputs
        """
        #summary
        o=torch.zeros(self.C)
        column_outputs = []
        column_traces = []

        for i,col in enumerate(self.columns):
            col_vec=col.last_output
            o[i]=float(self.reduce_fn(col_vec))
        # 2. recurrent pre-activation
        recurrent_input = self.W.matmul(o)  # shape (C,)
        input_proj = torch.zeros(self.C)
        if self.use_input_proj and x is not None:
            input_proj = self.B.matmul(x)  # shape (C,)

        self.z = (1.0 - self.rho) * self.z + self.rho * (recurrent_input + input_proj)    
        out=self.z/(1+torch.abs(self.z))
         # Update block eligibility (outer product)
        self.eligibility = self.elig_decay * self.eligibility + torch.outer(out, out)

        self.last_o = out.clone()
        self.last_raw_o = o.clone()

        # Check for Global Workspace Ignition using column outputs
        if column_outputs:
            gwi_result = self.workspace(column_outputs, column_traces)
            self.last_ignition = gwi_result
        
        return out
    
    def learn(self,delta=None, gate: float = 1.0, mix_local: float = 0.5):
        """
        Update inter-column W using Oja's rule:
        ΔW_ij = eta * (o_i * o_j - o_i^2 * W_ij) - weight_decay * W_ij
        """
        effective_gate = gate * self.plasticity_gate

        o = self.last_o
        # compute outer product o o^T
        outer = o.unsqueeze(1) * o.unsqueeze(0)  # C x C
        o_sq = o * o  # C
        # Oja update per element
        # ΔW = eta * (outer - diag(o_sq) @ W)  (we expand diag multiplication manually)
        # Equivalent elementwise:
        o_sq_mat = o_sq.unsqueeze(1)  # C x 1
        oja_term = outer - o_sq_mat * self.W  # broadcasting

        # DA-modulated update
        if delta is not None:
            da_term = delta * self.eligibility * effective_gate
            deltaW = self.eta * (oja_term + da_term) - self.weight_decay * self.W
        else:
            deltaW = self.eta * oja_term - self.weight_decay * self.W
        
        self.W += deltaW

        # optional normalization to stabilize rows
        row_norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        self.W = self.W / row_norms
        # Pass learning to columns
        for col in self.columns:
            col.learn(delta=delta, gate=effective_gate, mix_local=mix_local)

    def reset_eligibility(self):
        """Reset block eligibility trace."""
        self.eligibility = torch.zeros_like(self.W)
        
if __name__ == "__main__":
    print("Testing CorticalBlock with Dopamine Modulation...")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Create test columns
    input_dim = 8
    num_columns = 4
    neurons_per_column = 5
    
    columns = []
    for i in range(num_columns):
        col = ProtoColumn(input_size=input_dim,num_neurons=neurons_per_column,lr=0.01,elig_decay=0.9)
        columns.append(col)
    
    # Test 1: Basic initialization
    print("\n1. Testing initialization...")
    
    block = CorticalBlock( columns=columns,input_proj=True,rho=0.3, eta=0.01,weight_decay=0.001,elig_decay=0.9)
    
    print(f"Block created with {block.C} columns")
    print(f"Weight matrix W shape: {block.W.shape}")
    print(f"Input projection B shape: {block.B.shape if block.B is not None else 'None'}")
    print(f"Eligibility matrix shape: {block.eligibility.shape}")
    print(f"Recurrent state z shape: {block.z.shape}")
    print(f"Rho (leak rate): {block.rho}")
    print(f"Eta (learning rate): {block.eta}")
    
    # Test 2: Forward pass with input projection
    print("\n2. Testing forward pass with input projection...")
    
    # First activate all columns
    x = torch.randn(input_dim)
    for col in columns:
        col.forward(x)
    
    # Block forward pass
    block_output = block.forward(x=x)
    
    print(f"Input shape: {x.shape}")
    print(f"Block output shape: {block_output.shape}")
    print(f"Block output range: [{block_output.min():.4f}, {block_output.max():.4f}]")
    print(f"All outputs in [-1, 1]: {torch.all(block_output >= -1) and torch.all(block_output <= 1)}")
    print(f"Last_o stored: {block.last_o is not None}")
    print(f"Last_raw_o stored: {block.last_raw_o is not None}")
    
    # Test 3: Forward pass without input projection
    print("\n3. Testing forward pass without input projection...")
    
    block_no_proj = CorticalBlock(columns=columns,input_proj=False,rho=0.3)
    
    # Re-activate columns with new input
    x2 = torch.randn(input_dim)
    for col in columns:
        col.forward(x2)
    
    output_no_proj = block_no_proj.forward()
    
    print(f"Output shape (no projection): {output_no_proj.shape}")
    print(f"Output mean: {output_no_proj.mean():.4f}")
    
    # Test 4: Learning without dopamine (local Oja only)
    print("\n4. Testing learning without dopamine...")
    
    block_local = CorticalBlock(columns=columns,input_proj=False,eta=0.02)
    # Store initial weights
    W_before = block_local.W.clone()
    # Forward pass
    x_local = torch.randn(input_dim)
    for col in columns:
        col.forward(x_local)
    block_local.forward()
    # Learn without dopamine
    block_local.learn(delta=None, gate=1.0, mix_local=1.0)  
    W_after = block_local.W.clone()
    weight_change = torch.mean(torch.abs(W_after - W_before)).item()
    
    print(f"Weight matrix changed: {weight_change > 1e-6}")
    print(f"Average weight change: {weight_change:.6f}")
    
    # Check row normalization
    row_norms = torch.norm(block_local.W, dim=1)
    print(f"Row norms after learning: {row_norms}")
    print(f"All rows normalized to ~1: {torch.allclose(row_norms, torch.ones_like(row_norms), rtol=0.01)}")
    
    # Test 5: Learning with dopamine
    print("\n5. Testing learning with dopamine modulation...")
    
    block_da = CorticalBlock(columns=columns,input_proj=False,eta=0.01,elig_decay=0.9 )
    
    # Store initial weights
    W_before_da = block_da.W.clone()
    eligibility_before = block_da.eligibility.clone()
    
    # Forward pass to build eligibility
    x_da = torch.randn(input_dim)
    for col in columns:
        col.forward(x_da)
    block_da.forward()
    
    # Learn with dopamine
    delta = 0.3  # Positive dopamine
    block_da.learn(delta=delta, gate=1.0, mix_local=0.0)
    
    W_after_da = block_da.W.clone()
    eligibility_after = block_da.eligibility.clone()
    
    weight_change_da = torch.mean(torch.abs(W_after_da - W_before_da)).item()
    eligibility_change = torch.mean(torch.abs(eligibility_after - eligibility_before)).item()
    
    print(f"Dopamine learning weight change: {weight_change_da:.6f}")
    print(f"Eligibility matrix changed: {eligibility_change > 0}")
    print(f"Eligibility decay working: {torch.all(eligibility_after >= 0)}")
    
    # Test 6: Mixed learning
    print("\n6. Testing mixed learning...")
    
    block_mixed = CorticalBlock(columns=columns,input_proj=True,eta=0.015)
    W_before_mixed = block_mixed.W.clone()
    
    # Forward pass
    x_mixed = torch.randn(input_dim)
    for col in columns:
        col.forward(x_mixed)
    block_mixed.forward(x=x_mixed)
    
    # Mixed learning (50% local, 50% DA)
    block_mixed.learn(delta=0.2, gate=0.8, mix_local=0.5)
    
    W_after_mixed = block_mixed.W.clone()
    weight_change_mixed = torch.mean(torch.abs(W_after_mixed - W_before_mixed)).item()
    
    print(f"Mixed learning weight change: {weight_change_mixed:.6f}")
    
    # Test 7: Plasticity gate effects
    print("\n7. Testing plasticity gate effects...")
    
    block_gated = CorticalBlock(columns=columns,input_proj=False)
    block_gated.plasticity_gate = 0.5  # Half plasticity
    W_before_gated = block_gated.W.clone()
    
    # Forward
    x_gated = torch.randn(input_dim)
    for col in columns:
        col.forward(x_gated)
    block_gated.forward()
    
    # Learn with external gate
    block_gated.learn(delta=0.1, gate=0.5, mix_local=0.5)
    
    W_after_gated = block_gated.W.clone()
    weight_change_gated = torch.mean(torch.abs(W_after_gated - W_before_gated)).item()
    
    print(f"Effective gate = 0.5 * 0.5 = 0.25")
    print(f"Gated learning weight change: {weight_change_gated:.6f}")
    
    # Test 8: Different reduction functions
    print("\n8. Testing different reduction functions...")
    
    reduction_functions = [
        ("mean", lambda v: v.mean()),
        ("max", lambda v: v.max()),
        ("norm", lambda v: torch.norm(v)),
        ("sparse_mean", lambda v: torch.relu(v).mean())
    ]
    
    for name, reduce_fn in reduction_functions:
        block_reduce = CorticalBlock(columns=columns,input_proj=False,reduce_fn=reduce_fn)   
        # Forward
        x_reduce = torch.randn(input_dim)
        for col in columns:
            col.forward(x_reduce)
        
        output = block_reduce.forward()
        print(f"Reduction '{name}': output mean={output.mean():.4f}, std={output.std():.4f}")
    
    # Test 9: Eligibility reset
    print("\n9. Testing eligibility trace reset...")
    
    block_reset = CorticalBlock(
        columns=columns,
        input_proj=False,
        elig_decay=0.9
    )
    
    # Build up eligibility
    for _ in range(3):
        x_reset = torch.randn(input_dim)
        for col in columns:
            col.forward(x_reset)
        block_reset.forward()
    
    eligibility_before_reset = block_reset.eligibility.clone()
    print(f"Eligibility before reset: mean={eligibility_before_reset.mean():.6f}")
    
    # Reset
    block_reset.reset_eligibility()
    eligibility_after_reset = block_reset.eligibility.clone()
    
    print(f"Eligibility after reset: mean={eligibility_after_reset.mean():.6f}")
    print(f"Eligibility reset to zero: {torch.allclose(eligibility_after_reset, torch.zeros_like(eligibility_after_reset), atol=1e-6)}")
    
    # Test 10: Temporal sequence processing
    print("\n10. Testing temporal sequence processing...")
    
    block_seq = CorticalBlock(
        columns=columns,
        input_proj=True,
        rho=0.5  # Faster dynamics
    )
    
    sequence_length = 5
    sequence_outputs = []
    
    for step in range(sequence_length):
        x_seq = torch.randn(input_dim)
        
        # Activate columns
        for col in columns:
            col.forward(x_seq)
        
        # Block forward
        output = block_seq.forward(x=x_seq)
        sequence_outputs.append(output.detach().numpy())
        
        # Learn with varying dopamine
        if step % 2 == 0:
            block_seq.learn(delta=0.1, mix_local=0.3)
        else:
            block_seq.learn(delta=-0.05, mix_local=0.7)
        
        print(f"Step {step+1}: output mean={output.mean():.4f}, z-norm={torch.norm(block_seq.z):.4f}")
    
    print(f"Processed sequence of {sequence_length} steps")
    
    # Test 11: Error handling and state checking
    print("\n11. Testing error handling...")
    
    try:
        # Try to learn without forward pass
        block_error = CorticalBlock(columns=columns, input_proj=False)
        block_error.learn()
        print("✗ Should have errored - no forward pass before learn")
    except Exception as e:
        print(f"✓ Correctly requires forward before learn")
    
    # Test with proper sequence
    try:
        block_good = CorticalBlock(columns=columns, input_proj=False)
        for col in columns:
            col.forward(torch.randn(input_dim))
        block_good.forward()
        block_good.learn(delta=0.1)
        print("✓ Proper forward→learn sequence works")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test 12: Weight matrix properties
    print("\n12. Testing weight matrix properties...")
    
    block_final = CorticalBlock(
        columns=columns,
        input_proj=False,
        init_scale=0.1
    )
    
    # Analyze initial W
    W_initial = block_final.W
    print(f"Initial W stats:")
    print(f"  Mean: {torch.mean(W_initial):.4f}")
    print(f"  Std: {torch.std(W_initial):.4f}")
    print(f"  Min: {torch.min(W_initial):.4f}")
    print(f"  Max: {torch.max(W_initial):.4f}")
    print(f"  Positive fraction: {(W_initial > 0).float().mean().item():.1%}")
    
    # After learning
    for col in columns:
        col.forward(torch.randn(input_dim))
    block_final.forward()
    block_final.learn(delta=0.15)
    
    W_final = block_final.W
    print(f"\nFinal W stats after learning:")
    print(f"  Mean: {torch.mean(W_final):.4f}")
    print(f"  Std: {torch.std(W_final):.4f}")
    
    print("\n" + "=" * 60)
    print("All CorticalBlock tests completed!")
    print("=" * 60)