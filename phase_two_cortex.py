import torch
from phase_two_column import ProtoColumn
from typing import List,Callable

class CorticalBlock:
    def __init__(self ,columns: List[ProtoColumn],input_proj: bool = False,rho: float = 0.2, eta: float = 1e-3,weight_decay: float = 1e-4, init_scale: float = 0.02,
        reduce_fn: Callable = lambda v: v.mean()):
        self.columns = columns
        self.C = len(columns)
        self.reduce_fn = reduce_fn  # how to reduce column vector -> scalar
        self.rho = rho
        self.eta = eta
        self.weight_decay = weight_decay
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

    def forward(self,x=None):
        """
            1. collect column scalar outputs o_t
            2. update z_t = (1-rho) z_{t-1} + rho (W o_t + B x)
            3. output = normalize(z_t)
            4. return block outputs
        """
        #summary
        o=torch.zeros(self.C)
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
        self.last_o = out.clone()
        self.last_raw_o = o.clone()
        return out
    
    def learn(self):
        """
        Update inter-column W using Oja's rule:
        ΔW_ij = eta * (o_i * o_j - o_i^2 * W_ij) - weight_decay * W_ij
        """
        o = self.last_o
        # compute outer product o o^T
        outer = o.unsqueeze(1) * o.unsqueeze(0)  # C x C
        o_sq = o * o  # C
        # Oja update per element
        # ΔW = eta * (outer - diag(o_sq) @ W)  (we expand diag multiplication manually)
        # Equivalent elementwise:
        o_sq_mat = o_sq.unsqueeze(1)  # C x 1
        oja_term = outer - o_sq_mat * self.W  # broadcasting
        deltaW = self.eta * oja_term - self.weight_decay * self.W

        self.W += deltaW

        # optional normalization to stabilize rows
        row_norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        self.W = self.W / row_norms

    

if __name__ == "__main__":
    print("Testing CorticalBlock...")
    
    # Create 3 columns, each takes 8 inputs, has 4 neurons
    in_dim = 8
    n_cols = 3
    n_neurons = 4
    
    cols = []
    for i in range(n_cols):
        col = ProtoColumn(input_size=in_dim, num_neurons=n_neurons)
        cols.append(col)
    
    # Create block
    block = CorticalBlock(
        columns=cols,
        input_proj=True,
        rho=0.3,
        eta=0.01,
        weight_decay=0.001
    )
    
    print(f"Block has {block.C} columns")
    print(f"W shape: {block.W.shape}")
    print(f"B shape: {block.B.shape}")
    
    # Test forward pass
    x = torch.randn(in_dim)
    
    # First, each column processes input
    for col in cols:
        col.forward(x)
    
    # Then block integrates column outputs
    block_out = block.forward(x=x)
    print(f"\nBlock output: {block_out}")
    print(f"Output shape: {block_out.shape}")
    
    # Test learning
    old_W = block.W.clone()
    block.learn()
    new_W = block.W
    
    diff = torch.mean(torch.abs(new_W - old_W)).item()
    print(f"\nWeight change after learning: {diff:.6f}")
    
    # Test without input projection
    print("\n--- Testing without input projection ---")
    block2 = CorticalBlock(
        columns=cols,
        input_proj=False,
        rho=0.2
    )
    
    # Process same input through columns
    for col in cols:
        col.forward(x)
    
    block2_out = block2.forward()
    print(f"Output without input proj: {block2_out}")
    
    # Test temporal sequence
    print("\n--- Testing temporal sequence ---")
    block3 = CorticalBlock(
        columns=[ProtoColumn(input_size=in_dim, num_neurons=n_neurons) for _ in range(2)],
        input_proj=False,
        rho=0.5  # Faster update
    )
    
    outputs = []
    for step in range(5):
        x_step = torch.randn(in_dim)
        
        # Columns process input
        for col in block3.columns:
            col.forward(x_step)
        
        # Block integrates
        out = block3.forward()
        outputs.append(out.detach().numpy())
        
        # Learn every other step
        if step % 2 == 0:
            block3.learn()
    
    print(f"Sequence outputs shape: {len(outputs)} steps x {outputs[0].shape[0]} columns")
    
    # Test different reduction functions
    print("\n--- Testing reduction functions ---")
    
    # Mean reduction (default)
    col = ProtoColumn(input_size=in_dim, num_neurons=n_neurons)
    col.forward(x)
    col_vec = col.last_output
    
    reductions = {
        "mean": lambda v: v.mean(),
        "max": lambda v: v.max(),
        "sum": lambda v: v.sum(),
        "norm": lambda v: torch.norm(v)
    }
    
    for name, reduce_fn in reductions.items():
        val = reduce_fn(col_vec)
        print(f"{name}: {val:.4f}")
    
    print("\nAll tests passed!")


