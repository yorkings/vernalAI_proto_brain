import torch
from phase_one_layers_neurons import ProtoLayer

class ProtoColumn:
    def __init__( self,input_size: int,num_neurons: int = 5,lr: float = 0.01,inhibition_strength: float = 0.2, cooperation_strength: float = 0.1,trace_decay: float = 0.9):
        self.layer = ProtoLayer(input_size=input_size,lr=lr,num_neurons=num_neurons)
        # Dynamics parameters
        self.inhibition_strength = inhibition_strength
        self.cooperation_strength = cooperation_strength
        self.gamma = trace_decay
        # Column-level trace (slow state)
        self.column_trace = torch.zeros(1)
        # Internal memory
        self.last_output = None
        self.last_act = None

    def forward(self,x):
        """
         hi​=ai​
         lateral inhibitor : hi′​=hi​−λ(hi​−hˉ)
         coop exitation: c=h+η⋅mean(a)
         satbilize/normalize
         column_trace: T←γT+oˉ
        """    
        act=self.layer.forward(x)
        self.last_act=act
        mean_out=torch.mean(act)
        inhibitor = act - self.inhibition_strength*(act -mean_out)
        
        coop_boost=self.cooperation_strength*mean_out
        excited=inhibitor+coop_boost

        # 4 — Winner-biased normalization
        max_act = torch.max(excited)
        normalized = excited / (1e-6 + torch.abs(max_act))
        self.column_trace=(self.gamma*self.column_trace + torch.mean(normalized))
    
        self.last_output=normalized
        return normalized
    
    def learn(self, target):
        """
        Column learning rule:
        """
        # Bias target using column state
        mod_target = target + 0.05 * self.column_trace
        for neuron in self.layer.neurons:
            neuron.learn(mod_target)


if __name__ == "__main__":
    # Test 1: Basic initialization and forward pass
    print("Test 1: Basic initialization and forward pass")
    print("-" * 50)
    
    input_size = 10
    num_neurons = 5
    column = ProtoColumn(input_size=input_size, num_neurons=num_neurons)
    
    # Create a random input
    x = torch.randn(input_size)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = column.forward(x)
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Check normalization
    print(f"Is normalized (max close to 1): {torch.abs(output.max()) > 0.9}")
    
    # Test 2: Trace dynamics
    print("\nTest 2: Trace dynamics")
    print("-" * 50)
    
    # Multiple forward passes to see trace accumulation
    column2 = ProtoColumn(input_size=input_size, num_neurons=num_neurons, trace_decay=0.5)
    trace_values = []
    
    for i in range(5):
        x = torch.randn(input_size)
        output = column2.forward(x)
        trace_values.append(column2.column_trace.item())
        print(f"Step {i+1}: Trace = {column2.column_trace.item():.4f}, Mean output = {output.mean():.4f}")
    
    # Test 3: Learning - FIXED VERSION
    print("\nTest 3: Learning")
    print("-" * 50)
    
    column3 = ProtoColumn(input_size=input_size, num_neurons=num_neurons, lr=0.1)
    
    # Store initial weights for comparison
    initial_weights = []
    for neuron in column3.layer.neurons:
        initial_weights.append(neuron.weights.clone())
    
    # Forward pass
    x = torch.randn(input_size)
    output = column3.forward(x)
    print(f"Before learning - Output: {output}")
    
    # IMPORTANT FIX: Each neuron needs its own error signal
    # The column output is [neuron1_output, neuron2_output, ...]
    # We need error_i for each neuron i
    target = torch.randn(num_neurons)  # Target for each neuron
    print(f"Target: {target}")
    
    # Calculate error per neuron
    errors = target - output
    print(f"Errors per neuron: {errors}")
    
    # Learn - each neuron gets its own error
    for i, neuron in enumerate(column3.layer.neurons):
        # Extract the error for this specific neuron
        neuron_error = errors[i].item()  # Convert to scalar
        neuron.learn(neuron_error)
    
    # Alternative: Use the column's learn method but it needs to be fixed
    # For now, use the manual approach above
    
    # Forward pass with same input after learning
    output_after = column3.forward(x)
    print(f"After learning - Output: {output_after}")
    
    # Check if weights changed
    weight_changed = False
    for i, neuron in enumerate(column3.layer.neurons):
        if not torch.allclose(neuron.weights, initial_weights[i], rtol=1e-5):
            weight_changed = True
            print(f"Neuron {i} weights changed")
            break
    print(f"Weights changed: {weight_changed}")
    
    # Test 4: Test the actual column.learn() method
    print("\nTest 4: Testing column.learn() method")
    print("-" * 50)
    
    column4 = ProtoColumn(input_size=input_size, num_neurons=num_neurons, lr=0.01)
    
    # Forward pass
    x = torch.randn(input_size)
    output_before = column4.forward(x)
    
    # Create target and use column.learn()
    target = torch.randn(num_neurons)
    
    try:
        column4.learn(target)
        print("column.learn() executed successfully")
        
        # Check if it modified the trace properly
        print(f"Column trace after learning: {column4.column_trace.item():.4f}")
        
    except Exception as e:
        print(f"Error in column.learn(): {e}")
        print("This suggests the neuron.learn() method expects a scalar, not a vector")
    
    # Test 5: Inhibition and cooperation effects
    print("\nTest 5: Inhibition and cooperation effects")
    print("-" * 50)
    
    # Test with different inhibition strengths
    for inhibition in [0.0, 0.2, 0.5, 1.0]:
        column_test = ProtoColumn(
            input_size=input_size, 
            num_neurons=num_neurons,
            inhibition_strength=inhibition,
            cooperation_strength=0.0  # Turn off cooperation for this test
        )
        
        x = torch.randn(input_size)
        act = column_test.layer.forward(x)  # Get raw activations
        output = column_test.forward(x)     # Get after inhibition
        
        # Calculate spread (standard deviation) to see inhibition effect
        act_spread = torch.std(act).item()
        out_spread = torch.std(output).item()
        
        print(f"Inhibition={inhibition:.1f}: Raw spread={act_spread:.4f}, After inhibition={out_spread:.4f}")
    
    # Test 6: Cooperation effects
    print("\nTest 6: Cooperation effects")
    print("-" * 50)
    
    for cooperation in [0.0, 0.1, 0.5]:
        column_test = ProtoColumn(
            input_size=input_size, 
            num_neurons=num_neurons,
            inhibition_strength=0.0,  # Turn off inhibition for this test
            cooperation_strength=cooperation
        )
        
        x = torch.randn(input_size)
        act = column_test.layer.forward(x)  # Get raw activations
        output = column_test.forward(x)     # Get after cooperation
        
        # Calculate mean activation
        act_mean = torch.mean(act).item()
        out_mean = torch.mean(output).item()
        
        print(f"Cooperation={cooperation:.1f}: Raw mean={act_mean:.4f}, After cooperation={out_mean:.4f}")
    
    # Test 7: Batch processing test
    print("\nTest 7: Batch processing")
    print("-" * 50)
    
    batch_size = 3
    column_batch = ProtoColumn(input_size=input_size, num_neurons=num_neurons)
    
    batch_input = torch.randn(batch_size, input_size)
    batch_output = []
    
    for i in range(batch_size):
        output = column_batch.forward(batch_input[i])
        batch_output.append(output)
        print(f"Batch item {i}: output shape = {output.shape}")
    
    batch_tensor = torch.stack(batch_output)
    print(f"Batch output tensor shape: {batch_tensor.shape}")
    
    # Test 8: Edge cases
    print("\nTest 8: Edge cases")
    print("-" * 50)
    
    # Zero input
    x_zero = torch.zeros(input_size)
    column_zero = ProtoColumn(input_size=input_size, num_neurons=num_neurons)
    output_zero = column_zero.forward(x_zero)
    print(f"Zero input - Output: {output_zero}")
    print(f"Zero input - All zeros: {torch.all(output_zero == 0)}")
    print(f"Zero input - Column trace: {column_zero.column_trace.item():.4f}")
    
    # Very large input
    x_large = torch.ones(input_size) * 100
    output_large = column_zero.forward(x_large)
    print(f"\nLarge input - Output range: [{output_large.min():.4f}, {output_large.max():.4f}]")
    print(f"Large input - Max close to 1: {torch.abs(output_large.max()) > 0.99}")
    
    print("\nAll tests completed!") 
        