import torch
from phase_two_modulated_neuron import ProtoNeuronMod
"""lr=learning-rate """
class ProtoLayer:
    def __init__(self,input_size:int,lr:float=0.01,num_neurons:int=5,feedback_strength:float=0.1, elig_decay: float = 0.9):
        self.input_size=input_size
        self.lr=lr
        self.num_neurons=num_neurons
        self.feedback_strength=feedback_strength
        self.neurons = [ProtoNeuronMod(input_size, learning_rate=lr,elig_decay=elig_decay)  for _ in range(num_neurons)]
        self.last_outputs=None
        self.plasticity_gate = 1.0



    def forward(self,x):
        """
        1. output[i] = neuron[i].forward(x)

        2. feedback = β * mean(output)
           output[i] += feedback

        3. stabilization:
           out = out / (1 + |out|)
        """
        if len(x) != self.input_size:
          raise ValueError(f"Input size {len(x)} doesn't match {self.input_size}")
        
        outputs=torch.zeros(self.num_neurons)
        for i,neuron in enumerate(self.neurons):
            outputs[i]=neuron.forward(x)

        mean_outputs= torch.mean(outputs)
        feedback=self.feedback_strength *mean_outputs
        outputs+=feedback
        #stabilize
        outputs = outputs / (1 +torch.abs(outputs))
        #make it non linear
        outputs = torch.tanh(outputs)
        self.last_outputs=outputs
        return outputs
    
    def learn(self,target=None,delta=None, gate: float = 1.0, mix_local: float = 0.5):
        # Combine layer gate with external gate
        effective_gate = gate * self.plasticity_gate
        for i,neuron in enumerate(self.neurons):
            neuron_target = target[i] if target is not None and len(target) > i else None
            neuron.learn(target=neuron_target, delta=delta,gate=effective_gate, mix_local=mix_local)

    def reset_eligibilities(self):
        """Reset eligibility traces for all neurons."""
        for neuron in self.neurons:
            neuron.reset_eligibility()        

if __name__ == "__main__":
    print("Testing ProtoLayer with Modulated Neurons...")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Test 1: Basic initialization and forward pass
    print("\n1. Testing initialization and forward pass...")
    input_size = 10
    num_neurons = 5
    layer = ProtoLayer(input_size=input_size, lr=0.01, num_neurons=num_neurons)
    
    print(f"Created layer with {layer.num_neurons} neurons")
    print(f"Input size: {layer.input_size}")
    print(f"Feedback strength: {layer.feedback_strength}")
    print(f"Number of neurons: {len(layer.neurons)}")
    
    # Test forward pass
    x = torch.randn(input_size)
    outputs = layer.forward(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
    print(f"Output values: {outputs}")
    print(f"All outputs in [-1, 1]: {torch.all(outputs >= -1) and torch.all(outputs <= 1)}")
    
    # Test 2: Learning without dopamine
    print("\n2. Testing learning without dopamine...")
    
    # Store initial weights
    initial_weights = []
    for neuron in layer.neurons:
        initial_weights.append(neuron.weights.clone())
    
    # Learn with target (local learning)
    target = torch.randn(num_neurons)
    layer.learn(target=target, delta=None, gate=1.0, mix_local=1.0)
    
    # Check if weights changed
    weights_changed = False
    for i, neuron in enumerate(layer.neurons):
        if not torch.allclose(neuron.weights, initial_weights[i], rtol=1e-5):
            weights_changed = True
            break
    
    print(f"Local learning weights changed: {weights_changed}")
    print(f"Last outputs stored: {layer.last_outputs is not None}")
    
    # Test 3: Learning with dopamine
    print("\n3. Testing learning with dopamine modulation...")
    
    layer2 = ProtoLayer(input_size=input_size, num_neurons=num_neurons, elig_decay=0.9)
    
    # Forward pass
    x2 = torch.randn(input_size)
    outputs2 = layer2.forward(x2)
    
    # Store initial weights
    initial_weights2 = []
    for neuron in layer2.neurons:
        initial_weights2.append(neuron.weights.clone())
    
    # Learn with dopamine
    delta = 0.5  # Positive dopamine signal
    layer2.learn(target=None, delta=delta, gate=1.0, mix_local=0.0)
    
    # Check if weights changed
    weights_changed2 = False
    for i, neuron in enumerate(layer2.neurons):
        if not torch.allclose(neuron.weights, initial_weights2[i], rtol=1e-5):
            weights_changed2 = True
            break
    
    print(f"Dopamine learning weights changed: {weights_changed2}")
    
    # Test 4: Mixed learning (local + dopamine)
    print("\n4. Testing mixed learning...")
    
    layer3 = ProtoLayer(input_size=input_size, num_neurons=num_neurons)
    x3 = torch.randn(input_size)
    outputs3 = layer3.forward(x3)
    
    # Store initial weights
    initial_weights3 = []
    for neuron in layer3.neurons:
        initial_weights3.append(neuron.weights.clone())
    
    # Learn with both local and dopamine
    target3 = torch.randn(num_neurons)
    delta3 = 0.3
    layer3.learn(target=target3, delta=delta3, gate=0.8, mix_local=0.5)
    
    # Check if weights changed
    weights_changed3 = False
    for i, neuron in enumerate(layer3.neurons):
        if not torch.allclose(neuron.weights, initial_weights3[i], rtol=1e-5):
            weights_changed3 = True
            break
    
    print(f"Mixed learning weights changed: {weights_changed3}")
    
    # Test 5: Feedback strength effect
    print("\n5. Testing feedback strength effect...")
    
    for feedback in [0.0, 0.1, 0.5, 1.0]:
        test_layer = ProtoLayer(
            input_size=input_size, 
            num_neurons=num_neurons,
            feedback_strength=feedback
        )
        
        test_input = torch.randn(input_size)
        test_output = test_layer.forward(test_input)
        
        # Calculate standard deviation (feedback should reduce variance)
        std_dev = torch.std(test_output).item()
        mean_val = torch.mean(test_output).item()
        print(f"Feedback={feedback:.1f}: Output std={std_dev:.4f}, mean={mean_val:.4f}")
    
    # Test 6: Eligibility trace reset
    print("\n6. Testing eligibility trace reset...")
    
    layer4 = ProtoLayer(input_size=input_size, num_neurons=num_neurons)
    x4 = torch.randn(input_size)
    
    # Forward to build eligibility
    layer4.forward(x4)
    
    # Check initial eligibility
    initial_elig = []
    for neuron in layer4.neurons:
        initial_elig.append(neuron.eligibility.clone())
    
    # Reset eligibilities
    layer4.reset_eligibilities()
    
    # Check if reset
    elig_reset = True
    for i, neuron in enumerate(layer4.neurons):
        if not torch.allclose(neuron.eligibility, torch.zeros_like(neuron.eligibility)):
            elig_reset = False
            break
    
    print(f"Eligibility traces reset: {elig_reset}")
    
    # Test 7: Plasticity gate
    print("\n7. Testing plasticity gate...")
    
    layer5 = ProtoLayer(input_size=input_size, num_neurons=num_neurons)
    layer5.plasticity_gate = 0.5  # Half plasticity
    
    # Store initial weights
    initial_weights5 = []
    for neuron in layer5.neurons:
        initial_weights5.append(neuron.weights.clone())
    
    # Forward
    x5 = torch.randn(input_size)
    layer5.forward(x5)
    
    # Learn with gate
    layer5.learn(target=torch.randn(num_neurons), delta=0.2, gate=0.5, mix_local=0.5)
    
    # Check weight changes
    weight_changes5 = []
    for i, neuron in enumerate(layer5.neurons):
        change = torch.norm(neuron.weights - initial_weights5[i]).item()
        weight_changes5.append(change)
    
    avg_change = sum(weight_changes5) / len(weight_changes5)
    print(f"Average weight change with gate=0.25: {avg_change:.6f}")
    
    # Test 8: Error handling
    print("\n8. Testing error handling...")
    
    try:
        bad_layer = ProtoLayer(input_size=5, num_neurons=3)
        bad_input = torch.randn(3)  # Wrong size
        bad_layer.forward(bad_input)
    except ValueError as e:
        print(f"✓ Correctly caught size mismatch: {e}")
    
    # Test 9: Multiple forward passes
    print("\n9. Testing multiple forward passes...")
    
    layer6 = ProtoLayer(input_size=input_size, num_neurons=num_neurons)
    outputs_history = []
    
    for i in range(3):
        x_iter = torch.randn(input_size)
        output = layer6.forward(x_iter)
        outputs_history.append(output)
        print(f"  Pass {i+1}: shape={output.shape}, mean={output.mean():.4f}")
    
    # Test 10: Batch-like processing
    print("\n10. Testing batch-like processing...")
    
    batch_size = 3
    layer7 = ProtoLayer(input_size=input_size, num_neurons=num_neurons)
    
    batch_outputs = []
    for i in range(batch_size):
        x_batch = torch.randn(input_size)
        output = layer7.forward(x_batch)
        batch_outputs.append(output)
        
        # Learn each step
        if i % 2 == 0:
            layer7.learn(target=torch.randn(num_neurons), delta=0.1)
    
    batch_tensor = torch.stack(batch_outputs)
    print(f"Batch outputs shape: {batch_tensor.shape}")
    print(f"Batch mean: {batch_tensor.mean():.4f}")
    
    print("\n" + "=" * 60)
    print("All ProtoLayer tests completed!")
    print("=" * 60)

