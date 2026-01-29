import torch
from phase_one_layers_neurons import ProtoLayer
import random
from typing import List,Optional
torch.set_default_dtype(torch.float32)

class ProtoColumn:
    def __init__( self,input_size: int,num_neurons: int = 5,lr: float = 0.01,inhibition_strength: float = 0.2, cooperation_strength: float = 0.1,
                 trace_decay: float = 0.9,elig_decay: float = 0.9,gamma_per_theta:int=8):
        self.layer = ProtoLayer(input_size=input_size,lr=lr,num_neurons=num_neurons,elig_decay=elig_decay)
        # Dynamics parameters
        self.inhibition_strength = inhibition_strength
        self.cooperation_strength = cooperation_strength
        self.gamma = trace_decay
        # Column-level trace (slow state)
        self.column_trace = torch.zeros(1)
        # Internal memory
        self.last_output = None
        self.last_act = None
        self.preferred_gamma_bin = random.randint(0, gamma_per_theta-1)
        # Column-level plasticity gate
        self.plasticity_gate = 1.0

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
    

    def learn(self, target=None, delta=None, gate: float = 1.0, mix_local: float = 0.5,
              DA_phasic: Optional[float] = None,
              DA_tonic: Optional[float] = None,
              DA_baseline: float = 0.0,
              prediction_error: Optional[float] = None):
        """
        Enhanced learning with Phase 5.10 plasticity gate support
        """
        # Combine layer gate with external gate
        effective_gate = gate * self.plasticity_gate
        
        for i, neuron in enumerate(self.layer.neurons):
            neuron_target = target[i] if target is not None and len(target) > i else None
            
            # Call neuron learn with plasticity gate parameters
            neuron.learn(
                target=neuron_target,
                delta=delta,
                gate=effective_gate,
                mix_local=mix_local,
                DA_phasic=DA_phasic,
                DA_tonic=DA_tonic,
                DA_baseline=DA_baseline,
                prediction_error=prediction_error
            )

if __name__ == "__main__":
    print("Testing ProtoColumn with Modulated Learning...")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Test 1: Basic initialization and forward pass
    print("\n1. Testing initialization and forward pass...")
    
    input_size = 10
    num_neurons = 6
    column = ProtoColumn(
        input_size=input_size,
        num_neurons=num_neurons,
        lr=0.01,
        inhibition_strength=0.2,
        cooperation_strength=0.1,
        trace_decay=0.9,
        elig_decay=0.9
    )
    
    print(f"Column created with {num_neurons} neurons")
    print(f"Inhibition strength: {column.inhibition_strength}")
    print(f"Cooperation strength: {column.cooperation_strength}")
    print(f"Trace decay: {column.gamma}")
    print(f"Layer input size: {column.layer.input_size}")
    print(f"Layer neurons: {len(column.layer.neurons)}")
    
    # Test forward pass
    x = torch.randn(input_size)
    output = column.forward(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Output max abs close to 1: {torch.abs(output).max() > 0.9}")
    print(f"Column trace: {column.column_trace.item():.4f}")
    
    # Test 2: Inhibition effects
    print("\n2. Testing inhibition effects...")
    
    for inhibition in [0.0, 0.2, 0.5, 1.0]:
        test_col = ProtoColumn(
            input_size=input_size,
            num_neurons=num_neurons,
            inhibition_strength=inhibition,
            cooperation_strength=0.0
        )
        
        test_output = test_col.forward(x)
        std_dev = torch.std(test_output).item()
        active_neurons = (torch.abs(test_output) > 0.1).sum().item()
        
        print(f"Inhibition={inhibition:.1f}: std={std_dev:.4f}, active={active_neurons}/{num_neurons}")
    
    # Test 3: Cooperation effects
    print("\n3. Testing cooperation effects...")
    
    for cooperation in [0.0, 0.1, 0.5]:
        test_col = ProtoColumn(
            input_size=input_size,
            num_neurons=num_neurons,
            inhibition_strength=0.0,
            cooperation_strength=cooperation
        )
        
        test_output = test_col.forward(x)
        mean_val = torch.mean(test_output).item()
        
        print(f"Cooperation={cooperation:.1f}: mean={mean_val:.4f}")
    
    # Test 4: Trace dynamics
    print("\n4. Testing trace dynamics...")
    
    trace_col = ProtoColumn(
        input_size=input_size,
        num_neurons=num_neurons,
        trace_decay=0.5  # Fast decay for testing
    )
    
    trace_values = []
    for step in range(5):
        x_step = torch.randn(input_size)
        output_step = trace_col.forward(x_step)
        trace_values.append(trace_col.column_trace.item())
        print(f"Step {step+1}: trace={trace_col.column_trace.item():.4f}, mean={output_step.mean():.4f}")
    
    # Test 5: Learning without dopamine (local only)
    print("\n5. Testing learning without dopamine...")
    
    col_local = ProtoColumn(input_size=input_size, num_neurons=num_neurons)
    
    # Store initial weights
    initial_weights = []
    for neuron in col_local.layer.neurons:
        initial_weights.append(neuron.weights.clone())
    
    # Forward
    x_local = torch.randn(input_size)
    output_local = col_local.forward(x_local)
    
    # Learn with target only
    target = torch.randn(num_neurons)
    col_local.learn(target=target, delta=None, gate=1.0, mix_local=1.0)
    
    # Check if weights changed
    weights_changed = False
    for i, neuron in enumerate(col_local.layer.neurons):
        if not torch.allclose(neuron.weights, initial_weights[i], rtol=1e-5):
            weights_changed = True
            break
    
    print(f"Local learning weights changed: {weights_changed}")
    
    # Test 6: Learning with dopamine only
    print("\n6. Testing learning with dopamine only...")
    
    col_da = ProtoColumn(input_size=input_size, num_neurons=num_neurons)
    
    # Store initial weights
    initial_weights_da = []
    for neuron in col_da.layer.neurons:
        initial_weights_da.append(neuron.weights.clone())
    
    # Forward
    x_da = torch.randn(input_size)
    output_da = col_da.forward(x_da)
    
    # Learn with dopamine only
    delta = 0.3
    col_da.learn(target=None, delta=delta, gate=1.0, mix_local=0.0)
    
    # Check if weights changed
    weights_changed_da = False
    for i, neuron in enumerate(col_da.layer.neurons):
        if not torch.allclose(neuron.weights, initial_weights_da[i], rtol=1e-5):
            weights_changed_da = True
            break
    
    print(f"Dopamine learning weights changed: {weights_changed_da}")
    
    # Test 7: Mixed learning
    print("\n7. Testing mixed learning...")
    
    col_mixed = ProtoColumn(input_size=input_size, num_neurons=num_neurons)
    
    # Store initial weights
    initial_weights_mixed = []
    for neuron in col_mixed.layer.neurons:
        initial_weights_mixed.append(neuron.weights.clone())
    
    # Forward
    x_mixed = torch.randn(input_size)
    output_mixed = col_mixed.forward(x_mixed)
    
    # Learn with both
    target_mixed = torch.randn(num_neurons)
    delta_mixed = 0.2
    col_mixed.learn(target=target_mixed, delta=delta_mixed, gate=0.8, mix_local=0.5)
    
    # Check if weights changed
    weights_changed_mixed = False
    for i, neuron in enumerate(col_mixed.layer.neurons):
        if not torch.allclose(neuron.weights, initial_weights_mixed[i], rtol=1e-5):
            weights_changed_mixed = True
            break
    
    print(f"Mixed learning weights changed: {weights_changed_mixed}")
    
    # Test 8: Trace-modulated target
    print("\n8. Testing trace-modulated target...")
    
    col_trace = ProtoColumn(
        input_size=input_size,
        num_neurons=num_neurons,
        trace_decay=0.9
    )
    
    # Build up trace
    for _ in range(3):
        x_trace = torch.randn(input_size)
        col_trace.forward(x_trace)
    
    trace_value = col_trace.column_trace.item()
    print(f"Column trace before learning: {trace_value:.4f}")
    
    target_original = torch.randn(num_neurons)
    col_trace.learn(target=target_original, delta=None, gate=1.0, mix_local=1.0)
    
    print(f"Trace used to modulate target (+0.05*trace)")
    
    # Test 9: Plasticity gate
    print("\n9. Testing plasticity gate...")
    
    col_gated = ProtoColumn(input_size=input_size, num_neurons=num_neurons)
    col_gated.plasticity_gate = 0.3  # Reduce plasticity
    
    # Store initial weights
    initial_weights_gated = []
    for neuron in col_gated.layer.neurons:
        initial_weights_gated.append(neuron.weights.clone())
    
    # Forward
    x_gated = torch.randn(input_size)
    col_gated.forward(x_gated)
    
    # Learn with external gate
    col_gated.learn(target=torch.randn(num_neurons), delta=0.1, gate=0.5, mix_local=0.5)
    
    # Check weight changes
    weight_changes = []
    for i, neuron in enumerate(col_gated.layer.neurons):
        change = torch.norm(neuron.weights - initial_weights_gated[i]).item()
        weight_changes.append(change)
    
    avg_change = sum(weight_changes) / len(weight_changes)
    print(f"Average weight change with effective gate=0.15: {avg_change:.6f}")
    
    # Test 10: Multiple forward-learn cycles
    print("\n10. Testing multiple forward-learn cycles...")
    
    col_seq = ProtoColumn(input_size=input_size, num_neurons=num_neurons)
    outputs_history = []
    
    for step in range(5):
        x_seq = torch.randn(input_size)
        output_seq = col_seq.forward(x_seq)
        outputs_history.append(output_seq.detach().numpy())
        
        # Alternate learning modes
        if step % 2 == 0:
            # Local learning
            col_seq.learn(target=torch.randn(num_neurons), delta=None, mix_local=1.0)
        else:
            # Dopamine learning
            col_seq.learn(delta=0.1, mix_local=0.0)
        
        print(f"Step {step+1}: output mean={output_seq.mean():.4f}, trace={col_seq.column_trace.item():.4f}")
    
    print(f"Sequence length: {len(outputs_history)} steps")
    
    # Test 11: Error cases
    print("\n11. Testing error cases...")
    
    # Should work fine
    try:
        col_test = ProtoColumn(input_size=5, num_neurons=3)
        x_test = torch.randn(5)
        col_test.forward(x_test)
        col_test.learn(target=torch.randn(3))
        print("✓ Normal operation successful")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    print("\n" + "=" * 60)
    print("All ProtoColumn tests completed!")
    print("=" * 60)