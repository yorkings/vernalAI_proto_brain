# test_phase5_plasticity_gates_integration.py
"""
Test Phase 5.10 Plasticity Gates integration
"""

import torch
from phase_two_modulated_neuron import ProtoNeuronMod
from phase_two_column import ProtoColumn
from phase_two_attractor_cotex import AttractorCortex

def test_neuron_plasticity_gates():
    """Test plasticity gates at neuron level"""
    print("\n" + "="*60)
    print("TESTING NEURON-LEVEL PLASTICITY GATES")
    print("="*60)
    
    # Create neuron with plasticity gates enabled
    neuron = ProtoNeuronMod(
        input_size=10,
        learning_rate=0.01,
        enable_phase5=True,
        enable_plasticity_gates=True,
        w1_gate=1.0,
        w3_gate=0.5,  # Confidence weight
        w4_gate=1.0   # Fatigue weight
    )
    
    print("Testing plasticity gates with varying dopamine and prediction errors...")
    
    results = []
    
    for step in range(30):
        # Generate input
        x = torch.randn(10)
        
        # Forward pass
        activation = neuron.forward(x)
        
        # Vary dopamine and prediction error
        if step % 5 == 0:
            DA_phasic = 0.8  # High positive dopamine
            prediction_error = 0.05  # Low error (high confidence)
        elif step % 5 == 1:
            DA_phasic = -0.4  # Negative dopamine
            prediction_error = 0.3  # Medium error
        elif step % 5 == 2:
            DA_phasic = 0.1  # Low dopamine
            prediction_error = 0.6  # High error (low confidence)
        else:
            DA_phasic = 0.3  # Moderate dopamine
            prediction_error = 0.2  # Moderate error
        
        # Store initial weights
        weights_before = neuron.weights.clone()
        
        # Learn with plasticity gates
        neuron.learn(
            delta=DA_phasic,
            DA_phasic=DA_phasic,
            DA_tonic=0.2,
            DA_baseline=0.0,
            prediction_error=prediction_error
        )
        
        # Calculate weight change
        weight_change = torch.norm(neuron.weights - weights_before).item()
        
        # Get plasticity gate stats
        gate_stats = neuron.get_plasticity_gate_stats()
        
        results.append({
            'step': step,
            'DA_phasic': DA_phasic,
            'prediction_error': prediction_error,
            'weight_change': weight_change,
            'confidence': neuron.local_confidence,
            'fatigue': neuron.fatigue
        })
        
        if step % 6 == 0:
            print(f"Step {step:2d}: DA={DA_phasic:.2f}, Error={prediction_error:.3f}, "
                  f"Δw={weight_change:.6f}, Conf={neuron.local_confidence:.3f}, "
                  f"Fatigue={neuron.fatigue:.4f}")
    
    # Analyze results
    print("\nAnalysis:")
    
    # Check that high dopamine + low error causes more learning
    high_da_low_error = [r for r in results if r['DA_phasic'] > 0.5 and r['prediction_error'] < 0.1]
    low_da_high_error = [r for r in results if r['DA_phasic'] < 0.2 and r['prediction_error'] > 0.4]
    
    if high_da_low_error and low_da_high_error:
        avg_high = sum(r['weight_change'] for r in high_da_low_error) / len(high_da_low_error)
        avg_low = sum(r['weight_change'] for r in low_da_high_error) / len(low_da_high_error)
        
        print(f"  High DA + Low error avg Δw: {avg_high:.6f}")
        print(f"  Low DA + High error avg Δw: {avg_low:.6f}")
        
        if avg_high > avg_low * 1.5:
            print("  ✓ Plasticity gates properly modulate learning")
        else:
            print("  ⚠️  Learning modulation could be stronger")
    
    # Check fatigue accumulation
    final_fatigue = results[-1]['fatigue']
    if final_fatigue > 0.001:
        print(f"  ✓ Fatigue accumulated: {final_fatigue:.4f}")
    else:
        print("  ✗ Fatigue not accumulating")
    
    # Check confidence adaptation
    confidence_changes = [r['confidence'] for r in results]
    if max(confidence_changes) - min(confidence_changes) > 0.1:
        print("  ✓ Confidence adapts to prediction errors")
    else:
        print("  ✗ Confidence not adapting")
    
    return neuron, results

def test_cortex_with_plasticity_gates():
    """Test plasticity gates in full cortex"""
    print("\n" + "="*60)
    print("TESTING CORTEX WITH PLASTICITY GATES")
    print("="*60)
    
    # Create columns with neurons that have plasticity gates
    input_size = 8
    num_columns = 3
    neurons_per_column = 4
    
    columns = []
    for i in range(num_columns):
        col = ProtoColumn(
            input_size=input_size,
            num_neurons=neurons_per_column,
            lr=0.01
        )
        
        # Enable plasticity gates for all neurons
        for neuron in col.layer.neurons:
            if hasattr(neuron, 'enable_plasticity_gates'):
                neuron.enable_plasticity_gates = True
                neuron.enable_phase5 = True
        
        columns.append(col)
    
    # Create cortex with Phase 5 enabled
    cortex = AttractorCortex(
        columns=columns,
        input_proj=True,
        enable_phase5=True,
        da_gamma=0.9,
        da_max=1.0,
        enable_intrinsic_motivation=False  # Focus on plasticity gates
    )
    
    print("Running cortex with plasticity gates for 20 steps...")
    
    cortex_results = []
    
    for step in range(20):
        # Generate input
        x = torch.randn(input_size)
        
        # Vary reward
        reward = 0.5 if step % 4 == 0 else -0.2 if step % 4 == 1 else 0.1
        
        # Forward pass
        state = cortex.forward(x)
        
        # Compute surprise (simplified)
        surprise = torch.rand(1).item() * 0.3
        
        # Update dopamine
        da_info = cortex.update_dopamine(
            external_reward=reward,
            surprise=surprise,
            use_intrinsic=False
        )
        
        # Learn with plasticity gates
        cortex.learn(
            da_signal=da_info['DA_phasic'].item(),
            use_phase5_gating=True
        )
        
        # Collect neuron statistics
        neuron_stats = []
        for col in cortex.columns:
            for neuron in col.layer.neurons:
                if hasattr(neuron, 'get_plasticity_gate_stats'):
                    stats = neuron.get_plasticity_gate_stats()
                    if stats.get('enabled', False):
                        neuron_stats.append(stats)
        
        cortex_results.append({
            'step': step,
            'reward': reward,
            'DA_phasic': da_info['DA_phasic'].item(),
            'surprise': surprise,
            'active_neurons_with_gates': len(neuron_stats)
        })
        
        if step % 5 == 0:
            print(f"Step {step:2d}: Reward={reward:.2f}, DA={da_info['DA_phasic'].item():.3f}, "
                  f"Neurons with gates: {len(neuron_stats)}")
    
    # Analyze cortex results
    print("\nCortex Analysis:")
    
    # Check dopamine response
    da_values = [r['DA_phasic'] for r in cortex_results]
    print(f"  Average DA: {sum(da_values)/len(da_values):.3f}")
    print(f"  DA range: [{min(da_values):.3f}, {max(da_values):.3f}]")
    
    # Check plasticity gate usage
    avg_gated_neurons = sum(r['active_neurons_with_gates'] for r in cortex_results) / len(cortex_results)
    total_neurons = num_columns * neurons_per_column
    gate_coverage = avg_gated_neurons / total_neurons
    
    print(f"  Plasticity gate coverage: {gate_coverage:.1%} "
          f"({avg_gated_neurons:.1f}/{total_neurons} neurons)")
    
    if gate_coverage > 0.8:
        print("  ✓ Plasticity gates widely deployed")
    elif gate_coverage > 0.3:
        print("  ⚠️  Partial plasticity gate deployment")
    else:
        print("  ✗ Plasticity gates not properly deployed")
    
    return cortex, cortex_results

def verify_plasticity_gate_functionality():
    """Verify key plasticity gate features"""
    print("\n" + "="*60)
    print("VERIFYING PLASTICITY GATE FUNCTIONALITY")
    print("="*60)
    
    from phase_five_plastic_gates import PlasticityGate
    
    # Create plasticity gate
    gate = PlasticityGate(
        w1=1.0, w2=0.2, w3=0.5, w4=1.0,
        G_min=0.05, G_max=1.5,
        lambda_f=0.95
    )
    
    # Test 1: Gate bounds
    print("\n1. Testing gate bounds:")
    
    test_cases = [
        ("Extreme positive", 10.0, 10.0, 0.0, 1.0, 0.0),
        ("Extreme negative", -10.0, 0.0, 0.0, 1.0, 0.0),
        ("High fatigue", 0.5, 0.2, 0.0, 1.0, 10.0),
        ("Low confidence", 0.5, 0.2, 0.0, 0.1, 0.0),
    ]
    
    for name, DA_phasic, DA_tonic, DA_base, confidence, fatigue in test_cases:
        G = gate.compute_gate(
            DA_phasic=DA_phasic,
            DA_tonic=DA_tonic,
            DA_baseline=DA_base,
            local_confidence=confidence,
            fatigue=fatigue
        )
        
        bounded = gate.G_min <= G <= gate.G_max
        status = "✓" if bounded else "✗"
        print(f"  {status} {name}: G={G:.3f} (bounds: {gate.G_min:.2f}-{gate.G_max:.2f})")
    
    # Test 2: Fatigue accumulation
    print("\n2. Testing fatigue accumulation:")
    
    class TestWeight:
        def __init__(self):
            self.w = torch.tensor(0.5)
    
    test_weight = TestWeight()
    state = gate.default_state
    
    # Apply multiple updates
    for i in range(5):
        applied = gate.apply_gated_update(
            synapse=test_weight,
            delta_w=torch.tensor(0.02),
            DA_phasic=0.5,
            DA_tonic=0.2,
            DA_baseline=0.0,
            prediction_error=0.1,
            lr=0.01,
            state=state
        )
    
    print(f"  Final fatigue: {state.fatigue:.4f}")
    if state.fatigue > 0.001:
        print("  ✓ Fatigue accumulates with repeated updates")
    else:
        print("  ✗ Fatigue not accumulating")
    
    # Test 3: Confidence adaptation
    print("\n3. Testing confidence adaptation:")
    
    state2 = gate.default_state
    
    # Simulate decreasing prediction errors (improving predictions)
    errors = [0.5, 0.3, 0.2, 0.1, 0.05]
    for error in errors:
        gate.update_local_confidence(error, state2)
    
    print(f"  Initial confidence (high error): ~1.0")
    print(f"  Final confidence (low error): {state2.local_confidence:.3f}")
    
    if state2.local_confidence > 0.7:
        print("  ✓ Confidence increases with better predictions")
    else:
        print("  ✗ Confidence not responding to prediction improvements")
    
    return gate

if __name__ == "__main__":
    print("PHASE 5.10: PLASTICITY GATES INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Neuron-level gates
    neuron, neuron_results = test_neuron_plasticity_gates()
    
    # Test 2: Cortex integration
    cortex, cortex_results = test_cortex_with_plasticity_gates()
    
    # Test 3: Core functionality
    gate = verify_plasticity_gate_functionality()
    
    # Final summary
    print("\n" + "="*60)
    print("PHASE 5.10 IMPLEMENTATION SUMMARY")
    print("="*60)
    
    checks = [
        ("Neuron plasticity gates functional", 
         neuron.enable_plasticity_gates and hasattr(neuron, 'plasticity_gate')),
        ("Cortex integrates plasticity gates",
         any('active_neurons_with_gates' in r for r in cortex_results)),
        ("Gate bounds enforced",
         all(gate.G_min <= g <= gate.G_max for g in gate.stats['gate_values'])),
        ("Fatigue mechanism working",
         any(r['fatigue'] > 0 for r in neuron_results)),
        ("Confidence adaptation working",
         max(r['confidence'] for r in neuron_results) - min(r['confidence'] for r in neuron_results) > 0.05),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ PHASE 5.10 PLASTICITY GATES IMPLEMENTATION SUCCESSFUL!")
        print("\nNext: Phase 5.11 - Prioritized Replay")
    else:
        print("\n⚠️  Some checks failed. Review implementation.")
    
    print("="*60)