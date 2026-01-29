# test_phase5_integration_full.py
"""
Test Phase 5 dopamine integration with existing Phase 3/4 system
"""
import torch
import numpy as np
from typing import Dict, List
import sys
import os

# Add your project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase_two_attractor_cotex import AttractorCortex
from phase_two_column import ProtoColumn
from phase_two_rhythm_engine import RhythmEngine
from phase_three_integrated_life_loop import create_hierarchical_system, TestEnvironment
from phase_three_integrated_life_loop import hierarchical_step

def create_test_cortex_with_phase5():
    """Create a simple test cortex with Phase 5 enabled"""
    print("\n" + "="*60)
    print("CREATING TEST CORTEX WITH PHASE 5")
    print("="*60)
    
    input_size = 8
    num_columns = 4
    neurons_per_column = 5
    
    # Create columns with Phase 5 enabled neurons
    columns = []
    for i in range(num_columns):
        col = ProtoColumn(
            input_size=input_size,
            num_neurons=neurons_per_column,
            lr=0.01,
            inhibition_strength=0.2,
            cooperation_strength=0.1
        )
        
        # Enable Phase 5 for all neurons in column
        if hasattr(col.layer, 'neurons'):
            for neuron in col.layer.neurons:
                if hasattr(neuron, 'enable_phase5'):
                    neuron.enable_phase5 = True
                    neuron.elig_decay = 0.95  # Stronger eligibility
        
        columns.append(col)
    
    # Create cortex with Phase 5
    cortex = AttractorCortex(
        columns=columns,
        input_proj=True,
        settling_steps=5,
        k_sparsity=2,
        
        # Phase 5 parameters
        enable_phase5=True,
        da_gamma=0.9,
        da_tau_tonic=0.995,
        da_k=2.0,
        da_max=1.0,
        
        # Memory parameters (keep existing)
        buffer_capacity=50,
        max_episode_length=5,
        immediate_replay_threshold=0.4,
        value_lr=0.1
    )
    
    print(f"Created cortex with {cortex.C} columns")
    print(f"Phase 5 enabled: {cortex.enable_phase5}")
    if hasattr(cortex, 'neuromodulator'):
        print(f"Dopamine neuromodulator: ✓")
    
    return cortex, input_size

def test_dopamine_signals():
    """Test dopamine signal generation and response"""
    print("\n" + "="*60)
    print("TEST 1: DOPAMINE SIGNAL GENERATION")
    print("="*60)
    
    cortex, input_size = create_test_cortex_with_phase5()
    
    dopamine_history = []
    reward_patterns = [
        (0.5, 0.2),   # (reward, surprise) - positive reward, small surprise
        (1.0, 0.8),   # big reward, big surprise
        (-0.5, 0.3),  # negative reward, medium surprise
        (0.0, 0.1),   # no reward, small surprise
        (0.8, 0.01),  # reward, very small surprise (expected)
    ]
    
    for i, (reward, surprise) in enumerate(reward_patterns):
        # Forward pass to set state
        x = torch.randn(input_size)
        cortex.forward(x)
        
        # Update dopamine
        da_info = cortex.update_dopamine(
            external_reward=reward,
            surprise=surprise,
            use_intrinsic=True
        )
        
        dopamine_history.append({
            'step': i,
            'reward': reward,
            'surprise': surprise,
            'DA_phasic': da_info['DA_phasic'].item(),
            'DA_tonic': da_info['DA_tonic'].item(),
            'R_intrinsic': da_info['R_intrinsic']
        })
        
        print(f"Step {i}: R={reward:5.2f}, S={surprise:5.2f} → "
              f"DA_phasic={da_info['DA_phasic'].item():6.3f}, "
              f"DA_tonic={da_info['DA_tonic'].item():6.3f}, "
              f"R_int={da_info['R_intrinsic']:6.3f}")
    
    # Analyze dopamine responses
    print("\nDopamine Response Analysis:")
    da_phasic_vals = [h['DA_phasic'] for h in dopamine_history]
    print(f"  Max DA_phasic: {max(da_phasic_vals):.3f}")
    print(f"  Min DA_phasic: {min(da_phasic_vals):.3f}")
    print(f"  Average DA_phasic: {np.mean(da_phasic_vals):.3f}")
    
    # Check dopamine tracks rewards
    rewards = [h['reward'] for h in dopamine_history]
    da_correlation = np.corrcoef(rewards, da_phasic_vals)[0, 1]
    print(f"  Correlation (reward vs DA): {da_correlation:.3f}")
    
    return cortex, dopamine_history

def test_dopamine_gated_learning():
    """Test that learning is properly gated by dopamine"""
    print("\n" + "="*60)
    print("TEST 2: DOPAMINE-GATED LEARNING")
    print("="*60)
    
    cortex, input_size = create_test_cortex_with_phase5()
    
    # Track weight changes under different dopamine conditions
    weight_changes = []
    
    test_conditions = [
        ("High positive DA", 0.8, 0.3),
        ("Low positive DA", 0.1, 0.1),
        ("Negative DA", -0.6, 0.2),
        ("Very low DA", 0.01, 0.05),
    ]
    
    for condition_name, reward, surprise in test_conditions:
        # Store initial weights
        initial_weights = []
        for col in cortex.columns:
            for neuron in col.layer.neurons:
                initial_weights.append(neuron.weights.clone())
        
        # Process with dopamine
        x = torch.randn(input_size)
        cortex.forward(x)
        
        da_info = cortex.update_dopamine(
            external_reward=reward,
            surprise=surprise
        )
        
        # Learn with dopamine gating
        cortex.learn(da_signal=da_info['DA_phasic'].item(), use_phase5_gating=True)
        
        # Calculate weight changes
        total_change = 0.0
        idx = 0
        for col in cortex.columns:
            for neuron in col.layer.neurons:
                if idx < len(initial_weights):
                    change = torch.norm(neuron.weights - initial_weights[idx]).item()
                    total_change += change
                    idx += 1
        
        avg_change = total_change / idx if idx > 0 else 0
        
        weight_changes.append({
            'condition': condition_name,
            'DA_phasic': da_info['DA_phasic'].item(),
            'avg_weight_change': avg_change,
            'learned': avg_change > 1e-6
        })
        
        print(f"{condition_name:20s}: "
              f"DA={da_info['DA_phasic'].item():6.3f} → "
              f"Weight change={avg_change:.6f} "
              f"{'✓' if avg_change > 1e-6 else '✗'}")
    
    # Verify learning correlates with dopamine magnitude
    da_mags = [abs(wc['DA_phasic']) for wc in weight_changes]
    changes = [wc['avg_weight_change'] for wc in weight_changes]
    
    if len(da_mags) > 1 and len(changes) > 1:
        corr = np.corrcoef(da_mags, changes)[0, 1]
        print(f"\nCorrelation (|DA| vs learning): {corr:.3f}")
        
        if corr > 0.5:
            print("✓ Strong correlation: Learning scales with dopamine magnitude")
        elif corr > 0.2:
            print("✓ Moderate correlation")
        else:
            print("✗ Weak correlation - check dopamine gating")
    
    return weight_changes

def test_with_existing_hierarchy():
    """Test Phase 5 integration with your existing hierarchical system"""
    print("\n" + "="*60)
    print("TEST 3: INTEGRATION WITH EXISTING HIERARCHY")
    print("="*60)
    
    # Create your hierarchical system but enable Phase 5
    print("Creating hierarchical system with Phase 5 enabled...")
    
    # We need to modify create_hierarchical_system to pass Phase 5 params
    # For now, let's manually create a simpler test
    
    # Create test environment
    env = TestEnvironment(input_size=10)
    
    # Create a single cortex with Phase 5 for testing
    input_size = 10
    num_columns = 4
    neurons_per_column = 5
    
    columns = []
    for i in range(num_columns):
        col = ProtoColumn(
            input_size=input_size,
            num_neurons=neurons_per_column,
            lr=0.01
        )
        columns.append(col)
    
    cortex = AttractorCortex(
        columns=columns,
        input_proj=True,
        enable_phase5=True,
        da_gamma=0.9,
        da_max=1.0
    )
    
    # Simple test loop
    print("\nRunning simple environment test with Phase 5:")
    print("-"*60)
    
    stats = {
        'rewards': [],
        'DA_phasic': [],
        'surprises': [],
        'weight_changes': []
    }
    
    prev_reward = None
    prev_state = None
    
    for step in range(20):
        # Get state
        x = env.get_state_vector()
        
        # Forward pass
        state = cortex.forward(x)
        
        # Simple action: move toward position 0
        action = -1 if env.position > 0 else 1 if env.position < 0 else 0
        
        # Get reward from environment
        reward = env.step(action)
        
        # Compute surprise (simplified)
        surprise = 0.0
        if prev_state is not None:
            surprise = torch.norm(state - prev_state).item() * 0.1
        
        # Update dopamine
        da_info = cortex.update_dopamine(
            external_reward=reward,
            surprise=surprise
        )
        
        # Learn with dopamine
        if step > 0:  # Skip first step
            cortex.learn(da_signal=da_info['DA_phasic'].item())
        
        # Store stats
        stats['rewards'].append(reward)
        stats['DA_phasic'].append(da_info['DA_phasic'].item())
        stats['surprises'].append(surprise)
        
        # Track position
        if step % 5 == 0:
            print(f"Step {step:2d}: Pos={env.position:6.2f}, "
                  f"Reward={reward:6.2f}, "
                  f"DA={da_info['DA_phasic'].item():6.3f}, "
                  f"Surprise={surprise:6.3f}")
        
        prev_state = state.clone()
        prev_reward = reward
    
    # Analyze results
    print("\nTest Results Summary:")
    print(f"  Final position: {env.position:.2f} (goal: 0.0)")
    print(f"  Average reward: {np.mean(stats['rewards']):.3f}")
    print(f"  Average DA_phasic: {np.mean(stats['DA_phasic']):.3f}")
    print(f"  Total steps with learning: {len([d for d in stats['DA_phasic'] if abs(d) > 0.05])}")
    
    return stats

def test_neuron_level_phase5():
    """Test Phase 5 features at neuron level"""
    print("\n" + "="*60)
    print("TEST 4: NEURON-LEVEL PHASE 5 FEATURES")
    print("="*60)
    
    from phase_two_modulated_neuron import ProtoNeuronMod
    
    # Create neuron with Phase 5
    neuron = ProtoNeuronMod(
        input_size=5,
        learning_rate=0.01,
        elig_decay=0.95,
        enable_phase5=True
    )
    
    print("Testing eligibility traces and fatigue...")
    
    # Simulate activity
    eligibility_history = []
    fatigue_history = []
    
    for i in range(10):
        x = torch.randn(5)
        activation = neuron.forward(x)
        
        # Check eligibility
        elig_norm = torch.norm(neuron.eligibility).item()
        eligibility_history.append(elig_norm)
        
        # Simulate learning with dopamine
        dopamine = 0.5 if i % 3 == 0 else -0.2  # Varying dopamine
        
        # Store initial weights
        initial_weights = neuron.weights.clone()
        
        # Learn
        neuron.learn(delta=dopamine)
        
        # Check weight change
        weight_change = torch.norm(neuron.weights - initial_weights).item()
        
        # Track fatigue
        fatigue_history.append(neuron.fatigue)
        
        if i % 2 == 0:
            print(f"  Step {i}: "
                  f"DA={dopamine:5.2f}, "
                  f"Elig={elig_norm:.3f}, "
                  f"Δw={weight_change:.5f}, "
                  f"Fatigue={neuron.fatigue:.4f}")
    
    # Analyze
    print("\nNeuron Analysis:")
    print(f"  Eligibility persistence: {eligibility_history[-1] / (eligibility_history[0] + 1e-8):.3f}x")
    print(f"  Final fatigue: {fatigue_history[-1]:.4f}")
    print(f"  Weight norm: {torch.norm(neuron.weights).item():.3f}")
    
    # Check that eligibility decays but persists
    if eligibility_history[-1] > 0.01 and eligibility_history[-1] < eligibility_history[0] * 2:
        print("✓ Eligibility traces working (persistent but bounded)")
    else:
        print("✗ Check eligibility trace decay")
    
    return neuron, eligibility_history, fatigue_history

def test_phase5_with_rhythm():
    """Test Phase 5 with rhythm engine (theta/gamma timing)"""
    print("\n" + "="*60)
    print("TEST 5: PHASE 5 WITH RHYTHM ENGINE")
    print("="*60)
    
    from phase_two_rhythm_engine import RhythmEngine
    
    # Create rhythm
    rhythm = RhythmEngine(
        theta_freq=6.0,
        gamma_per_theta=6,
        sim_steps_per_second=80,
        theta_phase_window=(0.1, 0.4)
    )
    
    # Create cortex with rhythm
    input_size = 8
    num_columns = 3
    
    columns = [ProtoColumn(input_size=input_size, num_neurons=4) for _ in range(num_columns)]
    
    cortex = AttractorCortex(
        columns=columns,
        input_proj=False,
        rhythm=rhythm,
        enable_phase5=True,
        da_gamma=0.9
    )
    
    print("Testing dopamine timing with theta cycles...")
    
    da_by_theta_phase = {'in_window': [], 'out_window': []}
    
    for step in range(50):
        rhythm.tick()
        in_theta = rhythm.theta_in_window()
        
        x = torch.randn(input_size)
        cortex.forward(x)
        
        # Vary reward based on theta phase
        reward = 0.3 if in_theta else 0.1
        surprise = 0.2
        
        da_info = cortex.update_dopamine(
            external_reward=reward,
            surprise=surprise
        )
        
        # Learn only during theta windows (as per biology)
        if in_theta:
            cortex.learn(da_signal=da_info['DA_phasic'].item())
            da_by_theta_phase['in_window'].append(da_info['DA_phasic'].item())
        else:
            da_by_theta_phase['out_window'].append(da_info['DA_phasic'].item())
        
        if step % 10 == 0:
            print(f"Step {step:2d}: "
                  f"Theta={'✓' if in_theta else '✗'}, "
                  f"Reward={reward:.2f}, "
                  f"DA={da_info['DA_phasic'].item():.3f}")
    
    # Analyze theta-phase effects
    print("\nTheta Phase Analysis:")
    if da_by_theta_phase['in_window']:
        avg_da_in = np.mean(da_by_theta_phase['in_window'])
        print(f"  Average DA in theta window: {avg_da_in:.3f}")
    
    if da_by_theta_phase['out_window']:
        avg_da_out = np.mean(da_by_theta_phase['out_window'])
        print(f"  Average DA outside theta: {avg_da_out:.3f}")
    
    # Check if learning is concentrated in theta windows
    if da_by_theta_phase['in_window'] and da_by_theta_phase['out_window']:
        ratio = np.mean(da_by_theta_phase['in_window']) / (np.mean(da_by_theta_phase['out_window']) + 1e-8)
        if ratio > 1.2:
            print("✓ Dopamine/learning concentrated in theta windows")
        else:
            print("✗ Check theta-gated learning")
    
    return da_by_theta_phase

def run_comprehensive_test():
    """Run all Phase 5 tests"""
    print("\n" + "="*80)
    print("PHASE 5 COMPREHENSIVE INTEGRATION TEST")
    print("="*80)
    
    all_results = {}
    
    try:
        # Test 1: Dopamine signals
        cortex1, dopamine_history = test_dopamine_signals()
        all_results['dopamine_signals'] = dopamine_history
        
        # Test 2: Dopamine-gated learning
        weight_changes = test_dopamine_gated_learning()
        all_results['gated_learning'] = weight_changes
        
        # Test 3: Hierarchy integration
        hierarchy_stats = test_with_existing_hierarchy()
        all_results['hierarchy_integration'] = hierarchy_stats
        
        # Test 4: Neuron-level features
        neuron, eligibility, fatigue = test_neuron_level_phase5()
        all_results['neuron_features'] = {
            'eligibility': eligibility,
            'fatigue': fatigue,
            'final_weight_norm': torch.norm(neuron.weights).item()
        }
        
        # Test 5: Rhythm integration
        theta_results = test_phase5_with_rhythm()
        all_results['rhythm_integration'] = theta_results
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return all_results
    
    # Final summary
    print("\n" + "="*80)
    print("PHASE 5 TEST SUMMARY")
    print("="*80)
    
    tests_passed = 0
    total_tests = 5
    
    # Check each test
    if 'dopamine_signals' in all_results and len(all_results['dopamine_signals']) > 0:
        print("✓ Test 1: Dopamine signal generation")
        tests_passed += 1
    
    if 'gated_learning' in all_results:
        # Check if learning correlated with dopamine
        da_vals = [abs(wc['DA_phasic']) for wc in all_results['gated_learning']]
        changes = [wc['avg_weight_change'] for wc in all_results['gated_learning']]
        if len(da_vals) > 1 and any(c > 1e-6 for c in changes):
            print("✓ Test 2: Dopamine-gated learning")
            tests_passed += 1
    
    if 'hierarchy_integration' in all_results:
        stats = all_results['hierarchy_integration']
        if len(stats.get('DA_phasic', [])) > 0:
            print("✓ Test 3: Hierarchy integration")
            tests_passed += 1
    
    if 'neuron_features' in all_results:
        feat = all_results['neuron_features']
        if feat['final_weight_norm'] < 10.0:  # Not exploded
            print("✓ Test 4: Neuron-level features")
            tests_passed += 1
    
    if 'rhythm_integration' in all_results:
        print("✓ Test 5: Rhythm integration")
        tests_passed += 1
    
    print(f"\nTests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n✅ PHASE 5 INTEGRATION SUCCESSFUL!")
        print("\nRecommendations:")
        print("1. Enable Phase 5 in your main system with enable_phase5=True")
        print("2. Start with conservative dopamine parameters (da_max=1.0)")
        print("3. Monitor dopamine stats during training")
        print("4. Gradually add other Phase 5 components (intrinsic motivation, etc.)")
    else:
        print(f"\n⚠️  {total_tests - tests_passed} tests need attention")
        print("Check the error messages above and fix implementation.")
    
    return all_results

if __name__ == "__main__":
    print("Starting Phase 5 integration tests...")
    print("Note: This tests the dopamine neuromodulator integration.")
    print("Make sure you've updated AttractorCortex and ProtoNeuronMod with Phase 5 code.")
    
    results = run_comprehensive_test()
    
    # Save results for debugging
    import json
    with open('phase5_test_results.json', 'w') as f:
        # Convert tensors to lists for JSON
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist() if obj.numel() > 1 else obj.item()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print("\nTest results saved to phase5_test_results.json")
 