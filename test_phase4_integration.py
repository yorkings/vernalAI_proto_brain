"""
Test Phase 4 Integration
"""
import torch
import numpy as np
import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_three_integrated_life_loop import create_hierarchical_system, TestEnvironment
from phase_four_planning_integration import (
    create_phase4_system,
    test_phase4_features,
    enhanced_life_loop_tick
)

def test_complete_phase4_life_loop():
    """Test the complete Phase 4 system in action"""
    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE SYSTEM TEST")
    print("=" * 80)
    
    # 1. Create base system
    print("\n1. Creating base Phase 3 system...")
    base_system = create_hierarchical_system(
        input_size=10,
        base_columns=4,
        hierarchy_factor=0.75,
        neurons_per_column=5,
        enable_planning=False
    )
    
    # 2. Add Phase 4 features
    print("\n2. Enhancing with Phase 4 features...")
    phase4_system = create_phase4_system(base_system)
    
    # 3. Create environment
    env = TestEnvironment(input_size=10)
    
    # 4. Run enhanced life loop
    print("\n3. Running enhanced life loop (50 steps)...")
    print("-" * 60)
    
    stats = {
        'rewards': [],
        'actions': [],
        'chains_used': [],
        'loops_detected': 0,
        'templates_used': 0
    }
    
    prev_reward = None
    
    for step in range(50):
        # Get observation
        x = env.get_state_vector()
        
        # Use enhanced life loop
        result = enhanced_life_loop_tick(
            system=phase4_system,
            observation=x,
            step_counter=step,
            prev_reward=prev_reward
        )
        
        # Execute action
        current_reward = env.step(result['final_action'])
        prev_reward = current_reward
        
        # Collect stats
        stats['rewards'].append(current_reward)
        stats['actions'].append(result['final_action'])
        stats['chains_used'].append(len(result['chain_used']) if result['chain_used'] else 0)
        
        # Check if loop was detected
        if 'loop_detected' in result:
            stats['loops_detected'] += 1
        
        # Check if template was used
        if result['chain_used'] and 'template_used' in result:
            stats['templates_used'] += 1
        
        # Periodic reporting
        if step % 10 == 0:
            print(f"Step {step:3d}: "
                  f"Pos={env.position:6.2f}, "
                  f"Reward={current_reward:6.2f}, "
                  f"Action={result['final_action']}, "
                  f"Chain_len={len(result['chain_used']) if result['chain_used'] else 0}")
    
    # 5. Analyze results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    avg_reward = np.mean(stats['rewards'])
    avg_chain_length = np.mean([c for c in stats['chains_used'] if c > 0]) if any(stats['chains_used']) else 0
    
    print(f"\nPerformance Metrics:")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Final position: {env.position:.3f} (goal: 0.0)")
    print(f"  Chains used: {sum(1 for c in stats['chains_used'] if c > 0)}/{len(stats['chains_used'])} steps")
    print(f"  Average chain length: {avg_chain_length:.2f}")
    print(f"  Loops detected: {stats['loops_detected']}")
    print(f"  Templates used: {stats['templates_used']}")
    
    # Check action distribution
    actions = np.array(stats['actions'])
    action_counts = {action: np.sum(actions == action) for action in [-1, 0, 1]}
    print(f"\nAction Distribution:")
    for action, count in action_counts.items():
        print(f"  Action {action}: {count} times ({count/len(actions)*100:.1f}%)")
    
    # Check system stability
    print(f"\nSystem Health Checks:")
    
    # Check momentum
    momentum_norm = phase4_system['action_momentum'].momentum.norm().item()
    print(f"  Momentum norm: {momentum_norm:.3f} {'✓' if 0 < momentum_norm < 10 else '⚠'}")
    
    # Check template memory
    template_count = len(phase4_system['template_memory'].templates)
    print(f"  Templates stored: {template_count}")
    
    # Check loop registry
    loop_registry_size = len(phase4_system['loop_detector'].registry)
    print(f"  Loop registry size: {loop_registry_size}")
    
    # Check goal trace
    goal_norm = phase4_system['goal_trace'].norm().item()
    print(f"  Goal trace norm: {goal_norm:.3f}")
    
    print("\n" + "=" * 80)
    if avg_reward > -2.0 and template_count > 0:
        print("✅ PHASE 4 SYSTEM IS WORKING!")
    else:
        print("⚠️  System needs tuning")
    
    return stats, phase4_system

def debug_phase4_components():
    """Debug individual Phase 4 components"""
    print("\n" + "=" * 80)
    print("PHASE 4 COMPONENT DEBUG")
    print("=" * 80)
    
    # Test component imports
    print("\n1. Testing component imports...")
    try:
        from phase_four_action_momentum import ActionMomentum, LoopDetector
        from phase_four_chain import ChainBuilder
        from phase_four_memory_transfer import TemplateMemory
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Test ActionMomentum
    print("\n2. Testing ActionMomentum...")
    momentum = ActionMomentum()
    print(f"  Initial momentum: {momentum.momentum}")
    momentum.update(0)
    print(f"  After updating action 0: {momentum.momentum}")
    bonus = momentum.get_momentum_bonus(0)
    print(f"  Momentum bonus for action 0: {bonus}")
    
    # Test LoopDetector
    print("\n3. Testing LoopDetector...")
    detector = LoopDetector()
    test_states = [torch.randn(4) for _ in range(3)]
    is_loop1 = detector.check_loop(test_states)
    is_loop2 = detector.check_loop(test_states)  # Same states should be a loop
    print(f"  First check (new): {is_loop1}")
    print(f"  Second check (same): {is_loop2} {'✓' if is_loop2 else '✗'}")
    
    # Test TemplateMemory
    print("\n4. Testing TemplateMemory...")
    memory = TemplateMemory(capacity=3)
    test_fp = torch.randn(4)
    memory.store_template(test_fp, [0, 1, 2])
    template = memory.find_similar(test_fp)
    print(f"  Template stored: {'✓' if template else '✗'}")
    print(f"  Template chain: {template['chain'] if template else 'None'}")
    
    print("\n" + "=" * 80)
    print("COMPONENT DEBUG COMPLETE")

if __name__ == "__main__":
    # Run debug first
    debug_phase4_components()
    
    # Then run complete test
    stats, system = test_complete_phase4_life_loop()
    
    # Additional analysis
    print("\n" + "=" * 80)
    print("ADDITIONAL ANALYSIS")
    print("=" * 80)
    
    # Check chain builder
    if 'chain_builder' in system:
        print("\nChain Builder Status:")
        print(f"  Lookahead steps (K): {system['chain_builder'].K}")
        print(f"  Number of actions: {len(system['chain_builder'].actions)}")
    
    # Check hierarchy state
    hierarchy_state = system['hierarchy'].get_hierarchy_state()
    print(f"\nHierarchy State:")
    for i, level in enumerate(hierarchy_state['levels']):
        print(f"  Level {i}: error={hierarchy_state['errors'][i]:.3f}, "
              f"precision={hierarchy_state['precisions'][i]:.3f}")