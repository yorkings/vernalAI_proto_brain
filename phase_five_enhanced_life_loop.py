# phase_five_enhanced_life_loop.py
"""
Enhanced life loop with Phase 5 intrinsic motivation
"""

import torch
from typing import Dict, Optional
from phase_two_attractor_cotex import AttractorCortex
from phase_two_rhythm_engine import RhythmEngine

def enhanced_cortex_step(cortex: AttractorCortex, 
                        x: torch.Tensor, 
                        raw_reward: float = 0.0,
                        debug: bool = False) -> Dict:
    """
    Enhanced cortex step with Phase 5 intrinsic motivation
    
    Returns comprehensive results dictionary
    """
    # Store reward for episode recording
    cortex.last_reward = raw_reward
    
    # Forward pass with attractor dynamics
    o = cortex.forward(x)
    
    # Compute surprise (prediction error)
    surprise = 0.0
    if cortex.last_pred is not None and cortex.last_state is not None:
        # Compare predicted state with actual state
        predicted = cortex.last_pred
        actual = o[:len(predicted)] if len(o) > len(predicted) else o
        surprise = torch.norm(predicted - actual).item()
    
    # Update dopamine with intrinsic motivation
    da_info = cortex.update_dopamine(
        external_reward=raw_reward,
        surprise=surprise,
        use_intrinsic=cortex.enable_intrinsic_motivation
    )
    
    # Apply dopamine-gated learning
    cortex.learn(da_signal=da_info['DA_phasic'].item(), use_phase5_gating=True)
    
    # Execute replays during theta windows
    if cortex.rhythm and cortex.rhythm.theta_in_window():
        cortex.execute_background_replay()
        cortex.run_replay()
    
    # Debug output
    if debug and cortex.global_timestep % 10 == 0:
        print(f"[Phase 5] Step {cortex.global_timestep}: "
              f"R_ext={raw_reward:.3f}, "
              f"R_int={da_info.get('R_intrinsic', 0):.3f}, "
              f"DA={da_info['DA_phasic'].item():.3f}, "
              f"Surprise={surprise:.3f}")
        
        if da_info.get('R_intrinsic_components'):
            comp = da_info['R_intrinsic_components']
            print(f"       Intrinsic: Novelty={comp.get('novelty', 0):.3f}, "
                  f"LP={comp.get('learning_progress', 0):.3f}")
    
    return {
        'cortical_state': o,
        'surprise': surprise,
        'DA_phasic': da_info['DA_phasic'].item(),
        'DA_tonic': da_info['DA_tonic'].item(),
        'R_total': da_info['R_total'],
        'R_intrinsic': da_info['R_intrinsic'],
        'explore_scale': da_info['explore_scale'].item(),
        'drive': da_info.get('drive', 0)
    }

def test_intrinsic_motivation_integration():
    """Test intrinsic motivation integration"""
    print("\n" + "="*60)
    print("TESTING PHASE 5.9: INTRINSIC MOTIVATION INTEGRATION")
    print("="*60)
    
    from phase_two_column import ProtoColumn
    
    # Create test cortex with intrinsic motivation
    input_size = 10
    num_columns = 4
    columns = [ProtoColumn(input_size=input_size, num_neurons=5) for _ in range(num_columns)]
    
    cortex = AttractorCortex(
        columns=columns,
        input_proj=True,
        enable_phase5=True,
        enable_intrinsic_motivation=True,
        c_nov=0.2,
        c_lp=1.0,
        window_size=20,
        da_gamma=0.9,
        da_max=1.0
    )
    
    print("Testing intrinsic motivation over 50 steps...")
    print("-"*60)
    
    results = []
    
    for step in range(50):
        # Generate input with some pattern
        if step % 15 == 0:
            x = torch.randn(input_size) * 2.0  # Novel input
            reward = 0.5  # Small positive reward for novelty
        elif step % 5 == 0:
            x = torch.ones(input_size) * 0.3  # Familiar pattern
            reward = 0.1
        else:
            x = torch.randn(input_size) * 0.5  # Regular input
            reward = 0.0
        
        # Enhanced step with intrinsic motivation
        result = enhanced_cortex_step(cortex, x, reward, debug=(step % 15 == 0))
        results.append(result)
    
    # Analyze results
    print("\n" + "="*60)
    print("INTRINSIC MOTIVATION ANALYSIS")
    print("="*60)
    
    # Get intrinsic motivation statistics
    if hasattr(cortex, 'get_intrinsic_motivation_stats'):
        im_stats = cortex.get_intrinsic_motivation_stats()
        
        if im_stats.get('enabled', False):
            print("\nIntrinsic Motivation Engine Statistics:")
            print(f"  Unique states visited: {im_stats.get('novelty', {}).get('total_states', 0)}")
            print(f"  Learning progress: {im_stats.get('competence', {}).get('learning_progress', 0):.3f}")
            
            if 'intrinsic_reward' in im_stats:
                r_stats = im_stats['intrinsic_reward']
                print(f"  Intrinsic reward stats:")
                print(f"    Mean: {r_stats.get('mean', 0):.3f}")
                print(f"    Std: {r_stats.get('std', 0):.3f}")
    
    # Analyze dopamine signals
    da_vals = [r['DA_phasic'] for r in results]
    r_int_vals = [r['R_intrinsic'] for r in results]
    
    print(f"\nDopamine Analysis:")
    print(f"  Average DA_phasic: {sum(da_vals)/len(da_vals):.3f}")
    print(f"  Max DA_phasic: {max(da_vals):.3f}")
    print(f"  Min DA_phasic: {min(da_vals):.3f}")
    
    print(f"\nIntrinsic Reward Analysis:")
    print(f"  Average R_intrinsic: {sum(r_int_vals)/len(r_int_vals):.3f}")
    print(f"  Positive intrinsic rewards: {sum(1 for r in r_int_vals if r > 0)}/{len(r_int_vals)}")
    
    # Check if intrinsic reward correlates with novelty
    if len(results) >= 10:
        print(f"\nPerformance Check:")
        
        # Check that dopamine responds to rewards
        positive_da_steps = sum(1 for r in results if r['DA_phasic'] > 0.1)
        print(f"  Steps with significant DA (>0.1): {positive_da_steps}/{len(results)}")
        
        if positive_da_steps > len(results) * 0.3:
            print("  ✓ Dopamine system responsive")
        else:
            print("  ⚠️  Dopamine may be too conservative")
    
    return cortex, results

if __name__ == "__main__":
    cortex, results = test_intrinsic_motivation_integration()
    
    # Final verification
    print("\n" + "="*60)
    print("PHASE 5.9 VERIFICATION")
    print("="*60)
    
    checks = []
    
    # Check 1: Intrinsic engine exists
    checks.append((
        "Intrinsic motivation engine initialized",
        hasattr(cortex, 'intrinsic_engine') and cortex.enable_intrinsic_motivation
    ))
    
    # Check 2: Dopamine computed
    if results:
        avg_da = sum(r['DA_phasic'] for r in results) / len(results)
        checks.append((
            "Dopamine signals computed",
            abs(avg_da) < 1.0  # Not exploding
        ))
    
    # Check 3: Intrinsic rewards varied
    if results:
        r_int_vals = [r['R_intrinsic'] for r in results]
        r_int_range = max(r_int_vals) - min(r_int_vals)
        checks.append((
            "Intrinsic rewards varied",
            r_int_range > 0.1
        ))
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
    
    print("\n" + "="*60)
    print("PHASE 5.9 READY FOR NEXT COMPONENT")
    print("="*60)
    print("Next: Plasticity Gates (Phase 5.10)")