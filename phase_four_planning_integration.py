"""
Phase 4: Planning Integration
Wrapper functions that add goal-oriented planning to existing hierarchical system
"""

import torch
from typing import Dict, Optional
from phase_three_integrated_life_loop import hierarchical_step
import numpy as np

# Import planning components (you'll create these)
try:
    from phase_four_planning import ForwardSimulator, GoalSystem, BiologicalPlanner
    PLANNING_AVAILABLE = True
except ImportError:
    print("[WARNING] Planning modules not available. Running without planning.")
    PLANNING_AVAILABLE = False


def hierarchical_step_with_planning(
    system: Dict, 
    x: torch.Tensor, 
    agent_action: int, 
    step_counter: int,
    prev_reward: Optional[float] = None,
    goal_input: Optional[torch.Tensor] = None,
    planning_enabled: bool = True,
    planning_weight: float = 0.3
) -> Dict:
    """
    Enhanced hierarchical step with optional goal-oriented planning
    
    Args:
        system: Hierarchical system (must include planning components if enabled)
        x: Sensory input (state)
        agent_action: Baseline agent policy action
        step_counter: Current step count
        prev_reward: Reward from previous step (for learning)
        goal_input: Goal vector for planning (optional)
        planning_enabled: Whether to use planning (default: True)
        planning_weight: How much to weight planned action vs original (0-1)
    
    Returns:
        Dictionary with all original results + planning information
    """
    # 1. Get base hierarchical step result
    base_result = hierarchical_step(
        system=system,
        x=x,
        agent_action=agent_action,
        step_counter=step_counter,
        prev_reward=prev_reward
    )
    
    # 2. Add planning if enabled, goal provided, and planning components exist
    planning_used = False
    planned_action = None
    goal_trace = None
    plan_score = None
    
    if (planning_enabled and 
        goal_input is not None and
        PLANNING_AVAILABLE and
        'goal_system' in system and
        'planner' in system and
        'forward_sim' in system):
        
        try:
            # Update goal trace
            goal_trace = system['goal_system'].update_goal(goal_input)
            
            # Get current state for planning (from bottom cortex)
            current_state = system['cortices'][0].stable_state
            if current_state is None:
                current_state = system['cortices'][0].last_o
            
            # Plan best action
            if current_state is not None:
                planned_action, plan_score = system['planner'].plan(
                    current_state=current_state,
                    goal_trace=goal_trace
                )
                
                planning_used = True
                
                # Debug
                if step_counter % 50 == 0:
                    print(f"[PLANNING] Step {step_counter}: "
                          f"Planned action: {planned_action}, "
                          f"Score: {plan_score:.3f}, "
                          f"Goal trace norm: {goal_trace.norm():.3f}")
                
                # Blend planned action with original action
                if planned_action is not None:
                    # Convert planned action index to value (-1, 0, 1)
                    if planned_action == 0:  # Left
                        planned_value = -1.0
                    elif planned_action == 1:  # Center
                        planned_value = 0.0
                    else:  # Right
                        planned_value = 1.0
                     # DEBUG: Print what's happening
                    print(f"[ACTION BLENDING] Step {step_counter}:")
                    print(f"  Original action: {base_result['final_action']} (value: {float(base_result['final_action'])})")
                    print(f"  Planned action: {planned_action} (value: {planned_value})")
                    print(f"  Planning weight: {planning_weight}")     
                    
                    # Blend with original action
                    original_value = float(base_result['final_action'])
                    blended_value = (1 - planning_weight) * original_value + planning_weight * planned_value

                    threshold = 0.2
                    # Convert back to discrete action
                    if blended_value > threshold:
                        final_action = 1
                    elif blended_value < -threshold:
                        final_action = -1
                    else:
                        final_action = 0
                    
                    # Update result
                    base_result['final_action'] = final_action
                    base_result['blended_action'] = blended_value
                    
        except Exception as e:
            print(f"[PLANNING ERROR] Step {step_counter}: {e}")
            planning_used = False
    
    # 3. Add planning information to result
    base_result.update({
        'planning_used': planning_used,
        'planned_action': planned_action,
        'plan_score': plan_score,
        'goal_trace': goal_trace,
        'planning_weight': planning_weight if planning_used else 0.0,
        'planning_enabled': planning_enabled
    })
    
    return base_result


def create_planning_system(
    base_system: Dict,
    input_size: int = 10,
    goal_dim: int = 32,
    lookahead_steps: int = 3
) -> Dict:
    """
    Add planning capabilities to an existing hierarchical system
    
    Args:
        base_system: Existing hierarchical system from create_hierarchical_system()
        input_size: Size of goal input vectors
        goal_dim: Dimension of goal representation
        lookahead_steps: How many steps to simulate ahead
    
    Returns:
        Enhanced system with planning components
    """
    if not PLANNING_AVAILABLE:
        print("[WARNING] Cannot add planning - modules not available")
        return base_system
    
    try:
        # Create forward simulator using bottom cortex
        forward_sim = ForwardSimulator(
            cortical_block=base_system['cortices'][0],
            lookahead_steps=lookahead_steps
        )
        
        # Create goal system
        goal_system = GoalSystem(
            input_size=input_size,
            goal_dim=goal_dim
        )
        
        # Create planner
        planner = BiologicalPlanner(
            forward_sim=forward_sim,
            goal_system=goal_system
        )
        
        # Add planning components to system
        enhanced_system = base_system.copy()
        enhanced_system.update({
            'forward_sim': forward_sim,
            'goal_system': goal_system,
            'planner': planner,
            'has_planning': True
        })
        
        print(f"[PLANNING] Added planning capabilities to system")
        print(f"          Lookahead steps: {lookahead_steps}")
        print(f"          Goal dimension: {goal_dim}")
        
        return enhanced_system
        
    except Exception as e:
        print(f"[ERROR] Failed to add planning: {e}")
        return base_system


def compare_planning_vs_no_planning(
    system_with_planning: Dict,
    system_without_planning: Dict,
    env,
    num_steps: int = 200
) -> Dict:
    """
    Compare performance with vs without planning
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        'with_planning': {'rewards': [], 'positions': [], 'actions': []},
        'without_planning': {'rewards': [], 'positions': [], 'actions': []}
    }
    
    def run_test(system, use_planning: bool, label: str):
        """Run a test with given system and planning setting"""
        reward_from_previous = None
        
        for step in range(num_steps):
            x = env.get_state_vector()
            goal = env.get_goal_vector() if hasattr(env, 'get_goal_vector') else None
            
            agent_action = 0  # Simple baseline
            
            if use_planning:
                result = hierarchical_step_with_planning(
                    system=system,
                    x=x,
                    agent_action=agent_action,
                    step_counter=step,
                    prev_reward=reward_from_previous,
                    goal_input=goal,
                    planning_enabled=True
                )
            else:
                result = hierarchical_step(
                    system=system,
                    x=x,
                    agent_action=agent_action,
                    step_counter=step,
                    prev_reward=reward_from_previous
                )
            
            current_reward = env.step(result['final_action'])
            reward_from_previous = current_reward
            
            # Store results
            results[label]['rewards'].append(current_reward)
            results[label]['positions'].append(env.position)
            results[label]['actions'].append(result['final_action'])
            
            # Reset environment for next test
            if step == num_steps - 1:
                env.reset()
    
    # Run both tests
    print("\n" + "=" * 60)
    print("PLANNING COMPARISON TEST")
    print("=" * 60)
    
    # Test WITH planning
    print("\n1. Testing WITH planning...")
    env.reset()
    run_test(system_with_planning, use_planning=True, label='with_planning')
    
    # Test WITHOUT planning
    print("\n2. Testing WITHOUT planning...")
    env.reset()
    run_test(system_without_planning, use_planning=False, label='without_planning')
    
    # Analyze results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    for label in ['with_planning', 'without_planning']:
        rewards = results[label]['rewards']
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        final_pos = results[label]['positions'][-1] if results[label]['positions'] else 0
        
        print(f"\n{label.replace('_', ' ').title()}:")
        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  Final position: {final_pos:.3f}")
        print(f"  Distance to goal: {abs(final_pos):.3f}")
    
    return results


# Simple test function
def test_planning_integration():
    """Quick test of planning integration"""
    from phase_three_integrated_life_loop import create_hierarchical_system, TestEnvironment
    
    print("Testing Planning Integration")
    print("=" * 60)
    
    # Create base system
    base_system = create_hierarchical_system(
        input_size=10,
        base_columns=4,
        hierarchy_factor=0.75,
        neurons_per_column=5
    )
    
    # Add planning
    if PLANNING_AVAILABLE:
        planning_system = create_planning_system(
            base_system=base_system,
            input_size=10,
            goal_dim=32,
            lookahead_steps=3
        )
        
        # Create simple goal
        goal = torch.randn(10) * 0.5
        
        # Test one step
        env = TestEnvironment(input_size=10)
        x = env.get_state_vector()
        
        result = hierarchical_step_with_planning(
            system=planning_system,
            x=x,
            agent_action=0,
            step_counter=0,
            prev_reward=None,
            goal_input=goal,
            planning_enabled=True,
            planning_weight=0.5  # INCREASE TO 0.5
        )
        
        print(f"\nPlanning test result:")
        print(f"  Final action: {result['final_action']}")
        print(f"  Planning used: {result.get('planning_used', False)}")
        print(f"  Planned action: {result.get('planned_action', 'N/A')}")
        print(f"  Planning weight used: {result.get('planning_weight', 0.0)}")
        
        if result.get('planning_used', False):
            if result['final_action'] != 0:  # Action actually changed
                print("✅ Planning integration successful AND action changed!")
            else:
                print("⚠️  Planning used but action didn't change (check blending)")
        else:
            print("❌ Planning not used (needs debugging)")
    else:
        print("Planning modules not available. Create phase_four_planning.py first.")

def verify_planning_impact():
    """Verify planning actually improves performance"""
    from phase_three_integrated_life_loop import create_hierarchical_system, TestEnvironment
    
    print("\n" + "=" * 60)
    print("PLANNING IMPACT VERIFICATION")
    print("=" * 60)
    
    # Create base system
    base_system = create_hierarchical_system(
        input_size=10,
        base_columns=4,
        hierarchy_factor=0.75,
        neurons_per_column=5
    )
    
    # Add planning to create a system that CAN plan
    planning_system = create_planning_system(
        base_system=base_system,
        input_size=10,
        goal_dim=32,
        lookahead_steps=3
    )
    
    total_reward_with_planning = 0
    total_reward_without_planning = 0
    
    # Test both conditions
    for planning_enabled in [True, False]:
        env = TestEnvironment(input_size=10)
        reward_from_previous = None
        
        print(f"\nTesting {'WITH' if planning_enabled else 'WITHOUT'} planning:")
        
        for step in range(20):  # 20 steps for better statistics
            x = env.get_state_vector()
            
            # Create a simple goal: move toward position 0
            goal = torch.zeros(10)
            goal[0] = -np.sign(env.position) * 0.5  # Direction to goal
            goal[1] = 0.3  # Goal intensity
            
            result = hierarchical_step_with_planning(
                system=planning_system,
                x=x,
                agent_action=0,  # Simple baseline
                step_counter=step,
                prev_reward=reward_from_previous,
                goal_input=goal if planning_enabled else None,
                planning_enabled=planning_enabled,
                planning_weight=0.5
            )
            
            reward = env.step(result['final_action'])
            reward_from_previous = reward
            
            if step % 5 == 0:
                print(f"  Step {step}: pos={env.position:.2f}, "
                      f"reward={reward:.2f}, action={result['final_action']}")
            
            if planning_enabled:
                total_reward_with_planning += reward
            else:
                total_reward_without_planning += reward
    
    print(f"\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nWith planning: {total_reward_with_planning:.2f}")
    print(f"Without planning: {total_reward_without_planning:.2f}")
    
    if total_reward_without_planning != 0:
        improvement = ((total_reward_with_planning - total_reward_without_planning) 
                      / abs(total_reward_without_planning) * 100)
        print(f"Improvement: {improvement:.1f}%")
        
        if improvement > 0:
            print("✅ Planning improves performance!")
        else:
            print("⚠️  Planning doesn't improve (may need tuning)")
    else:
        print("Cannot compute improvement (zero baseline)")

# Then update the main block:
if __name__ == "__main__":
    # Run both tests
    test_planning_integration()
    print("\n" + "=" * 60)
    verify_planning_impact()

