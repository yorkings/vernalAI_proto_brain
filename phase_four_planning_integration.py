"""
Phase 4: Planning Integration
Wrapper functions that add goal-oriented planning to existing hierarchical system
"""

import torch
from typing import Dict, Optional
from phase_three_integrated_life_loop import hierarchical_step
import numpy as np
from phase_four_action_momentum import ActionMomentum,LoopDetector
from phase_four_chain import ChainBuilder
from phase_four_memory_transfer import TemplateMemory
import math
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

def enhanced_life_loop_tick(system, observation, step_counter, prev_reward):

    """
    Integrated Phase 4 planning and chain execution
    """
    # 1. Process reward and get dopamine
    dopamine_signal = 0.0
    if prev_reward is not None:
        dopamine_signal = system['cortices'][0].process_reward(prev_reward)
    
    # 2. Get current cortical state
    cortical_state = system['hierarchy'].step(
        sensory_input=observation,
        delta_da=dopamine_signal,
        plasticity_gate=1.0,
        inference_iterations=2
    )
    
    # 3. Get goal (simplified - could be from GoalSystem)
    # For now, create a simple goal: reduce distance to origin

        # Try different goal strategies
    goal_strategy = "mixed"  # Try: "zero", "exploration", "oscillation", "negative"
    
    if goal_strategy == "zero":
        goal_state = torch.zeros_like(cortical_state)
    elif goal_strategy == "exploration":
        # Explore: move away from current state
        goal_state = cortical_state + torch.randn_like(cortical_state) * 0.3
    elif goal_strategy == "oscillation":
        # Oscillating goal
        t = step_counter * 0.1
        goal_state = torch.ones_like(cortical_state) * 0.2 * math.sin(t)
    elif goal_strategy == "exploitation":
        # Pure exploitation: try to move toward position 0
        # We need to know what "position 0" looks like in cortical space
        # For now, assume negative first dimension means "left/move toward 0"
        goal_state = cortical_state.clone()
        goal_state[0] = -0.5  # Encourage moving left (toward negative)
        if len(goal_state) > 1:
            goal_state[1] = 0.0  # Stabilize second dimension    
    elif goal_strategy == "negative":
        # Try to move to negative state (might encourage action 0)
        goal_state = -torch.ones_like(cortical_state) * 0.5
    elif goal_strategy == "mixed":
        # 70% exploitation, 30% exploration
        if torch.rand(1).item() < 0.7:
            # Exploitation: move toward goal
            goal_state = cortical_state.clone()
            goal_state[0] = -0.3  # Gently encourage left movement
        else:
            # Exploration: random direction
            goal_state = cortical_state + torch.randn_like(cortical_state) * 0.2
    
    print(f"[GOAL] Strategy: {goal_strategy}, State[0]={cortical_state[0]:.3f}, Goal[0]={goal_state[0]:.3f}")
    
    # # Normalize goal
    # goal_norm = goal_state.norm().item()
    # if goal_norm > 1.0:
    #     goal_state = goal_state / goal_norm
    
    # print(f"[GOAL] Strategy: {goal_strategy}, State[0]={cortical_state[0]:.3f}, Goal[0]={goal_state[0]:.3f}")
    
    
    # 4. Compute fingerprint for template lookup
    if 'fingerprint_column' in system:
        fingerprint = system['fingerprint_column'].forward(cortical_state)
    else:
        # Simple fingerprint: just the state itself
        fingerprint = cortical_state.clone()
    
    # 5. Look for similar chain templates
    chain = None
    predicted_states = None
    
    if 'template_memory' in system:
        template = system['template_memory'].find_similar(fingerprint)
        if template:
            chain = template['chain']
            # For now, skip adaptation - implement later
            # chain = adapt_chain(chain, cortical_state, system['forward_sim'])
    
    # 6. Build new chain if no template found
    if chain is None and 'chain_builder' in system:
        chain, predicted_states = system['chain_builder'].build_chain(
            state=cortical_state,
            goal=goal_state,
            momentum_system=system['action_momentum']
        )
    
    # 7. Score and select action (first in chain)
    if chain:
        # Get momentum bonus for the first action
        momentum_bonus = system['action_momentum'].get_momentum_bonus(chain[0])
        
        # Convert action index to actual action
        action_map = {0: -1, 1: 0, 2: 1}  # Your action mapping
        final_action = action_map.get(chain[0], 0)
        
        # Update momentum
        system['action_momentum'].update(chain[0])
        
        # Check for loops
        if predicted_states and 'loop_detector' in system:
            is_loop = system['loop_detector'].check_loop(predicted_states)
            if is_loop:
                # Apply loop breaking
                system['goal_trace'] *= 0.97  # Decay goal
                # Add noise to momentum
                noise = torch.randn_like(system['action_momentum'].momentum) * 0.01
                system['action_momentum'].momentum += noise
                # Possibly choose random action
                if torch.rand(1).item() < 0.3:
                    final_action = np.random.choice([-1, 0, 1])
    else:
        # Fallback to basic action
        final_action = 0  # Default
    # DEBUG: Check what components exist
    if step_counter == 0:
        print(f"[DEBUG] System has chain_builder: {'chain_builder' in system}")
        print(f"[DEBUG] System has forward_sim: {'forward_sim' in system}")
        print(f"[DEBUG] System has template_memory: {'template_memory' in system}")    
    
    # 8. Execute action and observe result
    # (This happens in your environment step)
    
    # 9. Update chain memory based on success
    # (Implement after seeing the outcome)
    if chain and step_counter > 0 and 'template_memory' in system:
        # Simple success heuristic: if we moved toward a "better" state
        # For now, use random success for testing
        import random
        if random.random() < 0.3:  # 30% chance to store as "successful"
            system['template_memory'].store_template(
                fingerprint=fingerprint,
                chain=chain,
                success_count=1
            )
            if step_counter % 10 == 0:  # Print occasionally
                print(f"[MEMORY] Stored chain {chain} at step {step_counter}")
    
    return {
        'final_action': final_action,
        'cortical_state': cortical_state,
        'chain_used': chain if chain else [],
        'dopamine_signal': dopamine_signal,
        'goal_state': goal_state
    }
def create_phase4_system(base_system, config=None):
    """
    Enhance existing Phase 3 system with Phase 4 features
    """
    if config is None:
        config = {
            'lambda_m': 0.85,
            'm_scale': 0.5,
            'beta_trend': 0.8,
            'chain_length': 4,
            'similarity_threshold': 0.8,
            'memory_capacity': 200
        }
    
    # Add Phase 4 components to existing system
    phase4_system = base_system.copy()
    
    # Action Momentum
    phase4_system['action_momentum'] = ActionMomentum(
        num_actions=3,  # Left, Center, Right
        lambda_m=config['lambda_m'],
        m_scale=config['m_scale']
    )
    
    # Loop Detector
    phase4_system['loop_detector'] = LoopDetector(
        registry_size=8,
        loop_penalty=0.5
    )
    
    # Chain Builder
    if 'forward_sim' in phase4_system:
        actions = [
            torch.tensor([1.0, 0.0, 0.0]),  # Left
            torch.tensor([0.0, 1.0, 0.0]),  # Center
            torch.tensor([0.0, 0.0, 1.0]),  # Right
        ]
        
        phase4_system['chain_builder'] = ChainBuilder(
            forward_sim=phase4_system['forward_sim'],
            actions=actions,
            K=config['chain_length']
        )
    else:
                # CREATE A SIMPLE MOCK FORWARD SIMULATOR
        print("[WARNING] No forward_sim found, creating mock for testing")
        class MockForwardSim:
            def imagine_step(self, state, action_context):
                new_state = state.clone()
                
                action_idx = torch.argmax(action_context).item()
                
                # Create meaningful state transitions
                # Action 0 (Left): Decrease first dimension, affect others
                if action_idx == 0:
                    new_state[0] = new_state[0] - 0.15
                    if len(new_state) > 1:
                        new_state[1] = new_state[1] * 0.8  # Dampen second dimension
                # Action 1 (Center): Small random walk
                elif action_idx == 1:
                    new_state = new_state * 0.95  # Slight decay
                # Action 2 (Right): Increase first dimension
                else:
                    new_state[0] = new_state[0] + 0.15
                    if len(new_state) > 1:
                        new_state[1] = new_state[1] * 1.2  # Amplify second dimension
                
                # Add small noise
                noise = torch.randn_like(new_state) * 0.05
                return new_state + noise
                        
        mock_sim = MockForwardSim()
        actions = [
            torch.tensor([1.0, 0.0, 0.0]),  # Left
            torch.tensor([0.0, 1.0, 0.0]),  # Center
            torch.tensor([0.0, 0.0, 1.0]),  # Right
        ]
        phase4_system['chain_builder'] = ChainBuilder(
            forward_sim=mock_sim,
            actions=actions,
            K=config['chain_length']
        )

    # Template Memory
    phase4_system['template_memory'] = TemplateMemory(
        capacity=config['memory_capacity'],
        similarity_threshold=config['similarity_threshold']
    )
    
    # Simple goal trace (initialize)
    phase4_system['goal_trace'] = torch.zeros(phase4_system['hierarchy'].levels[-1].dim)
    
    return phase4_system

def test_phase4_features():
    """Test each Phase 4 component"""
    print("Testing Phase 4 Features...")
    print("=" * 60)
    
    # 1. Create base Phase 3 system
    from phase_three_integrated_life_loop import create_hierarchical_system
    base_system = create_hierarchical_system(
        input_size=10,
        base_columns=4,
        hierarchy_factor=0.75,
        neurons_per_column=5,
        enable_planning=False  # We'll add our own
    )
    
    # 2. Enhance with Phase 4
    phase4_system = create_phase4_system(base_system)
    
    print(f"✓ Phase 4 system created with:")
    print(f"  - Action momentum tracking")
    print(f"  - Loop detection")
    print(f"  - Chain building (K={phase4_system['chain_builder'].K})")
    print(f"  - Template memory (capacity={phase4_system['template_memory'].capacity})")
    
    # 3. Test chain building
    test_state = torch.randn(phase4_system['cortices'][0].C)
    test_goal = torch.zeros_like(test_state)
    
    print(f"\nTesting chain building...")
    chain, predicted_states = phase4_system['chain_builder'].build_chain(
        state=test_state,
        goal=test_goal,
        momentum_system=phase4_system['action_momentum']
    )
    
    print(f"  Built chain: {chain}")
    print(f"  Chain length: {len(chain)}")
    print(f"  Predicted states: {len(predicted_states)}")
    
    # 4. Test loop detection
    print(f"\nTesting loop detection...")
    is_loop = phase4_system['loop_detector'].check_loop(predicted_states)
    print(f"  First check: Loop detected = {is_loop}")
    
    # Check again (should be a loop)
    is_loop2 = phase4_system['loop_detector'].check_loop(predicted_states)
    print(f"  Second check (same chain): Loop detected = {is_loop2}")
    
    # 5. Test momentum update
    print(f"\nTesting action momentum...")
    print(f"  Initial momentum: {phase4_system['action_momentum'].momentum}")
    
    if chain:
        phase4_system['action_momentum'].update(chain[0])
        print(f"  After updating with action {chain[0]}: {phase4_system['action_momentum'].momentum}")
    
    # 6. Test template memory
    print(f"\nTesting template memory...")
    fingerprint = test_state  # Using state as fingerprint for testing
    phase4_system['template_memory'].store_template(
        fingerprint=fingerprint,
        chain=chain,
        success_count=1
    )
    
    template = phase4_system['template_memory'].find_similar(fingerprint)
    if template:
        print(f"  Template found: success_count={template['success_count']}")
    else:
        print(f"  No template found (should be found)")
    
    print("\n" + "=" * 60)
    print("Phase 4 Features Test Complete!")
    print("Ready for integration into life loop.")
    
    return phase4_system

# Then update the main block:
if __name__ == "__main__":
    # Run both tests
    test_planning_integration()
    print("\n" + "=" * 60)
    verify_planning_impact()

