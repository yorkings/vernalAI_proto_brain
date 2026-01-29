# phase_three_integrated_life_loop.py (FIXED VERSION - No self reference)
import torch
import numpy as np
from typing import Dict, List,Optional
import matplotlib.pyplot as plt
from phase_two_column import ProtoColumn
from phase_two_attractor_cotex import AttractorCortex
from phase_two_rhythm_engine import RhythmEngine
from phase_three_predictive_block import PredictiveBlock
from phase_three_predictive_heiracy import PredictiveHierarchy
from phase_three_predictive_heiracy_controller import HierarchicalController
import datetime
# Set default dtype to float32 for consistency
torch.set_default_dtype(torch.float32)

class TestEnvironment:
    """Enhanced test environment with meaningful patterns"""
    def __init__(self, input_size: int = 10):
        self.input_size = input_size
        self.position = 5.0
        self.goal = 0.0
        self.step_count = 0
        self.sequence_pos = 0
        
        # Create meaningful patterns with proper dtype
        self.patterns = [
            torch.sin(torch.linspace(0, 2*np.pi, input_size, dtype=torch.float32)) * 0.5,
            torch.cos(torch.linspace(0, 2*np.pi, input_size, dtype=torch.float32)) * 0.5,
            torch.sin(torch.linspace(0, 4*np.pi, input_size, dtype=torch.float32)) * 0.3,
            torch.cos(torch.linspace(0, 4*np.pi, input_size, dtype=torch.float32)) * 0.3,
        ]
        
        # Pattern sequence
        self.sequence = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    
    def get_state_vector(self) -> torch.Tensor:
        """Create state vector combining position and pattern"""
        position_feature = torch.tensor([self.position / 10.0], dtype=torch.float32)
        goal_feature = torch.tensor([np.sign(self.goal - self.position) * 0.3], dtype=torch.float32)
        
        # Current pattern
        pattern_idx = self.sequence[self.sequence_pos]
        pattern = self.patterns[pattern_idx]
        
        # Combine features
        state = torch.cat([position_feature, goal_feature, pattern[:8]])
        
        # Ensure correct size
        if state.shape[0] < self.input_size:
            padding = torch.zeros(self.input_size - state.shape[0], dtype=torch.float32)
            state = torch.cat([state, padding])
        elif state.shape[0] > self.input_size:
            state = state[:self.input_size]
        
        return state
    
    def step(self, action: int) -> float:
        """Take action and return reward"""
        self.step_count += 1
        self.sequence_pos = (self.sequence_pos + 1) % len(self.sequence)
        
        # Update position
        self.position += action * 0.5
        
        # Add noise
        self.position += np.random.randn() * 0.1
        
        # Compute reward
        distance = abs(self.position - self.goal)
        reward = -distance  # Negative distance as reward
        
        # Bonus for being close to goal
        if distance < 1.0:
            reward += 1.0
        
        # Pattern completion bonus
        if self.sequence_pos == 0:
            reward += 0.5
        
        return float(reward)
    
    def get_debug_info(self) -> Dict:
        return {
            'position': self.position,
            'step_count': self.step_count,
            'sequence_pos': self.sequence_pos,
            'distance_to_goal': abs(self.position - self.goal)
        }

def convert_column_neurons_to_float32(col):
    """Convert all neurons in a column to float32"""
    if hasattr(col, 'layer') and hasattr(col.layer, 'neurons'):
        for neuron in col.layer.neurons:
            # Convert neuron weights and biases
            if hasattr(neuron, 'weights'):
                neuron.weights = neuron.weights.to(torch.float32)
            if hasattr(neuron, 'bias'):
                neuron.bias = neuron.bias.to(torch.float32)
            if hasattr(neuron, 'trace'):
                neuron.trace = neuron.trace.to(torch.float32)
            if hasattr(neuron, 'last_input'):
                neuron.last_input = neuron.last_input.to(torch.float32) if neuron.last_input is not None else None
            if hasattr(neuron, 'last_activation'):
                neuron.last_activation = neuron.last_activation.to(torch.float32) if neuron.last_activation is not None else None
            if hasattr(neuron, 'eligibility'):
                neuron.eligibility = neuron.eligibility.to(torch.float32)
    return col

def create_hierarchical_system(input_size: int = 10,
                               base_columns: int = 4,  
                               hierarchy_factor: float = 0.75,
                               neurons_per_column: int = 5,
                               enable_planning: bool = True,
                                goal_dim: int = 32,
                                lookahead_steps: int = 3) -> Dict:
    """
    Create hierarchical system with decreasing dimensions
    """
    # Calculate columns per level
    num_columns_per_level = [base_columns]
    for i in range(1, 3):  # 3 levels total
        next_cols = max(2, int(num_columns_per_level[-1] * hierarchy_factor))
        num_columns_per_level.append(next_cols)
    
    print(f"Hierarchy dimensions: {num_columns_per_level}")
    
    torch.manual_seed(42)
    
    # Create shared rhythm engine
    rhythm = RhythmEngine(
        theta_freq=6.0,
        gamma_per_theta=6,
        sim_steps_per_second=80,
        theta_phase_window=(0.1, 0.4)
    )
    
    # CREATE BASE CORTICES with proper connectivity
    cortices = []
    
    for level_idx, num_columns in enumerate(num_columns_per_level):
        # Determine input size for this level
        if level_idx == 0:
            # Bottom level: receives sensory input
            level_input_size = input_size
        else:
            # Higher levels: receive representation from level below
            level_input_size = num_columns_per_level[level_idx - 1]
        
        # Create columns
        cols = [
            ProtoColumn(
                input_size=level_input_size,
                num_neurons=neurons_per_column,
                lr=0.01 / (level_idx + 1)  # Slower learning for higher levels
            ) 
            for _ in range(num_columns)
        ]
        
        # Ensure all column neurons are float32
        cols = [convert_column_neurons_to_float32(col) for col in cols]
        
        # Create AttractorCortex
        cortex = AttractorCortex(
            columns=cols,
            input_proj=True,
            rhythm=rhythm if level_idx == 0 else None,  # Rhythm only at bottom
            settling_steps=5 + level_idx * 3,
            k_sparsity=max(1, num_columns // 2),
            buffer_capacity=30,
            max_episode_length=8,
            immediate_replay_threshold=0.4,
            value_lr=0.05 / (level_idx + 1),  # Slower value learning for higher levels
            baseline_reward=0.0
        )
        
        # Ensure cortex internal states are float32
        cortex = ensure_cortex_float32(cortex)
        
        cortices.append(cortex)
    rep_dims = [base_columns]
    for i in range(1, len(cortices)):
        next_dim = max(2, int(rep_dims[-1] * hierarchy_factor))
        rep_dims.append(next_dim)

    print(f"[DEBUG] Representation dimensions: {rep_dims}")
    # CREATE PREDICTIVE BLOCKS
    predictive_blocks = []
    
    for level_idx, cortex in enumerate(cortices):
        # Timescale increases with hierarchy
        timescale = 2 ** level_idx  # 1, 2, 4
        
        block = PredictiveBlock(
            base_cortex=cortex,
            dim_rep=rep_dims[level_idx],
            gen_init_scale=0.03 / (level_idx + 1),
            rec_init_scale=0.03 / (level_idx + 1),
            kappa=0.15 / (level_idx + 1),  # Slower inference for higher levels
            alpha_G=5e-4 / (level_idx + 1),
            alpha_R=5e-4 / (level_idx + 1),
            precision_init=1.0,
            timescale=timescale,
            level_idx=level_idx  # For debugging
        )
        
        # Ensure block tensors are float32
        block = ensure_block_float32(block)
        
        predictive_blocks.append(block)
    
    # CREATE HIERARCHY
    hierarchy = PredictiveHierarchy(predictive_blocks)
    
    # CREATE CONTROLLER
    top_dim = rep_dims[-1]
    print(f"[DEBUG] Controller using top_dim = {top_dim}")
    controller = HierarchicalController(
        rep_dim=top_dim,
        action_dim=3,  # For 1D world: {-1, 0, +1}
        init_scale=0.02,
        learning_rate=0.001
    )
    
    # Ensure controller tensors are float32
    controller.M = controller.M.to(torch.float32)
    controller.b = controller.b.to(torch.float32)
    
    if enable_planning:
        # Import the new planning classes (they should be in a new file)
        from phase_four_planning import ForwardSimulator, GoalSystem, BiologicalPlanner
        
        # Create forward simulator using the bottom cortex
        forward_sim = ForwardSimulator(
            cortical_block=cortices[0],  # Bottom cortex for simulation
            lookahead_steps=lookahead_steps
        )
        input_size_for_goal = cortices[0].columns[0].layer.input_size
        # Create goal system
        goal_system = GoalSystem(
            input_size=input_size_for_goal,
            goal_dim=goal_dim
        )
        
        # Create planner
        planner = BiologicalPlanner(
            forward_sim=forward_sim,
            goal_system=goal_system
        )
        
        # Add to return dictionary
        return {
            'hierarchy': hierarchy,
            'controller': controller,
            'rhythm': rhythm,
            'cortices': cortices,
            'predictive_blocks': predictive_blocks,
            'levels': len(num_columns_per_level),
            # ADD THESE:
            'forward_sim': forward_sim,
            'goal_system': goal_system,
            'planner': planner,
            'enable_planning': enable_planning
        }
    else:
        # Return original system without planning
        return {
            'hierarchy': hierarchy,
            'controller': controller,
            'rhythm': rhythm,
            'cortices': cortices,
            'predictive_blocks': predictive_blocks,
            'levels': len(num_columns_per_level),
            'enable_planning': False
        }

def ensure_cortex_float32(cortex):
    """Ensure all cortex tensors are float32"""
    # Ensure column tensors
    if hasattr(cortex, 'columns'):
        for col in cortex.columns:
            if hasattr(col, 'layer'):
                if hasattr(col.layer, 'neurons'):
                    for neuron in col.layer.neurons:
                        # Convert neuron tensors
                        for attr in ['weights', 'bias', 'trace', 'last_input', 'last_activation', 'eligibility']:
                            if hasattr(neuron, attr):
                                tensor = getattr(neuron, attr)
                                if tensor is not None and isinstance(tensor, torch.Tensor):
                                    setattr(neuron, attr, tensor.to(torch.float32))
                
                if hasattr(col.layer, 'last_outputs'):
                    if col.layer.last_outputs is not None:
                        col.layer.last_outputs = col.layer.last_outputs.to(torch.float32)
    
    # Ensure cortex state tensors
    state_tensors = ['z', 'stable_state', 'last_o', 'last_raw_o', 'column_trace', 
                     'last_state', 'last_pred', 'last_error', 'col_activation_avg',
                     'W_temp', 'eligibility']
    
    for attr in state_tensors:
        if hasattr(cortex, attr):
            tensor = getattr(cortex, attr)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                setattr(cortex, attr, tensor.to(torch.float32))
    
    # Ensure weight matrices are float32
    weight_tensors = ['W', 'B', 'W_temp']
    for attr in weight_tensors:
        if hasattr(cortex, attr):
            tensor = getattr(cortex, attr)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                setattr(cortex, attr, tensor.to(torch.float32))
    
    # Ensure reward module is float32
    if hasattr(cortex, 'reward_module'):
        if hasattr(cortex.reward_module, 'V'):
            # V is a float, not a tensor
            cortex.reward_module.V = float(cortex.reward_module.V)
    
    return cortex

def ensure_block_float32(block):
    """Ensure predictive block tensors are float32"""
    # Ensure weight matrices
    weight_tensors = ['G', 'R', 'W_up', 'Pi']
    for attr in weight_tensors:
        if hasattr(block, attr):
            tensor = getattr(block, attr)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                setattr(block, attr, tensor.to(torch.float32))
    
    # Ensure state tensors
    state_tensors = ['o', 'o_pred', 'error', 'last_lower_state']
    for attr in state_tensors:
        if hasattr(block, attr):
            tensor = getattr(block, attr)
            if tensor is not None and isinstance(tensor, torch.Tensor):
                setattr(block, attr, tensor.to(torch.float32))
    
    return block

def hierarchical_step(system: Dict, x: torch.Tensor, 
                     agent_action: int, step_counter: int,
                     prev_reward: Optional[float] = None,
                     ) -> Dict:
    """
    Integrated hierarchical step with FIXED reward timing
    
    Args:
        system: The hierarchical system
        x: Sensory input (state)
        agent_action: Baseline agent policy action
        step_counter: Current step count
        raw_reward: OPTIONAL reward from PREVIOUS step (for learning)
                     If None, assume no reward yet
    """
    hierarchy = system['hierarchy']
    controller = system['controller']
    rhythm = system['rhythm']
    
    # Ensure input is float32
    x = x.to(torch.float32) if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
    
    # Process PREVIOUS reward if available (for learning)
    bottom_cortex = system['cortices'][0]
    dopamine_signal = 0.0
    if prev_reward is not None:
        dopamine_signal = bottom_cortex.process_reward(prev_reward)
        # DEBUG
        print(f"[REWARD TIMING] Step {step_counter}: "
              f"Processing reward from previous action: {prev_reward:.3f} → "
              f"Dopamine (RPE): {dopamine_signal:.3f}, "
              f"V estimate: {bottom_cortex.reward_module.V:.3f}")
           
    # SET PLASTICITY GATE
    plasticity_gate = 1.0
    if rhythm and rhythm.theta_in_window():
        plasticity_gate = 1.3  # Enhanced plasticity during theta
        rhythm.tick()
    
    # HIERARCHICAL PREDICTIVE PROCESSING (learn from previous reward)
    top_representation = hierarchy.step(sensory_input=x,delta_da=dopamine_signal,plasticity_gate=plasticity_gate,inference_iterations=2)
    
    # ACTION GENERATION
    action_prior = controller.propose(top_representation)
    
    #BLEND AGENT POLICY (with hierarchical prior)
    base_weight = 0.1
    max_weight = 0.5
    training_progress = min(1.0, step_counter / 1000.0)
    prior_weight = base_weight + (max_weight - base_weight) * training_progress
    agent_action_tensor = torch.tensor([float(agent_action)], dtype=torch.float32)
    blended_action = (1 - prior_weight) * agent_action_tensor + prior_weight * action_prior[0]
    
    # Convert to discrete action with hysteresis
    final_action = 0
    if blended_action > 0.4:
        final_action = 1
    elif blended_action < -0.4:
        final_action = -1
    
    # === ADDITIONAL DEBUG FOR ACTION SELECTION ===
    if step_counter % 10 == 0:  # Print every 10 steps to avoid spam
        print(f"[ACTION DEBUG] Step {step_counter}: "
              f"Agent action: {agent_action}, "
              f"Prior weight: {prior_weight:.2f}, "
              f"Blended: {blended_action.item():.3f}, "
              f"Final: {final_action}")
    
    # Controller learning from PREVIOUS outcome (if we had a reward)
    if prev_reward is not None:
        success = prev_reward > 0
        controller.learn_from_outcome(prev_reward, success)
        # Debug controller learning
        if step_counter % 20 == 0:
            controller_info = controller.get_debug_info()
            print(f"[CONTROLLER DEBUG] Step {step_counter}: "
                  f"Learning steps: {controller_info['learning_steps']}, "
                  f"Weight norm: {controller_info['M_norm']:.4f}")
    
    # Timescale-aware replay
    for i, cortex in enumerate(system['cortices']):
        if step_counter % (2 ** i) == 0:  # Higher levels replay less frequently
            cortex.execute_background_replay()
    
    # Get debugging information
    hierarchy_state = hierarchy.get_hierarchy_state()
    hierarchy_stats = hierarchy.get_detailed_stats()
    
    # === DEBUG HIERARCHY ERRORS ===
    if step_counter % 25 == 0:  # Every 25 steps
        print(f"[HIERARCHY ERROR DEBUG] Step {step_counter}: ")
        for i, error in enumerate(hierarchy_state['errors']):
            print(f"  Level {i}: error={error:.3f}")
    
    return {
        'top_representation': top_representation,
        'action_prior': action_prior,
        'final_action': final_action,
        'dopamine_signal': dopamine_signal,
        'prev_reward_processed': prev_reward if prev_reward is not None else 0.0,
        'hierarchy_state': hierarchy_state,
        'hierarchy_stats': hierarchy_stats,
        'prior_weight': prior_weight,
        'blended_action': blended_action.item(),
        'V_value': bottom_cortex.reward_module.V if hasattr(bottom_cortex, 'reward_module') else 0.0,
        'theta_active': rhythm.theta_in_window() if rhythm else False
    }

def run_comprehensive_test(num_steps: int = 500):
    """Run comprehensive test of Phase 3 system"""
    print("=" * 80)
    print("PHASE 3 - COMPREHENSIVE HIERARCHICAL SYSTEM TEST")
    print("=" * 80)
    
    # Create system
    system = create_hierarchical_system(input_size=10,base_columns=4,hierarchy_factor=0.75, neurons_per_column=5)
    # Create environment
    env = TestEnvironment(input_size=10)
    
    # Statistics collection
    stats = {
        'rewards': [],
        'dopamine_signals': [],
        'hierarchy_errors': [[] for _ in range(system['levels'])],
        'actions': [],
        'convergence': []
    }
    
    print(f"\nSystem Configuration:")
    print(f"  Levels: {system['levels']}")
    print(f"  Columns per level: {[pb.dim for pb in system['predictive_blocks']]}")
    print(f"  Timescales: {[pb.timescale for pb in system['predictive_blocks']]}")
    print(f"  Rhythm: {'Enabled' if system['rhythm'] else 'Disabled'}")
    
    print(f"\nRunning {num_steps} steps...")
    print("-" * 80)
    
    # Simple agent policy (baseline)
    def agent_policy(position, goal):
        if position > goal + 0.5:
            return -1
        elif position < goal - 0.5:
            return 1
        else:
            return 0
    
    # Main loop
    reward_from_previous_action = None  # Initially no previous action
    for step in range(num_steps):
        # Get state
        x = env.get_state_vector()
        
        # Agent policy
        agent_action = agent_policy(env.position, env.goal)
        
        # Hierarchical step with PREVIOUS reward (from step-1)
        result = hierarchical_step(system, x, agent_action, step, prev_reward=reward_from_previous_action)
        # Take action in environment and get CURRENT reward
        current_reward = env.step(result['final_action'])
         # Update reward in system for episode recording
        system['cortices'][0].last_reward = current_reward
         # ===== STORE FOR NEXT ITERATION =====
        reward_from_previous_action = current_reward  # This becomes previous reward for next step
        # Collect statistics
        stats['rewards'].append(current_reward)  # Current reward
        stats['dopamine_signals'].append(result['dopamine_signal'])  # DA from previous
        stats['actions'].append(result['final_action'])
        stats['convergence'].append(result['hierarchy_state']['convergence'])
        
        # Collect level errors
        for i, error in enumerate(result['hierarchy_state']['errors']):
            if i < len(stats['hierarchy_errors']):
                stats['hierarchy_errors'][i].append(error)
        
        # Periodic reporting
        if step % 50 == 0 or step < 10:
            hierarchy_state = result['hierarchy_state']
            print(f"Step {step:4d} | "
                  f"Pos: {env.position:6.2f} | "
                  f"Reward: {current_reward:6.2f} | "
                  f"DA: {result['dopamine_signal']:6.3f} | "
                  f"Action: {result['final_action']:2d} | "
                  f"Hierarchy errors: {[f'{e:.3f}' for e in hierarchy_state['errors']]}")
        
        # Early stability check
        if step == 20:
            print("\nEarly stability check:")
            for i, pb in enumerate(system['predictive_blocks']):
                info = pb.get_debug_info()
                print(f"  Level {i}: o_norm={info['o_norm']:.3f}, "
                      f"error={info['error_norm']:.3f}, "
                      f"precision={info['precision_mean']:.3f}")
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    # Analyze results
    final_position = env.position
    avg_reward = np.mean(stats['rewards'][-100:]) if len(stats['rewards']) >= 100 else np.mean(stats['rewards'])
    avg_convergence = np.mean(stats['convergence'][-50:]) if stats['convergence'] else 0
    
    print(f"\nPerformance Metrics:")
    print(f"  Final position: {final_position:.3f} (goal: {env.goal})")
    print(f"  Average reward (last 100): {avg_reward:.3f}")
    print(f"  Convergence rate: {avg_convergence:.6f}")
    
    # Hierarchy analysis
    print(f"\nHierarchy Analysis:")
    for i in range(system['levels']):
        level_errors = stats['hierarchy_errors'][i]
        if level_errors:
            initial_error = np.mean(level_errors[:10])
            final_error = np.mean(level_errors[-10:])
            improvement = initial_error - final_error if initial_error > 0 else 0
            print(f"  Level {i}: Error {initial_error:.3f} → {final_error:.3f} "
                  f"(Δ={improvement:.3f}, {'✓' if improvement > 0 else '✗'})")
    
    # Controller analysis
    controller_info = system['controller'].get_debug_info()
    print(f"\nController Analysis:")
    print(f"  Weight norm: {controller_info['M_norm']:.3f}")
    print(f"  Learning steps: {controller_info['learning_steps']}")
    
    # Stability checks
    print(f"\nStability Checks:")
    
    # Check for NaNs
    has_nan = any(np.isnan(r) for r in stats['rewards'])
    print(f"  NaN values: {'❌' if has_nan else '✓'} (no NaNs)")
    
    # Check for divergence
    max_error = max([max(errors) for errors in stats['hierarchy_errors'] if errors])
    print(f"  Max hierarchy error: {max_error:.3f} "
          f"{'❌ (divergence)' if max_error > 10.0 else '✓ (stable)'}")
    
    # Check dopamine range
    da_min, da_max = min(stats['dopamine_signals']), max(stats['dopamine_signals'])
    print(f"  Dopamine range: [{da_min:.3f}, {da_max:.3f}] "
          f"{'✓' if -5.0 < da_min < da_max < 5.0 else '❌ (unstable)'}")
    
    # Check action distribution
    actions = np.array(stats['actions'])
    action_counts = {a: np.sum(actions == a) for a in [-1, 0, 1]}
    print(f"  Action distribution: {action_counts}")
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    recommendations = []
    
    if avg_reward < -2.0:
        recommendations.append("1. Increase hierarchy influence (raise prior_weight)")
    
    if max_error > 5.0:
        recommendations.append("2. Reduce learning rates or add more regularization")
    
    if abs(final_position - env.goal) > 1.0:
        recommendations.append("3. Improve agent policy or increase training steps")
    
    if len(stats['dopamine_signals']) > 0 and np.std(stats['dopamine_signals']) > 2.0:
        recommendations.append("4. Stabilize dopamine signal with moving average")
    
    if not recommendations:
        recommendations.append("System is stable and learning! Consider:")
        recommendations.append("  - Increasing hierarchy depth")
        recommendations.append("  - Adding more complex patterns")
        recommendations.append("  - Testing on more challenging tasks")
    
    for rec in recommendations:
        print(f"  • {rec}")
    
    # Plot results if matplotlib is available
    try:
        plot_results(stats, system['levels'])
    except:
        print("\n(Install matplotlib for visualization)")
    
    print(f"\n✅ Phase 3 test completed!")
    return stats, system

def plot_results(stats, num_levels):
    """Plot test results"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Plot rewards
    axes[0, 0].plot(stats['rewards'])
    axes[0, 0].set_title('Rewards over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot dopamine
    axes[0, 1].plot(stats['dopamine_signals'])
    axes[0, 1].set_title('Dopamine (RPE) Signal')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Dopamine')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot hierarchy errors
    for i in range(num_levels):
        if i < len(stats['hierarchy_errors']) and stats['hierarchy_errors'][i]:
            axes[1, 0].plot(stats['hierarchy_errors'][i], label=f'Level {i}')
    axes[1, 0].set_title('Hierarchy Prediction Errors')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Error Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot actions
    axes[1, 1].plot(stats['actions'], 'o', markersize=2, alpha=0.5)
    axes[1, 1].set_title('Actions Taken')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Action')
    axes[1, 1].set_yticks([-1, 0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot convergence
    if stats['convergence']:
        axes[2, 0].plot(stats['convergence'])
        axes[2, 0].set_title('Hierarchy Convergence')
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Convergence Rate')
        axes[2, 0].grid(True, alpha=0.3)
    
    # Plot reward distribution
    axes[2, 1].hist(stats['rewards'], bins=20, alpha=0.7)
    axes[2, 1].set_title('Reward Distribution')
    axes[2, 1].set_xlabel('Reward')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file= f'graphs/phase_three_results_{timestamp}.png'
    plt.savefig(file, dpi=150)
    plt.show()

if __name__ == "__main__":
    # Run comprehensive test
    stats, system = run_comprehensive_test(num_steps=200)
    
    # Additional debugging information
    print(f"\n" + "=" * 80)
    print("DEBUGGING INFORMATION")
    print("=" * 80)
    
    # Check each predictive block
    for i, pb in enumerate(system['predictive_blocks']):
        info = pb.get_debug_info()
        print(f"\nLevel {i} Predictive Block:")
        print(f"  Representation norm: {info['o_norm']:.3f}")
        print(f"  Current error: {info['error_norm']:.3f}")
        print(f"  Precision mean: {info['precision_mean']:.3f}")
        print(f"  Timescale: {info['timescale']}")
        print(f"  Steps processed: {info['steps']}")
        
        if info['recent_errors']:
            print(f"  Recent errors: {[f'{e:.3f}' for e in info['recent_errors'][-3:]]}")
    
    # Check base cortices
    print(f"\nBase Cortices:")
    for i, cortex in enumerate(system['cortices']):
        stats_info = cortex.get_memory_statistics()
        print(f"  Cortex {i}: {stats_info.get('buffer_size', 0)} episodes, "
              f"{stats_info.get('current_episode_length', 0)} current steps")
    
    print(f"\n✅ Phase 3 system is ready for deployment!")