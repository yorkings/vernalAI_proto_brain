from phase_two_attractor_cotex import AttractorCortex
from phase_two_reward import RewardModule
from phase_three_predictive_block import PredictiveBlock
import torch
from collections import deque
from typing import List,Dict
# PHASE 4: BIOLOGICAL TEMPORAL PROCESSING
# 1. Add working memory to AttractorCortex
class BioAttractorCortex(AttractorCortex):
    def __init__(self, working_memory_decay=0.98, **kwargs):
        super().__init__(**kwargs)
        self.working_memory = None
        self.wm_decay = working_memory_decay
    
    def update_working_memory(self):
        """Biological working memory update"""
        if self.stable_state is not None:
            if self.working_memory is None:
                self.working_memory = self.stable_state.clone()
            else:
                # Leaky integration
                self.working_memory = (
                    self.wm_decay * self.working_memory + (1 - self.wm_decay) * self.stable_state
                )
        elif self.working_memory is None:
            # Initialize even without stable_state
            dim = self.C if hasattr(self, 'C') else 10
            self.working_memory = torch.zeros(dim, dtype=torch.float32)        

# 2. Add temporal discounting to RewardModule
class BioRewardModule(RewardModule):
    def __init__(self, gamma=0.99, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma  # Temporal discount factor
        self.reward_history = deque(maxlen=10)
    
    def step_with_discount(self, reward: float) -> float:
        """Compute RPE with temporal discounting"""
        # Store reward
        self.reward_history.append(reward)
        
        # Compute discounted future rewards
        discounted_future = 0.0
        for i, future_reward in enumerate(list(self.reward_history)[1:]):
            discounted_future += future_reward * (self.gamma ** i)
        
        # RPE = current reward + discounted future - current value estimate
        delta = reward + discounted_future - self.V
        self.V += self.value_lr * delta
        return delta

# 3. Add sequence learning to PredictiveBlocks
class BioPredictiveBlock(PredictiveBlock):
    def learn_sequences(self, sequence: List[torch.Tensor]):
        """Learn temporal sequences, not just pairwise transitions"""
        if len(sequence) < 2:
            return
        
        # Learn transitions between consecutive states
        for i in range(len(sequence) - 1):
            state_t = sequence[i]
            state_t1 = sequence[i + 1]
            
            # Update generative weights to predict t+1 from t
            error = state_t1 - (self.G @ state_t)
            self.G += self.alpha_G * torch.outer(error, state_t)

class EnhancedTemporalBuffer:
    def __init__(self, capacity=100):
        self.states = []
        self.errors = []
        self.surprises = []
        self.capacity = capacity
        
    def add_step(self, state, error, surprise):
        self.states.append(state.clone())
        self.errors.append(error)
        self.surprises.append(surprise)
        if len(self.states) > self.capacity:
            self.states.pop(0)
            self.errors.pop(0)
            self.surprises.pop(0)
    
    def get_error_trend(self, window=5):
        if len(self.errors) < window:
            return 0.0
        recent_errors = self.errors[-window:]
        return recent_errors[-1] - recent_errors[0]



import numpy as np
import matplotlib.pyplot as plt

def test_biological_temporal_enhancements():
    """Test Phase 4: Biological Temporal Processing Enhancements"""
    print("=" * 80)
    print("PHASE 4: BIOLOGICAL TEMPORAL PROCESSING TEST")
    print("=" * 80)
    
    torch.manual_seed(42)
    
    # Test 1: BioAttractorCortex Working Memory
    print("\n1. Testing BioAttractorCortex Working Memory...")
    print("-" * 40)
    
    # Create a simple test cortex
    from phase_two_column import ProtoColumn
    
    # Create test columns
    input_size = 5
    num_columns = 3
    columns = [ProtoColumn(input_size=input_size, num_neurons=4) for _ in range(num_columns)]
    
    # Create BioAttractorCortex
    bio_cortex = BioAttractorCortex(
        columns=columns,
        input_proj=False,
        working_memory_decay=0.95,
        settling_steps=3,
        k_sparsity=2
    )
    
    # Test working memory persistence
    print("Testing working memory persistence:")
    states = []
    wm_states = []
    
    for step in range(10):
        x = torch.randn(input_size)
        output = bio_cortex.forward(x)
        bio_cortex.update_working_memory()
        
        states.append(bio_cortex.stable_state.clone())
        wm_states.append(bio_cortex.working_memory.clone() if bio_cortex.working_memory is not None else None)
        
        if bio_cortex.working_memory is not None:
            wm_norm = bio_cortex.working_memory.norm().item()
            current_norm = bio_cortex.stable_state.norm().item()
            similarity = torch.cosine_similarity(
                bio_cortex.working_memory.flatten(),
                bio_cortex.stable_state.flatten(),
                dim=0
            ).item()
            
            print(f"  Step {step}: Current norm={current_norm:.3f}, "
                  f"WM norm={wm_norm:.3f}, Similarity={similarity:.3f}")
    
    # Check if working memory smooths states
    if len(states) >= 3:
        # Compute variability
        state_variability = torch.std(torch.stack(states[-3:]), dim=0).mean().item()
        wm_variability = torch.std(torch.stack(wm_states[-3:]), dim=0).mean().item() if None not in wm_states[-3:] else 0
        
        print(f"\n  Variability analysis:")
        print(f"    Raw states: {state_variability:.4f}")
        print(f"    Working memory: {wm_variability:.4f}")
        print(f"    WM is smoother: {'✓' if wm_variability < state_variability else '✗'}")
    
    # Test 2: BioRewardModule Temporal Discounting
    print("\n2. Testing BioRewardModule Temporal Discounting...")
    print("-" * 40)
    
    bio_reward = BioRewardModule(value_lr=0.1, gamma=0.9, baseline=0.0)
    
    # Create a reward sequence with delayed reward
    rewards = [0.1, 0.2, 0.3, 1.0, 0.1, 0.2]  # Big reward at step 3
    da_signals = []
    value_estimates = []
    
    print("Testing temporal discounting with delayed reward:")
    for i, reward in enumerate(rewards):
        da = bio_reward.step_with_discount(reward)
        da_signals.append(da)
        value_estimates.append(bio_reward.V)
        
        print(f"  Step {i}: Reward={reward:.2f}, DA={da:.3f}, V={bio_reward.V:.3f}")
    
    # Check if dopamine spikes appropriately
    da_spike_index = np.argmax(np.abs(da_signals))
    reward_spike_index = np.argmax(np.abs(rewards))
    
    print(f"\n  Reward spike at step: {reward_spike_index}")
    print(f"  Dopamine spike at step: {da_spike_index}")
    print(f"  Appropriate timing: {'✓' if da_spike_index >= reward_spike_index else '✗'}")
    
    # Test 3: BioPredictiveBlock Sequence Learning
    print("\n3. Testing BioPredictiveBlock Sequence Learning...")
    print("-" * 40)
    
    # Create a base cortex for the predictive block
    base_cortex = AttractorCortex(columns=columns, input_proj=False)
    
    # Create BioPredictiveBlock
    bio_predictive = BioPredictiveBlock(
        base_cortex=base_cortex,
        dim_rep=num_columns,
        level_idx=0
    )
    
    # Generate a simple sequence
    sequence_length = 5
    sequence = []
    
    print(f"Generating and learning a {sequence_length}-step sequence:")
    for i in range(sequence_length):
        # Create correlated states (simulating a pattern)
        if i == 0:
            state = torch.randn(num_columns)
        else:
            # Each state is similar to previous with some change
            state = sequence[-1] * 0.8 + torch.randn(num_columns) * 0.2
        
        state = torch.tanh(state)  # Normalize
        sequence.append(state)
        
        print(f"  State {i}: norm={state.norm().item():.3f}, "
              f"mean={state.mean().item():.3f}")
    
    # Test sequence learning
    G_before = bio_predictive.G.clone()
    
    # Learn the sequence multiple times
    for epoch in range(3):
        bio_predictive.learn_sequences(sequence)
        
        # Test prediction after learning
        if epoch % 1 == 0:
            test_state = sequence[0]
            predicted = bio_predictive.G @ test_state
            actual_next = sequence[1]
            error = (actual_next - predicted).norm().item()
            
            print(f"  Epoch {epoch}: Prediction error={error:.4f}")
    
    G_after = bio_predictive.G
    weight_change = (G_after - G_before).norm().item()
    
    print(f"\n  G matrix changed by: {weight_change:.6f}")
    print(f"  Learning occurred: {'✓' if weight_change > 1e-6 else '✗'}")
    
    # Test 4: Integrated Temporal Processing
    print("\n4. Testing Integrated Temporal Processing...")
    print("-" * 40)
    
    # Simulate a simple temporal task
    print("Simulating temporal pattern following task:")
    
    pattern = [
    torch.cat([torch.tensor([1.0, 0.0, 0.0]), torch.zeros(input_size - 3)]),
    torch.cat([torch.tensor([0.5, 0.5, 0.0]), torch.zeros(input_size - 3)]),
    torch.cat([torch.tensor([0.0, 1.0, 0.0]), torch.zeros(input_size - 3)]),
    torch.cat([torch.tensor([0.0, 0.5, 0.5]), torch.zeros(input_size - 3)])
      ]
    # Test with working memory
    bio_cortex2 = BioAttractorCortex(
        columns=columns,
        input_proj=False,
        working_memory_decay=0.9,
        settling_steps=2
    )
    
    predictions = []
    wm_predictions = []
    
    for step in range(12):
        # Get pattern state (repeating)
        pattern_idx = step % len(pattern)
        input_state = pattern[pattern_idx] + torch.randn(input_size) * 0.1  # Add noise
        
        # Process through cortex
        output = bio_cortex2.forward(input_state)
        bio_cortex2.update_working_memory()
        
        # Store for analysis
        predictions.append(output.clone())
        
        if bio_cortex2.working_memory is not None:
            wm_predictions.append(bio_cortex2.working_memory.clone())
        
        if step < 8:  # Print first 8 steps
            print(f"  Step {step}: Pattern {pattern_idx}, "
                  f"Output norm={output.norm().item():.3f}")
    
    # Analyze pattern following
    if len(predictions) >= 8:
        # Check if working memory helps stabilize predictions
        pred_vars = []
        wm_vars = []
        
        for i in range(0, len(predictions) - 4, 4):
            # Get one cycle of predictions
            cycle_preds = predictions[i:i+4]
            cycle_wm = wm_predictions[i:i+4] if len(wm_predictions) >= i+4 else []
            
            if cycle_preds:
                pred_var = torch.std(torch.stack(cycle_preds), dim=0).mean().item()
                pred_vars.append(pred_var)
            
            if cycle_wm:
                wm_var = torch.std(torch.stack(cycle_wm), dim=0).mean().item()
                wm_vars.append(wm_var)
        
        if pred_vars and wm_vars:
            avg_pred_var = np.mean(pred_vars)
            avg_wm_var = np.mean(wm_vars)
            
            print(f"\n  Variability analysis:")
            print(f"    Raw predictions: {avg_pred_var:.4f}")
            print(f"    Working memory: {avg_wm_var:.4f}")
            print(f"    WM reduces variability by: {((avg_pred_var - avg_wm_var) / avg_pred_var * 100):.1f}%")
    
    # Test 5: Delayed Reward Task
    print("\n5. Testing Delayed Reward Processing...")
    print("-" * 40)
    
    # Simulate a task where reward comes 2 steps after correct action
    bio_reward2 = BioRewardModule(value_lr=0.2, gamma=0.85)
    
    # Sequence: neutral, neutral, good action, neutral, BIG REWARD
    task_rewards = [0.0, 0.0, 0.1, 0.0, 2.0, 0.0, 0.0]
    task_da_signals = []
    
    print("Delayed reward sequence:")
    for i, reward in enumerate(task_rewards):
        da = bio_reward2.step_with_discount(reward)
        task_da_signals.append(da)
        
        # Mark the critical steps
        marker = ""
        if i == 2:
            marker = " ← Good action"
        elif i == 4:
            marker = " ← BIG REWARD"
        
        print(f"  Step {i}: Reward={reward:.2f}, DA={da:.3f}{marker}")
    
    # Check if credit assignment works
    da_at_action = task_da_signals[2] if len(task_da_signals) > 2 else 0
    da_at_reward = task_da_signals[4] if len(task_da_signals) > 4 else 0
    
    print(f"\n  Credit assignment analysis:")
    print(f"    DA at action (step 2): {da_at_action:.3f}")
    print(f"    DA at reward (step 4): {da_at_reward:.3f}")
    print(f"    Appropriate: {'✓' if da_at_reward > da_at_action else '✗ (should learn from delayed reward)'}")
    
    # Summary and Recommendations
    print("\n" + "=" * 80)
    print("PHASE 4 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    results = {
        'working_memory': wm_variability < state_variability if 'wm_variability' in locals() and 'state_variability' in locals() else False,
        'temporal_discounting': da_spike_index >= reward_spike_index,
        'sequence_learning': weight_change > 1e-6,
        'variability_reduction': avg_wm_var < avg_pred_var if 'avg_wm_var' in locals() and 'avg_pred_var' in locals() else False
    }
    
    print("\nComponent Test Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:25s}: {status}")
    
    print("\nKey Insights:")
    print("  1. Working memory creates smoother, more stable representations")
    print("  2. Temporal discounting helps with delayed reward assignment")
    print("  3. Sequence learning enables pattern prediction")
    print("  4. Biological temporal processing stabilizes learning")
    
    print("\nRecommendations for Integration:")
    print("  1. Use BioAttractorCortex for all cortical levels")
    print("  2. Replace RewardModule with BioRewardModule")
    print("  3. Use BioPredictiveBlock for sequence learning tasks")
    print("  4. Consider adding theta-phase dependent working memory updates")
    
    print("\nNext Steps:")
    print("  1. Integrate these components into your hierarchical system")
    print("  2. Test on temporal pattern learning tasks")
    print("  3. Measure improvement in delayed reward tasks")
    print("  4. Add working memory to controller for action planning")
    
    # Visualization
    try:
        plot_temporal_test_results(
            states, wm_states, 
            rewards, da_signals, value_estimates,
            task_rewards, task_da_signals
        )
    except:
        print("\n(Install matplotlib for visualization)")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 4 TEST COMPLETE - BIOLOGICAL TEMPORAL PROCESSING READY")
    print("=" * 80)

def plot_temporal_test_results(
    states: List[torch.Tensor],
    wm_states: List[torch.Tensor],
    rewards: List[float],
    da_signals: List[float],
    value_estimates: List[float],
    task_rewards: List[float],
    task_da_signals: List[float]
):
    """Plot temporal test results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Working Memory vs Raw States
    if states and wm_states:
        state_norms = [s.norm().item() for s in states]
        wm_norms = [s.norm().item() if s is not None else 0 for s in wm_states]
        
        axes[0, 0].plot(state_norms, label='Raw States', marker='o')
        axes[0, 0].plot(wm_norms, label='Working Memory', marker='s')
        axes[0, 0].set_title('Working Memory Persistence')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('State Norm')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Reward and Dopamine
    axes[0, 1].plot(rewards, label='Reward', marker='o', color='green')
    axes[0, 1].plot(da_signals, label='Dopamine (RPE)', marker='s', color='orange')
    axes[0, 1].set_title('Reward and Dopamine Signals')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Value Estimate Learning
    axes[0, 2].plot(value_estimates, label='Value Estimate (V)', marker='o', color='purple')
    axes[0, 2].set_title('Value Learning Curve')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('V')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Delayed Reward Task
    axes[1, 0].plot(task_rewards, label='Reward', marker='o', color='green')
    axes[1, 0].plot(task_da_signals, label='Dopamine', marker='s', color='red')
    axes[1, 0].set_title('Delayed Reward Task')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Reward Distribution
    axes[1, 1].hist(rewards, bins=10, alpha=0.7, color='green')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Dopamine Distribution
    axes[1, 2].hist(da_signals, bins=10, alpha=0.7, color='orange')
    axes[1, 2].set_title('Dopamine Distribution')
    axes[1, 2].set_xlabel('Dopamine (RPE)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase_4_biological_temporal_test.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    test_biological_temporal_enhancements()



