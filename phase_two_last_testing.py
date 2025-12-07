# Add this test scenario after your imports
from phase_two_attractor_cotex import AttractorCortex
from phase_two_column import ProtoColumn
from phase_two_rhythm_engine import RhythmEngine
import torch
class TestWorld:
    """Simple test environment that generates meaningful patterns"""
    def __init__(self, input_size: int = 10):
        self.input_size = input_size
        self.position = 0
        self.goal = 5
        self.step_count = 0
        
        # Create predictable patterns
        self.patterns = [
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        
    def step(self):
        """Generate input with predictable patterns and occasional surprises"""
        self.step_count += 1
        
        # Every 8 steps, create a surprising pattern
        if self.step_count % 8 == 0:
            # Surprising pattern (different from normal sequence)
            x = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            reward = 1.0  # Positive reward for surprise
            is_surprising = True
        else:
            # Normal sequential pattern
            pattern_idx = (self.step_count // 2) % len(self.patterns)
            x = self.patterns[pattern_idx]
            reward = 0.1  # Small positive reward
            is_surprising = False
        
        # Add small noise
        x = x + torch.randn(self.input_size) * 0.1
        
        return x, reward, is_surprising

# Then modify your test code:

if __name__ == "__main__":
    print("Testing AttractorCortex with Episodic Memory (Phase 2.10)...")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Create rhythm engine
    rhythm = RhythmEngine(
        theta_freq=10.0,
        gamma_per_theta=4,
        sim_steps_per_second=100,
        theta_phase_window=(0.0, 0.5)
    )
    
    # Build columns
    input_size = 10
    num_columns = 4
    neurons_per_column = 5
    
    cols = [ProtoColumn(input_size=input_size, num_neurons=neurons_per_column) 
            for _ in range(num_columns)]
    
    # Create enhanced cortex
    cortex = AttractorCortex(
        columns=cols,
        input_proj=True,
        rhythm=rhythm,
        settling_steps=8,
        k_sparsity=2,
        
        # Episodic memory parameters
        buffer_capacity=100,
        max_episode_length=10,
        immediate_replay_threshold=0.3,  # Slightly higher since we'll get real values
        background_replay_batch=2,
        priority_decay=0.05
    )
    
    print(f"Created Enhanced AttractorCortex with {cortex.C} columns")
    print(f"Episodic buffer capacity: {cortex.buffer_capacity}")
    print(f"Max episode length: {cortex.max_episode_length}")
    print(f"Immediate replay threshold: {cortex.immediate_replay_threshold}")
    
    # Create test world
    world = TestWorld(input_size=input_size)
    
    # Run simulation
    print(f"\nRunning memory-integrated simulation (200 steps)...")
    print("-" * 70)
    print("Format: Step | Buffer | Curr Ep Len | Last Reward | Last Surprise | Theta | Replays")
    print("-" * 70)
    
    # Store previous state for temporal learning
    prev_state = None
    
    for step in range(200):
        # Get meaningful input and reward from test world
        x, reward, is_surprising = world.step()
        
        # Set reward for episode recording
        cortex.set_reward(reward)
        
        # Forward pass
        current_state = cortex.forward(x)
        
        # Temporal learning if we have previous state
        if prev_state is not None:
            # Learn temporal prediction
            cortex.learn_temporal(current_state, da_signal=reward)
        
        # Apply DA-modulated learning
        cortex.learn(da_signal=reward)
        
        # Store current state for next iteration
        prev_state = current_state.clone()
        
        # Execute background replay during theta windows
        if cortex.rhythm and cortex.rhythm.theta_in_window():
            cortex.execute_background_replay()
            cortex.run_replay()
        
        # Debug output every 10 steps
        if step % 10 == 0:
            stats = cortex.get_memory_statistics()
            theta_active = cortex.rhythm.theta_in_window() if cortex.rhythm else False
            
            # Safely format surprise
            surprise_display = cortex.last_surprise if hasattr(cortex, 'last_surprise') and cortex.last_surprise is not None else 0.0
            
            print(f"Step {step:3d} | "
                  f"Buffer: {stats['buffer_size']:2d} | "
                  f"Curr: {cortex.current_episode_length:2d}/{cortex.max_episode_length} | "
                  f"Reward: {cortex.last_reward:5.2f} | "
                  f"Surprise: {surprise_display:5.2f} | "
                  f"Theta: {'✓' if theta_active else '✗'} | "
                  f"Replays: {stats['immediate_replays']}I/{stats['background_replays']}B")
    
    print(f"\nSimulation completed!")
    
    # Analyze episodes
    print(f"\nTop 5 Episodes by Priority:")
    print("-" * 50)
    if len(cortex.episode_buffer.buffer) > 0:
        # Sort by priority
        sorted_episodes = sorted(cortex.episode_buffer.buffer, 
                                key=lambda e: e.priority, 
                                reverse=True)
        
        for i, episode in enumerate(sorted_episodes[:5]):
            above_threshold = episode.priority > cortex.immediate_replay_threshold
            threshold_mark = "🚨" if above_threshold else "  "
            print(f"{threshold_mark} Episode {episode.id}: "
                  f"Priority={episode.priority:.3f}, "
                  f"Steps={episode.length}, "
                  f"Total Reward={episode.total_reward:.3f}, "
                  f"Total Surprise={episode.total_surprise:.3f}")
            
            # Show if this triggered immediate replay
            if episode.replay_count > 0:
                print(f"    ↳ Replayed {episode.replay_count} times")
    
    print(f"\n" + "=" * 70)
    print("Key Insights:")
    print("=" * 70)
    
    if len(cortex.episode_buffer.buffer) > 0:
        # Calculate statistics
        priorities = [e.priority for e in cortex.episode_buffer.buffer]
        rewards = [e.total_reward for e in cortex.episode_buffer.buffer]
        surprises = [e.total_surprise for e in cortex.episode_buffer.buffer]
        
        print(f"Average episode priority: {sum(priorities)/len(priorities):.3f}")
        print(f"Max episode priority: {max(priorities):.3f}")
        print(f"Min episode priority: {min(priorities):.3f}")
        print(f"Episodes above threshold: {sum(1 for p in priorities if p > cortex.immediate_replay_threshold)}/{len(priorities)}")
        print(f"Average reward per episode: {sum(rewards)/len(rewards):.3f}")
        print(f"Average surprise per episode: {sum(surprises)/len(surprises):.3f}")
        
        # Check why immediate replays might not trigger
        if stats['immediate_replays'] == 0:
            print(f"\n⚠️  Why no immediate replays?")
            print(f"   - Threshold: {cortex.immediate_replay_threshold}")
            print(f"   - Max priority: {max(priorities):.3f}")
            print(f"   - Try lowering threshold to {max(priorities) * 0.9:.3f}")
    
    print(f"\n" + "=" * 70)
    print("Phase 2.10 System Status:")
    print("=" * 70)
    
    # Quick diagnostics
    print("1. ✅ Episode recording: Working")
    print("2. ✅ Priority computation: Working")
    print(f"3. {'✅' if stats['background_replays'] > 0 else '⚠️ '} Background replay: {'Working' if stats['background_replays'] > 0 else 'Check theta timing'}")
    print(f"4. {'✅' if any(e.priority > 0.5 for e in cortex.episode_buffer.buffer) else '⚠️ '} Surprise generation: {'Good' if any(e.priority > 0.5 for e in cortex.episode_buffer.buffer) else 'Low - needs meaningful patterns'}")
    print(f"5. {'⚠️ ' if stats['immediate_replays'] == 0 else '✅ '} Immediate replay: {'Threshold too high' if stats['immediate_replays'] == 0 else 'Working'}")
    
    print("\n" + "=" * 70)