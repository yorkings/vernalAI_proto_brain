import torch
from phase_two_column import ProtoColumn
from phase_two_cortex import CorticalBlock  # Import the CorticalBlock
from phase_two_rhythm_engine import RhythmEngine
from typing import List,Callable,Optional,Dict
from phase_two_last import EpisodeBuffer,EpisodeStep,Episode
from phase_two_reward import RewardModule

import math

class AttractorCortex(CorticalBlock):  # Extend CorticalBlock
    def __init__(self, columns: List[ProtoColumn],input_proj: bool = False,rho: float = 0.3,eta: float = 0.01, weight_decay: float = 1e-4,init_scale: float = 0.02, 
                 reduce_fn: Callable = lambda v: v.mean(), settling_steps: int = 5, energy_weight: float = 0.01, convergence_thresh: float = 1e-4,
                 activation_target=0.2,homeo_lr = 0.01,synaptic_scaling_target = 1.0,
                 k_sparsity = 2,inhibition_strength = 1.0,beta_soft = 20,use_soft_k = False,
                 rhythm:Optional[RhythmEngine]=None,
                 #episodic memmory
                 buffer_capacity: int = 200, max_episode_length: int = 10,immediate_replay_threshold: float = 0.2, background_replay_batch: int = 2,priority_decay: float = 0.05,
                  value_lr: float = 0.01, baseline_reward: float = 0.0
                  ):
        # Initialize parent CorticalBlock
        super().__init__(columns, input_proj, rho, eta, weight_decay,init_scale, reduce_fn)
        
        # Attractor-specific parameters
        self.settling_steps = settling_steps  # Max settling iterations
        self.energy_weight = energy_weight    # Lambda for energy computation
        self.convergence_thresh = convergence_thresh


        self.col_activation_avg = torch.zeros(self.C)

        #homeostasis params
        self.homeo_lr=homeo_lr
        self.synaptic_scaling_target=synaptic_scaling_target
        self.activation_target=activation_target

        #sparsity
        self.k_sparsity=k_sparsity
        self.inhibition_strength=inhibition_strength
        self.beta_soft=beta_soft
        self.use_soft_k=use_soft_k

        # State tracking for attractor dynamics
        self.stable_state = None  # Last converged state
        self.energy_value = None  # Last computed energy
        self.converged = False    # Whether last forward converged
        self.settling_history = []  # Track convergence per step
        
        self.W_temp = torch.randn(self.C, self.C) * 0.05
        self.eta_temp = 0.01
        self.temp_weight_decay = 1e-4

        self.last_state = None
        self.last_pred = None
        self.last_error = None
        self.last_surprise = 0.0
        #rhythm
        self.rhythm = rhythm
        if rhythm is not None:
            for col in self.columns:
                col.preferred_gamma_bin= torch.randint(low=0,high=rhythm.gamma_per_theta, size=(1,)).item()
        # ========== REWARD MODULE (Phase 2.4) ==========
        self.reward_module = RewardModule(value_lr=value_lr, baseline=baseline_reward)
        self.last_raw_reward = 0.0  # Store the raw reward before RPE calculation
        self.last_dopamine_signal = 0.0  # Store the dopamine (RPE) signal        
         # ========== EPISODIC MEMORY ADDITIONS (Phase 2.10) ==========
        self.buffer_capacity = buffer_capacity
        self.max_episode_length = max_episode_length
        self.immediate_replay_threshold = immediate_replay_threshold
        self.background_replay_batch = background_replay_batch
        self.priority_decay = priority_decay
        
        # Episode buffer
        self.episode_buffer = EpisodeBuffer(capacity=buffer_capacity)
        
        # Current episode being recorded
        self.current_episode: Optional[Episode] = None
        self.current_episode_length = 0
        self.episode_start_time = 0
        
        # Global timestep counter
        self.global_timestep = 0
        
        # Replay mode flag
        self.replay_mode = False
        
        # Statistics
        self.memory_stats = {
            'episodes_recorded': 0,
            'steps_recorded': 0,
            'immediate_replays': 0,
            'background_replays': 0,
            'total_replay_steps': 0
        }
                
        # Replay memory
        self.replay_memory = []
        self.max_replay_length = 50   
        # For tracking reward/surprise
        self.last_reward = 0.0
        self.last_prediction_error = 0.0  

    def process_reward(self, raw_reward: float) -> float:
        """
        Process reward through RewardModule to get dopamine signal (RPE).
        
        Args:
            raw_reward: The observed reward from environment
            
        Returns:
            dopamine_signal: Reward Prediction Error (δ = r - V)
        """
        self.last_raw_reward = raw_reward
        self.last_dopamine_signal = self.reward_module.step(raw_reward)
        return self.last_dopamine_signal       

    def predict_next(self, o: torch.Tensor) -> torch.Tensor:
        """Predict next state: ô_{t+1} = normalize(W_temp @ o_t)."""
        pred = self.W_temp @ o
        pred = pred / (1 + torch.abs(pred))
        pred = self.apply_sparsity(pred)
        return pred       
            
    def compute_energy(self, o: torch.Tensor) -> float:
        """Compute energy: E(o) = -0.5*(o^T @ W @ o) + energy_weight*sum(o^2)."""
        recurrent_energy = -0.5 * torch.dot(o, self.W @ o)
        regularization = self.energy_weight * torch.sum(o**2)
        return (recurrent_energy + regularization).item()
    
    def forward(self, x: torch.Tensor = None,replay_mode: bool = False) -> torch.Tensor:
        """Forward pass with attractor settling, temporal prediction, AND episode recording"""
         # Set replay mode
        self.replay_mode = replay_mode

        # 1. Update columns if x provided
        if x is not None:
            for col in self.columns:
                col.forward(x)
        # 2. Get initial column summaries
        o = torch.zeros(self.C)
        for i, col in enumerate(self.columns):
            if col.last_output is None:
                # Need to compute column output first
                if x is None:
                    raise ValueError("Columns need input x or pre-computed outputs")
                col.forward(x)
            o[i] = float(self.reduce_fn(col.last_output)) 
        # Initialize recurrent state
        z = self.z.clone()
        prev_o = o.clone()        
        # 3. Settling loop
        self.converged = False
        self.settling_history = []
        
        for step in range(self.settling_steps):
            if self.rhythm:
                self.rhythm.tick()
            # Compute recurrent input
            if self.use_input_proj and x is not None:
                input_proj = self.B.matmul(x)
            else:
                input_proj = torch.zeros(self.C)            
            # Update recurrent state
            z = (1.0 - self.rho) * z + self.rho * (self.W @ o + input_proj)            
            # Normalize
            new_o = z / (1 + torch.abs(z))
            new_o=self.apply_sparsity(new_o)  
            # Apply temporal gating for eligibility updates
            if self.rhythm:
                for i, col in enumerate(self.columns):
                    _, gamma_gate, _ = self.rhythm.get_gates_for_column(col.preferred_gamma_bin)
                    if gamma_gate and hasattr(col.layer, 'neurons'):
                        # Update eligibility only when gamma gate allows
                        for neuron in col.layer.neurons:
                            if hasattr(neuron, 'eligibility') and hasattr(neuron, 'last_activation'):
                                # Simple eligibility update
                                neuron.eligibility = (0.9 * neuron.eligibility + neuron.last_activation * neuron.last_input)
          
            # Check convergence
            diff = torch.norm(new_o - prev_o).item()
            self.settling_history.append(diff)
            
            if diff < self.convergence_thresh:
                self.converged = True
                o = new_o
                break            
            prev_o = o
            o = new_o        
        # 4. Save final state
        self.z = z
        self.stable_state = o.clone()
        self.energy_value = self.compute_energy(o)
        self.last_o = o.clone()
        self.last_raw_o = torch.tensor([self.reduce_fn(col.last_output) for col in self.columns])
        # Temporal prediction for next timestep
        self.last_pred = self.predict_next(o)
        self.last_state = o.clone()

        if not self.replay_mode and x is not None:
            self._record_step(x, o)

        return o
    
    def _record_step(self, x: torch.Tensor, attractor_state: torch.Tensor):
        """Record current step in episode"""
        print(f"[DEBUG] Storing reward: last_reward={self.last_reward}, last_raw_reward={self.last_raw_reward}")
        # Start new episode if needed
        if self.current_episode is None:
            self.current_episode = Episode(start_timestamp=self.global_timestep)
            self.current_episode_length = 0
            self.episode_start_time = self.global_timestep     
        # Get column states
        column_states = torch.cat([col.last_output for col in self.columns])
        # Create step - ensure last_surprise is not None
        current_surprise = self.last_surprise if self.last_surprise is not None else 0.0
        current_pred_error = self.last_prediction_error if self.last_prediction_error is not None else 0.0
        # Create step
        step = EpisodeStep(timestamp=self.global_timestep,input_vector=x,column_states=column_states,attractor_state=attractor_state,reward=self.last_raw_reward,
                           surprise=self.last_surprise,pred_error=self.last_prediction_error)
        # Add to current episode
        self.current_episode.add_step(step)
        self.current_episode_length += 1
        self.global_timestep += 1 
        self.memory_stats['steps_recorded'] += 1
        # Check if episode should end
        should_end = (self.current_episode_length >= self.max_episode_length or   abs(self.last_reward) > 2.0 or  # Large reward event
            (self.last_surprise is not None and self.last_surprise > 0.5)  # High surprise
        )
        if should_end:
            self._finish_current_episode()
            
    def _finish_current_episode(self):
        """Finish and store current episode"""
        if self.current_episode is None or self.current_episode_length == 0:
            return    
        # Finalize episode
        self.current_episode.finalize(current_time=self.global_timestep) 
        # Add to buffer
        needs_immediate = self.episode_buffer.add_episode(self.current_episode)
        # Immediate replay if needed
        if needs_immediate:
            self._execute_immediate_replay(self.current_episode)      
        # Update statistics
        self.memory_stats['episodes_recorded'] += 1
        # Reset for next episode
        self.current_episode = None
        self.current_episode_length = 0
        
    def _execute_immediate_replay(self, episode: Episode):
        """Execute immediate replay of high-priority episode"""
        replay_count = self.episode_buffer.compute_replay_count(episode)
        # Execute reverse replay
        for _ in range(replay_count):
            steps = episode.get_steps_reversed()
            for step in steps:
                # Add small noise for generalization
                noisy_input = step.input_vector + torch.randn_like(step.input_vector) * 0.01  
                # Replay forward pass
                self.forward(x=noisy_input, replay_mode=True)
            # Learn from replay
            da_signal = episode.get_aggregate_dopamine()
            self.learn(da_signal=da_signal * 0.5)  # Weaker learning for replay
            self.memory_stats['total_replay_steps'] += len(steps)    
        # Decay priority
        episode.decay_priority(mu=self.priority_decay)
        self.memory_stats['immediate_replays'] += 1
        
    def execute_background_replay(self):
        """Execute background replay during theta windows"""
        if (self.replay_mode or len(self.episode_buffer.buffer) == 0 or (self.rhythm and not self.rhythm.theta_in_window())):
            return
        # Sample episodes for background replay
        episodes = self.episode_buffer.sample(k=self.background_replay_batch)
        for episode in episodes:
            replay_count = self.episode_buffer.compute_replay_count(episode)
            # Execute replay
            for _ in range(replay_count):
                steps = episode.get_steps_reversed()
                for step in steps:
                    # Add smaller noise for background replay
                    noisy_input = step.input_vector + torch.randn_like(step.input_vector) * 0.005
                    # Replay forward pass
                    self.forward(x=noisy_input, replay_mode=True)
                # Learn from replay with weak dopamine
                da_signal = episode.get_aggregate_dopamine()
                self.learn(da_signal=da_signal * 0.3)
                self.memory_stats['total_replay_steps'] += len(steps)   
            # Decay priority
            episode.decay_priority(mu=self.priority_decay * 0.5)  # Slower decay for background
            self.memory_stats['background_replays'] += 1
        
    #the three fuctions are for providing stability
    def homeostasis_update(self, o):
        """o = stable column activations (C-dim)"""
        # Update exponential moving average
        self.col_activation_avg = (0.99 * self.col_activation_avg + self.homeo_lr * o)
        # deviation from desired firing level
        delta = self.activation_target - self.col_activation_avg
        # shift recurrent weights
        self.W += self.homeo_lr * delta.unsqueeze(1)

    def synaptic_scaling(self):
        """Scale each row of W to maintain constant norm."""
        norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        target = self.synaptic_scaling_target
        self.W *= (target / norms)

    def normalize_recurrent_weights(self):
        """ Keep network stable."""
        # row normalize (keeps dynamics bounded)
        norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        self.W = self.W / norms

    def apply_sparsity(self, o: torch.Tensor) -> torch.Tensor:
        """k-Winner-Take-All (hard or soft) for sparse coding."""
        if self.use_soft_k:
            # Soft top-k via sharpened softmax
            o_ = torch.exp(self.beta_soft * o)
            return o_ / (o_.sum() + 1e-8)
        # Hard k-WTA
        C = o.shape[0]
        # If k >= C, do nothing
        if self.k_sparsity >= C:
            return o
        # Get threshold
        topk_vals, _ = torch.topk(o, self.k_sparsity)
        T = topk_vals[-1]
        # Apply competition mask
        mask = (o >= T).float()
        return o * mask        

    def learn(self,da_signal: float = 1.0) -> None:
        """
        Enhanced learn method that tracks surprise for episode recording
        Learn using Oja's rule with energy regularization.   
        Update rule: ΔW = η * (o o^T - energy_weight * W)
        """
        # Store prediction error as surprise metric
        if self.last_error is not None:
            self.last_surprise = torch.norm(self.last_error, p=1).item()
            self.last_prediction_error = self.last_surprise

        if self.rhythm and not self.rhythm.theta_in_window():
            return 
        if self.stable_state is None:
            raise ValueError("No stable state to learn from. Call forward() first.")
        
        o = self.stable_state
        
        # Outer product for Hebbian term
        outer = o.unsqueeze(1) * o.unsqueeze(0)  # C x C
        
        # Oja update with energy regularization
        deltaW = self.eta *da_signal* (outer - self.energy_weight * self.W)
        
        # Update weights
        self.W += deltaW
        
        # Normalize rows for stability
        row_norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-8
        self.W = self.W / row_norms
        #stabilize the weights and gradients
        o = self.stable_state
        self.homeostasis_update(o)
        self.synaptic_scaling()
        self.normalize_recurrent_weights()

  
    def get_memory_statistics(self) -> Dict:
        """Get episodic memory statistics"""
        return {
            **self.memory_stats,
            'buffer_size': len(self.episode_buffer.buffer),
            'current_episode_length': self.current_episode_length,
            'global_timestep': self.global_timestep
        }
        
    def reset_memory(self):
        """Reset episodic memory (keep buffer)"""
        self.current_episode = None
        self.current_episode_length = 0
        self.global_timestep = 0
        self.replay_mode = False
        self.last_reward = 0.0
        self.last_prediction_error = 0.0   
        self.reward_module.reset()  

    def learn_temporal(self, next_state: torch.Tensor,da_signal: float = 1.0):
        """Learn temporal transitions: ΔW_temp = η_temp*(o_t @ o_{t+1}^T - λ*W_temp)."""
        if self.last_state is None:
            return
        # Make next_state sparse
        next_state = self.apply_sparsity(next_state)
        # Compute prediction error
        error = next_state - self.last_pred
        self.last_error = error
        self.last_surprise = torch.norm(error, p=1).item() 
        # Hebbian temporal learning
        outer = torch.outer(self.last_state, next_state)
        dW = self.eta_temp *da_signal* (outer - self.temp_weight_decay * self.W_temp)
        self.W_temp += dW

    def _store_replay(self, x: torch.Tensor, state: torch.Tensor):
        self.replay_memory.append({
            'x': x.clone(),
            'state': state.clone(),
            't': self.rhythm.t if self.rhythm else 0
        })
        
        if len(self.replay_memory) > self.max_replay_length:
            self.replay_memory.pop(0)
    
    def run_replay(self):
        if not self.replay_memory or not self.rhythm:
            return
        
        # Only replay during theta windows
        if not self.rhythm.theta_in_window():
            return
        
        if len(self.replay_memory) > 0:
            import random
            memory = random.choice(self.replay_memory)
            
            # Replay the stored state
            replayed_state = self.forward(memory['x'])
            
            # Optionally learn from replay with weaker DA
            if torch.rand(1).item() < 0.3:
                self.learn(da_signal=0.5)    

    def get_convergence_info(self) -> dict:
        """Get information about last settling process."""
        if not self.settling_history:
            return {"converged": False, "steps": 0, "final_diff": None}
        
        return {
            "converged": self.converged,
            "steps": len(self.settling_history),
            "final_diff": self.settling_history[-1],
            "energy": self.energy_value
        }






def cortex_step(cortex: AttractorCortex, x: torch.Tensor, raw_reward: float = 0.0) -> torch.Tensor:
    """
    One full timestep of the AI.
    Includes:
      - rhythm
      - attractor settling
      - eligibility traces
      - DA plasticity
      - replay hooks
    """
    # Process reward through RewardModule to get dopamine signal (RPE)
    dopamine_signal = cortex.process_reward(raw_reward)
    # ALSO set last_reward for backward compatibility
    cortex.last_reward = raw_reward
    # forward pass with attractor dynamics
    o = cortex.forward(x)

    # apply DA-modulated learning
    cortex.learn(da_signal=dopamine_signal)

    # optional replay
    if cortex.rhythm and cortex.rhythm.theta_in_window():
        cortex.execute_background_replay()
        cortex.run_replay()

    return o


# Fix the test section - REPLACE THE ENTIRE TEST CODE:

if __name__ == "__main__":
    print("Testing AttractorCortex with RewardModule & Episodic Memory...")
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
    
    # Create enhanced cortex with BOTH systems
    cortex = AttractorCortex(
        columns=cols,
        input_proj=True,
        rhythm=rhythm,
        settling_steps=8,
        k_sparsity=2,
        
        # Episodic memory parameters
        buffer_capacity=100,
        max_episode_length=5,  # SHORTER for testing
        immediate_replay_threshold=0.5,  # ADJUSTED
        background_replay_batch=2,
        priority_decay=0.05,
        
        # Reward module parameters
        value_lr=0.1,
        baseline_reward=0.0
    )
    
    print(f"Created Enhanced AttractorCortex with:")
    print(f"  • {cortex.C} columns")
    print(f"  • RewardModule (V_lr={cortex.reward_module.value_lr})")
    print(f"  • Episodic buffer: {cortex.buffer_capacity} capacity")
    print(f"  • Max episode length: {cortex.max_episode_length}")
    print(f"  • Immediate replay threshold: {cortex.immediate_replay_threshold}")
    
    # Track big events
    big_reward_steps = []
    surprise_steps = []
    
    # Run simulation
    print(f"\nRunning integrated simulation (50 steps)...")
    print("-" * 70)
    print("Step | RawReward | Dopamine(RPE) | V | Buffer | Theta | I/B Replays")
    print("-" * 70)
    
    prev_state = None
    
    for step in range(50):
        # ====== CREATE MEANINGFUL INPUT AND REWARD ======
        x = torch.randn(input_size)
        raw_reward = 0.1  # Default baseline
        
        # Schedule BIG rewards at specific steps
        if step == 5:
            raw_reward = 2.0  # Big positive
            big_reward_steps.append(step)
            x = torch.ones(input_size) * 0.5  # Distinct pattern
        elif step == 15:
            raw_reward = 1.5  # Medium positive
            big_reward_steps.append(step)
            x = torch.ones(input_size) * -0.5  # Distinct pattern
        elif step == 25:
            raw_reward = -1.0  # Negative
            big_reward_steps.append(step)
            x = torch.zeros(input_size)  # Distinct pattern
        elif step == 35:
            raw_reward = 3.0  # Very big positive
            big_reward_steps.append(step)
            x = torch.randn(input_size) * 2.0  # High variance
        
        # Add surprising patterns every 7 steps
        if step % 7 == 0 and step > 0:
            raw_reward += 0.5  # Bonus for surprise
            surprise_steps.append(step)
            x = torch.randn(input_size) * 3.0  # High variance = surprising
        
        # ====== EXECUTE CORTEX STEP ======
        current_state = cortex_step(cortex, x, raw_reward)
        
        # Temporal learning
        if prev_state is not None:
            cortex.learn_temporal(current_state, da_signal=cortex.last_dopamine_signal)
        
        prev_state = current_state.clone()
        
        # ====== DEBUG OUTPUT ======
        if step % 5 == 0 or step in big_reward_steps or step in surprise_steps:
            stats = cortex.get_memory_statistics()
            theta_active = cortex.rhythm.theta_in_window() if cortex.rhythm else False
            
            event_marker = ""
            if step in big_reward_steps:
                event_marker = "🎯"
            elif step in surprise_steps:
                event_marker = "⚡"
            
            print(f"{event_marker}{step:3d} | "
                  f"Reward: {raw_reward:5.2f} | "
                  f"Dopamine: {cortex.last_dopamine_signal:6.3f} | "
                  f"V: {cortex.reward_module.V:5.3f} | "
                  f"Buffer: {stats['buffer_size']:2d} | "
                  f"Theta: {'✓' if theta_active else '✗'} | "
                  f"Replays: {stats['immediate_replays']}I/{stats['background_replays']}B")
    
    print(f"\nSimulation completed!")
    
    # ====== DETAILED ANALYSIS ======
    print(f"\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)
    
    # Reward Module Analysis
    print(f"\n1. REWARD MODULE ANALYSIS:")
    print("-" * 40)
    print(f"Final V estimate: {cortex.reward_module.V:.3f}")
    print(f"Big reward events at steps: {big_reward_steps}")
    print(f"Surprise events at steps: {surprise_steps}")
    
    # Episode Analysis
    print(f"\n2. EPISODE ANALYSIS (all {len(cortex.episode_buffer.buffer)} episodes):")
    print("-" * 40)
    
    if len(cortex.episode_buffer.buffer) > 0:
        immediate_triggered = 0
        high_priority_episodes = []
        
        for i, episode in enumerate(cortex.episode_buffer.buffer):
            above_threshold = episode.priority > cortex.immediate_replay_threshold
            triggered = episode.replay_count > 0
            
            if above_threshold:
                high_priority_episodes.append(episode)
            
            if triggered:
                immediate_triggered += 1
            
            status = "🚨" if triggered else "  "
            threshold_status = "ABOVE" if above_threshold else "below"
            
            print(f"{status} Ep {episode.id:2d}: "
                  f"Priority={episode.priority:.3f} ({threshold_status} {cortex.immediate_replay_threshold}), "
                  f"Steps={episode.length}, "
                  f"AvgReward={episode.total_reward/episode.length:.3f}, "
                  f"Replays={episode.replay_count}")
    
    # Immediate Replay Debug
    print(f"\n3. IMMEDIATE REPLAY DEBUG:")
    print("-" * 40)
    
    if len(cortex.episode_buffer.buffer) > 0:
        # Find episodes that SHOULD have triggered immediate replay
        should_trigger = [ep for ep in cortex.episode_buffer.buffer 
                         if ep.priority > cortex.immediate_replay_threshold]
        
        actually_triggered = [ep for ep in should_trigger if ep.replay_count > 0]
        
        print(f"Episodes above threshold ({cortex.immediate_replay_threshold}): {len(should_trigger)}")
        print(f"Episodes that actually triggered immediate replay: {len(actually_triggered)}")
        
        if len(should_trigger) > 0 and len(actually_triggered) == 0:
            print(f"\n❌ PROBLEM: Episodes above threshold but no immediate replay!")
            print(f"Possible causes:")
            print(f"1. _execute_immediate_replay() not being called")
            print(f"2. needs_immediate flag not set in add_episode()")
            print(f"3. Episode finishing after the step")
            
            # Debug the first high-priority episode
            if should_trigger:
                ep = should_trigger[0]
                print(f"\nDebug episode {ep.id}:")
                print(f"  Priority: {ep.priority:.3f}")
                print(f"  Threshold: {cortex.immediate_replay_threshold}")
                print(f"  Steps: {ep.length}")
                print(f"  Total reward: {ep.total_reward:.3f}")
                print(f"  Total surprise: {ep.total_surprise:.3f}")
    
    # System Check
    print(f"\n4. SYSTEM CHECK:")
    print("-" * 40)
    
    checks = [
        ("Reward stored in episodes", any(ep.total_reward != 0 for ep in cortex.episode_buffer.buffer)),
        ("Positive dopamine signals", any(step in big_reward_steps for step in range(50))),
        ("Episodes being created", len(cortex.episode_buffer.buffer) > 0),
        ("Background replays working", cortex.memory_stats['background_replays'] > 0),
        ("Immediate replays working", cortex.memory_stats['immediate_replays'] > 0),
    ]
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
    
    print(f"\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    
    if cortex.memory_stats['immediate_replays'] == 0:
        print("1. DEBUG _finish_current_episode() method - add print statements")
        print("2. Check EpisodeBuffer.add_episode() return value")
        print("3. Ensure immediate replay is called RIGHT AFTER episode finishes")
        print("4. Lower immediate_replay_threshold to 0.3")
    
    print(f"\nPhase 2.10 + 2.4 Integration: {'COMPLETE' if cortex.memory_stats['immediate_replays'] > 0 else 'NEEDS DEBUGGING'}")
    print("=" * 70)