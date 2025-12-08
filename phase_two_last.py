import torch
import numpy as np
import random
from typing import List


class EpisodeStep:
    """Single timestep in an episode"""
    def __init__(self,timestamp: int,input_vector: torch.Tensor,column_states: torch.Tensor,attractor_state: torch.Tensor, reward: float,surprise: float, pred_error: float):
        self.timestamp = timestamp
        self.input_vector = input_vector.clone()
        self.column_states = column_states.clone()
        self.attractor_state = attractor_state.clone()
        self.reward = float(reward)
        self.surprise = float(surprise)if surprise is not None else 0.0
        self.pred_error = float(pred_error)if pred_error is not None else 0.0

class Episode:
    """Complete episode stored in memory"""
    def __init__(self, start_timestamp: int, episode_id: int = 0):
        self.id = episode_id
        self.start_timestamp = start_timestamp
        self.steps: List[EpisodeStep] = []
        self.priority = 0.0
        self.replay_count = 0
        self.last_replayed = -1    
        # Aggregates
        self.total_reward = 0.0
        self.total_surprise = 0.0
        self.total_error = 0.0
        self.length = 0
        self.age = 0
        self.terminal = False
        
    def add_step(self, step: EpisodeStep):
        """Add a step to the episode"""
        self.steps.append(step)
        self.length += 1
        self.total_reward += step.reward
        self.total_surprise += step.surprise
        self.total_error += step.pred_error
        
    def finalize(self, terminal: bool = False, current_time: int = 0):
        """Finalize episode and compute priority"""
        self.terminal = terminal
        self.age = current_time - self.start_timestamp
        
        if self.length == 0:
            self.priority = 0.0
            return
        
        # Priority formula: P = α*S + β*R + γ*E + η/(1+τ*age)
        avg_surprise = self.total_surprise / self.length
        avg_reward = self.total_reward / self.length
        avg_error = self.total_error / self.length
        
        alpha = 1.0    # Surprise weight
        beta = 0.5     # Reward weight
        gamma = 0.7    # Error weight
        eta = 0.3      # Recency weight
        tau = 0.001    # Age decay
        
        self.priority = (alpha * avg_surprise +beta * avg_reward +gamma * avg_error + eta / (1.0 + tau * self.age))
        # Clip to reasonable range
        self.priority = max(0.0, min(self.priority, 10.0))
        
    def decay_priority(self, mu: float = 0.05):
        """Decay priority after replay"""
        self.priority *= np.exp(-mu)
        self.replay_count += 1
        
    def get_aggregate_dopamine(self) -> float:
        """Compute dopamine signal for replay learning"""
        if self.length == 0:
            return 0.0
            
        avg_reward = self.total_reward / self.length
        avg_surprise = self.total_surprise / self.length
        
        # Dopamine = reward + surprise bonus
        da_signal = avg_reward + 0.3 * avg_surprise
        return np.clip(da_signal, -1.0, 1.0)
        
    def get_steps_reversed(self) -> List[EpisodeStep]:
        """Get steps in reverse order (for replay)"""
        return list(reversed(self.steps))

class EpisodeBuffer:
    """Prioritized replay buffer for episodic memory"""
    def __init__(self, capacity: int = 200, priority_exponent: float = 0.6):
        self.capacity = capacity
        self.priority_exponent = priority_exponent
        self.buffer: List[Episode] = []
        self.next_id = 0
        
    def add_episode(self, episode: Episode,threshold: float = 1.5) -> bool:
        """Add episode, return True if needs immediate replay"""
        episode.id = self.next_id
        self.next_id += 1
        self.buffer.append(episode)
        # Evict if over capacity
        if len(self.buffer) > self.capacity:
            self._evict_lowest_priority()  
        #Use provided threshold (from cortex) or default    
        # Immediate replay if priority > threshold
        needs_immediate=episode.priority > threshold
         # DEBUG LOGGING
        if needs_immediate:
            print(f"[MEMORY] Episode {episode.id} triggered immediate replay: "
                f"priority={episode.priority:.3f} > threshold={threshold:.3f}") 
        return needs_immediate
            
    def _evict_lowest_priority(self):
        """Remove episode with lowest priority"""
        if not self.buffer:
            return
            
        min_idx = min(range(len(self.buffer)), 
                     key=lambda i: self.buffer[i].priority)
        self.buffer.pop(min_idx)
        
    def sample(self, k: int = 1) -> List[Episode]:
        """Sample k episodes using prioritized sampling"""
        if not self.buffer or k <= 0:
            return []
            
        priorities = torch.tensor([e.priority for e in self.buffer], 
                                 dtype=torch.float32)
        priorities = priorities.pow(self.priority_exponent) + 1e-8
        probs = priorities / priorities.sum()
        
        if k == 1:
            idx = torch.multinomial(probs, num_samples=1).item()
            return [self.buffer[idx]]
        else:
            idxs = torch.multinomial(probs, num_samples=min(k, len(self.buffer)), 
                                    replacement=False)
            return [self.buffer[i] for i in idxs]
            
    def compute_replay_count(self, episode: Episode, max_replays: int = 10) -> int:
        """Compute how many times to replay an episode"""
        if not self.buffer:
            return 1
            
        max_priority = max(e.priority for e in self.buffer)
        if max_priority == 0:
            return 1
            
        normalized = episode.priority / max_priority
        replays = int(np.ceil(1 + (max_replays - 1) * np.sqrt(normalized)))
        return min(replays, max_replays)