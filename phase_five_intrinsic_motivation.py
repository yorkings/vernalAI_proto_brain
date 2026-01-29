# phase_five_intrinsic_motivation.py
"""
Phase 5.9: Intrinsic Motivation Engine
- NoveltyMap for state novelty tracking
- CompetenceTracker for learning progress
- Combined intrinsic reward generation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from collections import deque, defaultdict

class NoveltyMap:
    """Coarse quantized novelty counter for state fingerprints"""
    
    def __init__(self, 
                 decay: float = 0.999, 
                 bins: int = 8,
                 novelty_scale: float = 0.2):
        self.visits: Dict[Tuple[int, ...], float] = defaultdict(float)
        self.decay = decay
        self.bins = bins
        self.novelty_scale = novelty_scale
        
    def quantize(self, state: torch.Tensor) -> Tuple[int, ...]:
        """Quantize state vector to discrete bins"""
        if state is None:
            return tuple([0] * min(4, self.bins))
        
        # Ensure we have a finite tensor
        state_np = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.array(state)
        
        # Flatten and take first few dimensions for fingerprint
        flat_state = state_np.flatten()[:16]  # Use first 16 dimensions max
        
        # Normalize and quantize
        clipped = np.tanh(flat_state)  # Bound to [-1, 1]
        q = tuple(((clipped + 1.0) * 0.5 * (self.bins - 1)).astype(int))
        
        # Pad or truncate to consistent length
        target_len = 8
        if len(q) < target_len:
            q = q + (0,) * (target_len - len(q))
        else:
            q = q[:target_len]
            
        return q
    
    def decay_all(self) -> None:
        """Apply decay to all visit counts"""
        keys = list(self.visits.keys())
        for key in keys:
            self.visits[key] *= self.decay
            if self.visits[key] < 1e-6:
                del self.visits[key]
    
    def visit(self, state: torch.Tensor) -> Tuple[Tuple[int, ...], float]:
        """Record state visit and return novelty score"""
        key = self.quantize(state)
        
        # Update visit count with decay
        current_count = self.visits.get(key, 0.0)
        self.visits[key] = current_count * self.decay + 1.0
        
        # Novelty score: inverse of visit count (more visits = less novel)
        # Use sqrt for smoother decay
        novelty = 1.0 / math.sqrt(self.visits[key] + 1.0)
        
        return key, novelty * self.novelty_scale
    
    def novelty_score(self, state: torch.Tensor) -> float:
        """Get novelty score without recording visit"""
        key = self.quantize(state)
        count = self.visits.get(key, 0.0)
        return 1.0 / math.sqrt(count + 1.0) * self.novelty_scale
    
    def get_statistics(self) -> Dict:
        """Get novelty map statistics"""
        if not self.visits:
            return {'total_states': 0, 'avg_visits': 0.0}
        
        total_visits = sum(self.visits.values())
        avg_visits = total_visits / len(self.visits)
        
        return {
            'total_states': len(self.visits),
            'total_visits': total_visits,
            'avg_visits': avg_visits,
            'max_visits': max(self.visits.values()) if self.visits else 0.0
        }

class CompetenceTracker:
    """Tracks sliding windows of prediction error for learning progress"""
    
    def __init__(self, 
                 window_size: int = 50,
                 lp_scale: float = 1.0):
        self.window_size = window_size
        self.lp_scale = lp_scale
        self.recent_errors: deque = deque(maxlen=window_size)
        self.older_errors: deque = deque(maxlen=window_size)
        self._tick = 0
        self._window_shift_counter = 0
        
    def push_error(self, error: float) -> None:
        """Add new prediction error"""
        self.recent_errors.append(error)
        self._tick += 1
        self._window_shift_counter += 1
        
        # Shift windows when enough samples collected
        if self._window_shift_counter >= self.window_size:
            self.older_errors = deque(list(self.recent_errors), maxlen=self.window_size)
            self.recent_errors.clear()
            self._window_shift_counter = 0
    
    def learning_progress(self) -> float:
        """Compute learning progress: reduction in prediction error"""
        if not self.older_errors:
            return 0.0
        
        if not self.recent_errors:
            return 0.0
        
        old_mean = float(np.mean(list(self.older_errors)))
        new_mean = float(np.mean(list(self.recent_errors)))
        
        # Positive when error decreased (improvement)
        lp = (old_mean - new_mean) * self.lp_scale
        
        # Clip to reasonable range
        return max(-1.0, min(1.0, lp))
    
    def get_statistics(self) -> Dict:
        """Get competence tracker statistics"""
        if not self.recent_errors and not self.older_errors:
            return {'recent_errors': 0, 'older_errors': 0, 'progress': 0.0}
        
        recent_stats = {
            'count': len(self.recent_errors),
            'mean': float(np.mean(list(self.recent_errors))) if self.recent_errors else 0.0,
            'std': float(np.std(list(self.recent_errors))) if len(self.recent_errors) > 1 else 0.0
        }
        
        older_stats = {
            'count': len(self.older_errors),
            'mean': float(np.mean(list(self.older_errors))) if self.older_errors else 0.0,
            'std': float(np.std(list(self.older_errors))) if len(self.older_errors) > 1 else 0.0
        }
        
        return {
            'recent': recent_stats,
            'older': older_stats,
            'learning_progress': self.learning_progress(),
            'total_samples': self._tick
        }

class IntrinsicMotivationEngine:
    """Combines novelty and competence progress for intrinsic motivation"""
    
    def __init__(self,
                 c_nov: float = 0.2,      # Novelty weight
                 c_lp: float = 1.0,       # Learning progress weight
                 novelty_decay: float = 0.999,
                 window_size: int = 50):
        
        self.c_nov = c_nov
        self.c_lp = c_lp
        
        # Components
        self.novelty_map = NoveltyMap(decay=novelty_decay, novelty_scale=c_nov)
        self.competence_tracker = CompetenceTracker(window_size=window_size, lp_scale=c_lp)
        
        # State tracking
        self.last_intrinsic_reward = 0.0
        self.reward_history = []
        
    def update(self, 
               state: torch.Tensor, 
               prediction_error: float) -> float:
        """
        Update intrinsic motivation systems and compute reward
        
        Args:
            state: Current state/observation
            prediction_error: Current prediction error magnitude
            
        Returns:
            Intrinsic reward value
        """
        # Update competence tracker with prediction error
        self.competence_tracker.push_error(prediction_error)
        
        # Compute novelty
        novelty_key, novelty_score = self.novelty_map.visit(state)
        
        # Apply decay to all novelty counts
        self.novelty_map.decay_all()
        
        # Compute learning progress
        learning_progress = self.competence_tracker.learning_progress()
        
        # Combine intrinsic rewards
        R_intrinsic = novelty_score + learning_progress
        
        # Clip to reasonable range
        R_intrinsic = max(-1.0, min(1.0, R_intrinsic))
        
        # Store
        self.last_intrinsic_reward = R_intrinsic
        self.reward_history.append(R_intrinsic)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
        
        return R_intrinsic
    
    def get_intrinsic_reward_components(self, state: torch.Tensor) -> Dict:
        """Get breakdown of intrinsic reward components without updating"""
        novelty_score = self.novelty_map.novelty_score(state)
        learning_progress = self.competence_tracker.learning_progress()
        
        return {
            'novelty': novelty_score,
            'learning_progress': learning_progress,
            'total': novelty_score + learning_progress
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        novelty_stats = self.novelty_map.get_statistics()
        competence_stats = self.competence_tracker.get_statistics()
        
        if self.reward_history:
            reward_stats = {
                'mean': float(np.mean(self.reward_history)),
                'std': float(np.std(self.reward_history)),
                'min': float(np.min(self.reward_history)),
                'max': float(np.max(self.reward_history)),
                'recent': self.reward_history[-10:] if len(self.reward_history) >= 10 else self.reward_history
            }
        else:
            reward_stats = {}
        
        return {
            'novelty': novelty_stats,
            'competence': competence_stats,
            'intrinsic_reward': reward_stats,
            'last_reward': self.last_intrinsic_reward,
            'weights': {'c_nov': self.c_nov, 'c_lp': self.c_lp}
        }
    
    def reset(self):
        """Reset intrinsic motivation engine"""
        self.novelty_map = NoveltyMap(decay=self.novelty_map.decay, 
                                     novelty_scale=self.c_nov)
        self.competence_tracker = CompetenceTracker(window_size=self.competence_tracker.window_size,
                                                   lp_scale=self.c_lp)
        self.last_intrinsic_reward = 0.0
        self.reward_history = []

# Test function
def test_intrinsic_motivation():
    """Test intrinsic motivation engine"""
    print("Testing Intrinsic Motivation Engine...")
    
    engine = IntrinsicMotivationEngine(
        c_nov=0.2,
        c_lp=1.0,
        window_size=20  # Smaller for testing
    )
    
    # Simulate learning progression
    print("\nSimulating learning with decreasing error:")
    
    for step in range(100):
        # Simulate state (somewhat random, somewhat repeating)
        if step % 10 == 0:
            state = torch.randn(10) * 2.0  # Novel state
        else:
            state = torch.randn(10) * 0.5  # Familiar state
        
        # Simulate decreasing prediction error (learning)
        prediction_error = 1.0 / (1.0 + step * 0.1) + torch.randn(1).item() * 0.1
        
        # Update engine
        R_int = engine.update(state, prediction_error)
        
        if step % 20 == 0:
            stats = engine.get_intrinsic_reward_components(state)
            print(f"Step {step:3d}: Error={prediction_error:.3f}, "
                  f"R_int={R_int:.3f} (Nov={stats['novelty']:.3f}, LP={stats['learning_progress']:.3f})")
    
    # Final statistics
    print("\nFinal Statistics:")
    stats = engine.get_statistics()
    
    print(f"Novelty: {stats['novelty']['total_states']} unique states")
    print(f"Learning Progress: {stats['competence']['learning_progress']:.3f}")
    
    if 'intrinsic_reward' in stats:
        print(f"Intrinsic Reward - Mean: {stats['intrinsic_reward'].get('mean', 0):.3f}, "
              f"Std: {stats['intrinsic_reward'].get('std', 0):.3f}")
    
    # Verify intrinsic reward is positive during learning
    if stats['competence']['learning_progress'] > 0:
        print("✓ Positive learning progress detected")
    else:
        print("✗ Check learning progress calculation")
    
    return engine

if __name__ == "__main__":
    test_intrinsic_motivation()