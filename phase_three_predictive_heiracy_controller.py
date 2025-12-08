import torch
from typing import Optional

class HierarchicalController:
    def __init__(self, rep_dim: int, action_dim: int, 
                 init_scale: float = 0.05, learning_rate: float = 0.001):
        self.rep_dim = rep_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Linear readout matrix
        self.M = torch.randn(action_dim, rep_dim) * init_scale
        self.b = torch.zeros(action_dim)
        
        # State tracking
        self.last_top_state = None
        self.last_action_prior = None
        self.last_raw_action = None
        
        # Learning statistics
        self.learning_history = []
        
    def propose(self, top_o: torch.Tensor) -> torch.Tensor:
        """Generate action prior from top-level representation"""
        self.last_top_state = top_o.clone().detach()
        
        # Linear transformation
        raw_action = self.M @ top_o + self.b
        
        # Store raw action for debugging
        self.last_raw_action = raw_action.clone()
        
        # Non-linearity (bounded output)
        action_prior = torch.tanh(raw_action)
        
        self.last_action_prior = action_prior.clone()
        return action_prior
    
    def learn_from_outcome(self, reward: float, success: bool = True, 
                          learning_rate: Optional[float] = None):
        """Reinforcement learning for controller"""
        if self.last_top_state is None or self.last_action_prior is None:
            return
        
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        # Scale reward for stability
        reward = torch.tanh(torch.tensor(reward)).item()
        
        # Positive outcome: strengthen mapping
        if success or reward > 0:
            # Hebbian-like update
            delta_M = lr * reward * torch.ger(self.last_action_prior, self.last_top_state)
            delta_b = lr * reward * self.last_action_prior * 0.1  # Smaller bias update
            
            self.M += delta_M
            self.b += delta_b
            
            # Record learning
            self.learning_history.append({
                'reward': reward,
                'M_norm': delta_M.norm().item(),
                'b_norm': delta_b.norm().item()
            })
            
            # Keep history bounded
            if len(self.learning_history) > 1000:
                self.learning_history.pop(0)
            
            # Normalize periodically
            if len(self.learning_history) % 100 == 0:
                self.normalize_weights()
    
    def normalize_weights(self):
        """Normalize controller weights for stability"""
        with torch.no_grad():
            # Normalize rows of M
            row_norms = torch.norm(self.M, dim=1, keepdim=True) + 1e-8
            self.M.data = self.M.data / row_norms
            
            # Clip bias
            self.b.data = torch.clamp(self.b.data, -2.0, 2.0)
    
    def reset(self):
        """Reset controller state"""
        self.last_top_state = None
        self.last_action_prior = None
        self.last_raw_action = None
    
    def get_debug_info(self):
        """Get debugging information"""
        return {
            'M_norm': self.M.norm().item(),
            'b_norm': self.b.norm().item(),
            'learning_steps': len(self.learning_history),
            'recent_updates': self.learning_history[-5:] if self.learning_history else []
        }