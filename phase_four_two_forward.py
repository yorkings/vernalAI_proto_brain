import torch
from phase_two_column import ProtoColumn

class ForwardSimulator:
    """Reuse your CorticalBlock for imagination"""
    def __init__(self, cortical_block,lookahead_steps: int = 3):
        self.block = cortical_block
        self.lookahead_steps = lookahead_steps  # Lookahead steps
        
    def imagine_step(self, state: torch.Tensor, action_context: torch.Tensor):
        """Single step imagination with dimension handling"""
        # Get expected input size
        expected_size = self.block.columns[0].layer.input_size
        
        # Current combined size
        current_size = state.shape[0] + action_context.shape[0]
        
        if current_size == expected_size:
            # Perfect match
            combined = torch.cat([state, action_context])
        elif current_size < expected_size:
            # Pad with zeros
            pad_size = expected_size - current_size
            padding = torch.zeros(pad_size, dtype=state.dtype)
            combined = torch.cat([state, action_context, padding])
        else:
            # Truncate (this shouldn't happen if dimensions are set up correctly)
            print(f"[WARNING] Truncating input: {current_size} > {expected_size}")
            combined = torch.cat([state, action_context])[:expected_size]
        
        # Forward pass without updates
        with torch.no_grad():
            imagined = self.block.forward(combined)
            
        return imagined


class GoalSystem:
    """Goal maintenance like prefrontal cortex"""
    def __init__(self, input_size: int, goal_dim: int = 32):
        self.goal_column = ProtoColumn(input_size=input_size, num_neurons=goal_dim)
        self.goal_trace = torch.zeros(goal_dim)
        self.alpha = 0.95  # Trace decay
        
    def update_goal(self, goal_input: torch.Tensor):
        """Encode goal and update trace"""
        goal_vector = self.goal_column.forward(goal_input)
        
        # Leaky integration: biological goal persistence
        self.goal_trace = (
            self.alpha * self.goal_trace + 
            (1 - self.alpha) * goal_vector
        )
        
        return self.goal_trace    