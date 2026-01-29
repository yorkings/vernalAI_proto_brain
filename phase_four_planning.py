import torch
from phase_four_two_forward import ForwardSimulator, GoalSystem
from collections import deque
from typing import List

class BiologicalPlanner:
    """Decision making with forward simulation"""
    def __init__(self, forward_sim: ForwardSimulator, goal_system: GoalSystem):
        self.forward_sim = forward_sim
        self.goal_system = goal_system
        
        # Get state dimension from forward simulator's cortical block
        self.state_dim = forward_sim.block.C  # Should be 4(number of columns)
        
        # Get goal dimension from goal system
        self.goal_dim = goal_system.goal_column.layer.num_neurons 
        
        print(f"[PLANNER INIT] State dim: {self.state_dim}, Goal dim: {self.goal_dim}")
        
        # Create projection from goal space to state space
        if self.state_dim != self.goal_dim:
            print(f"[PLANNER INIT] Creating projection: {self.goal_dim} -> {self.state_dim}")
            self.goal_projection = torch.nn.Linear(self.goal_dim, self.state_dim, bias=False)
            # Initialize with simple scaling
            with torch.no_grad():
                scale = 1.0 / self.goal_dim
                self.goal_projection.weight.data.fill_(scale)
        else:
            self.goal_projection = None
        
        # Action primitives (micro-actions)
        self.actions = [
            torch.tensor([1.0, 0.0, 0.0]),  # Left
            torch.tensor([0.0, 1.0, 0.0]),  # Center  
            torch.tensor([0.0, 0.0, 1.0]),  # Right
        ]
        
        # Biological momentum
        self.action_momentum = torch.zeros(len(self.actions))
        self.beta = 0.3  # Surprise weight
        
        # Loop detection
        self.recent_fingerprints = deque(maxlen=10)
        
    def _project_goal_to_state(self, goal_trace: torch.Tensor) -> torch.Tensor:
        """Project goal trace to state dimension"""
        if self.goal_projection is not None:
            return self.goal_projection(goal_trace)
        return goal_trace
    
    def _compute_fingerprint(self, chain: List[torch.Tensor]) -> float:
        """Compute a simple fingerprint for a chain of states"""
        if not chain:
            return 0.0
        # Simple fingerprint: mean of state norms
        return sum(s.norm().item() for s in chain) / len(chain)
    
    def _simulate_chain(self, current_state: torch.Tensor, action_idx: int) -> List[torch.Tensor]:
        """Simulate a chain for the given action"""
        predicted_chain = []
        s = current_state.clone()
        action_vec = self.actions[action_idx]
        
        for k in range(self.forward_sim.lookahead_steps):
            action_context = action_vec if k == 0 else torch.zeros_like(action_vec)
            s = self.forward_sim.imagine_step(s, action_context)
            predicted_chain.append(s)
        
        return predicted_chain
    
    def plan(self, current_state: torch.Tensor, goal_trace: torch.Tensor):
        """Choose best action with forward simulation"""
        print(f"[PLANNING DEBUG] current_state shape: {current_state.shape}, norm: {current_state.norm().item():.3f}")
        print(f"[PLANNING DEBUG] goal_trace shape: {goal_trace.shape}, norm: {goal_trace.norm().item():.3f}")
        
        # Project goal to state dimension
        projected_goal = self._project_goal_to_state(goal_trace)
        print(f"[PLANNING DEBUG] projected_goal shape: {projected_goal.shape}, norm: {projected_goal.norm().item():.3f}")
        
        best_action = None
        best_score = -float('inf')
        
        for i, action_vec in enumerate(self.actions):
            print(f"[PLANNING DEBUG] Testing action {i}: {action_vec}")
            
            try:
                # Forward simulation K steps
                predicted_chain = self._simulate_chain(current_state, i)
                
                # Final predicted state
                final_state = predicted_chain[-1]
                
                # Score = -distance_to_goal - surprise
                distance = (final_state - projected_goal).norm().item()
                print(f"[PLANNING DEBUG] Action {i} distance to goal: {distance:.3f}")
                
                # Compute surprise (prediction error chain)
                if len(predicted_chain) > 1:
                    surprises = []
                    for pred, next_pred in zip(predicted_chain[:-1], predicted_chain[1:]):
                        # Simple surprise: unpredictability of transitions
                        zero_action = torch.zeros_like(action_vec)
                        predicted_next = self.forward_sim.imagine_step(pred, zero_action)
                        surprise = (next_pred - predicted_next).norm().item()
                        surprises.append(surprise)
                    avg_surprise = sum(surprises) / len(surprises) if surprises else 0.0
                else:
                    avg_surprise = 0.0
                
                # Action momentum bonus
                momentum_bonus = self.action_momentum[i] * 0.2
                
                # Loop detection penalty
                loop_penalty = 0.0
                chain_fingerprint = self._compute_fingerprint(predicted_chain)
                if chain_fingerprint in self.recent_fingerprints:
                    loop_penalty = 0.5  # Penalize repeated patterns
                
                score = -distance - self.beta * avg_surprise + momentum_bonus - loop_penalty
                print(f"[PLANNING DEBUG] Action {i} score: {score:.3f} "
                      f"(dist: {distance:.3f}, surprise: {avg_surprise:.3f}, "
                      f"momentum: {momentum_bonus:.3f}, loop: {loop_penalty:.3f})")
                
                if score > best_score:
                    best_score = score
                    best_action = i
                    
            except Exception as e:
                print(f"[PLANNING ERROR] Action {i} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Update momentum
        if best_action is not None:
            self.action_momentum = 0.8 * self.action_momentum
            self.action_momentum[best_action] += 0.2
            
            # Store fingerprint for loop detection
            imagined_chain = self._simulate_chain(current_state, best_action)
            fingerprint = self._compute_fingerprint(imagined_chain)
            self.recent_fingerprints.append(fingerprint)
            
            print(f"[PLANNING RESULT] Best action: {best_action}, Score: {best_score:.3f}")
        else:
            print(f"[PLANNING WARNING] No valid action found!")
        
        return best_action, best_score
    
    def reset(self):
        """Reset planner state"""
        self.action_momentum = torch.zeros(len(self.actions))
        self.recent_fingerprints.clear()
        print("[PLANNER] Reset complete")



# Test function
if __name__ == "__main__":
    print("Testing BiologicalPlanner...")
    
    # Mock objects for testing
    class MockForwardSim:
        def __init__(self):
            self.block = type('MockBlock', (), {'C': 4})()  # state_dim = 4
            self.lookahead_steps = 3
        
        def imagine_step(self, state, action):
            # Simple mock: add some noise
            return state + torch.randn_like(state) * 0.1
    
    class MockGoalSystem:
        def __init__(self):
            self.goal_column = type('MockColumn', (), {
                'layer': type('MockLayer', (), {'num_neurons': 32})()
            })()
    
    # Create planner
    planner = BiologicalPlanner(MockForwardSim(), MockGoalSystem())
    
    # Test planning
    test_state = torch.randn(4)
    test_goal = torch.randn(32)
    
    action, score = planner.plan(test_state, test_goal)
    print(f"\nTest result: Action={action}, Score={score}")