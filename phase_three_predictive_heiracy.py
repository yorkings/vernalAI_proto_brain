import torch
from typing import List, Optional, Dict
from phase_three_predictive_block import PredictiveBlock

class PredictiveHierarchy:
    def __init__(self, levels: List[PredictiveBlock]):
        self.levels = levels
        self.L = len(levels)
        # Statistics with better tracking
        self.stats = {
            'total_steps': 0,
            'level_errors': [[] for _ in range(self.L)],
            'level_precisions': [[] for _ in range(self.L)],
            'convergence_history': [],
            'learning_updates': 0
        }
        
        # State tracking
        self.current_states = [None] * self.L
        # Initialize predictions with current states
        self.current_predictions = []
        for level in self.levels:
            # Initialize with whatever state the level has
            pred = level.topdown_predict() if hasattr(level, 'topdown_predict') else level.o.clone()
            self.current_predictions.append(pred)
        
    def step(self, sensory_input: torch.Tensor,delta_da: Optional[float] = None,plasticity_gate: float = 1.0,inference_iterations: int = 2) -> torch.Tensor:
        """
        Improved hierarchical inference-learning step with stability fixes
        """
        self.stats['total_steps'] += 1
        
        # 1. PROCESS SENSORY INPUT AT LEVEL 0
        level0_state = self.levels[0].step(
            x=sensory_input,
            do_learn=False,  # Defer learning to after error computation
            delta_da=delta_da,
            plasticity_gate=plasticity_gate
        )
        self.current_states[0] = level0_state
        
        # 2. BOTTOM-UP PASS
        current_lower_state = level0_state
        
        for ell in range(1, self.L):
            # Get prediction from level above (from previous iteration or initial)
            pred_from_above = (self.current_predictions[ell] 
                            if ell < len(self.current_predictions) 
                            and self.current_predictions[ell] is not None 
                            else None)
            
            # Update this level using lower state
            level_state = self.levels[ell].step(
                x=None,  # No direct sensory input for higher levels
                lower_prediction=pred_from_above,
                do_learn=False,
                delta_da=delta_da,
                plasticity_gate=plasticity_gate
            )
            
            self.current_states[ell] = level_state
            current_lower_state = level_state
        
        # 3. TOP-DOWN PREDICTION PASS
        for ell in range(self.L-1, 0, -1):
            prediction = self.levels[ell].topdown_predict()
            self.current_predictions[ell] = prediction
        
        # 4. ITERATIVE REFINEMENT (FIXED - SINGLE LOOP)
        for iteration in range(inference_iterations):
            # Bottom-up refinement
            for ell in range(1, self.L):
                # Get lower state with dimension checking
                lower_state = self.levels[ell-1].summarize_lower()
                # Re-infer with current predictions
                level_state = self.levels[ell].infer_update(lower_state)
                self.current_states[ell] = level_state
            
            # Top-down prediction update
            for ell in range(self.L-1, 0, -1):
                prediction = self.levels[ell].topdown_predict()
                self.current_predictions[ell] = prediction
        
        # 5. LEARNING PASS (now with proper coordination)
        for ell in range(self.L):
            if ell == 0:
                # Level 0 learns from its base cortex (already handled in step)
                # But we need to trigger learning if deferred
                lower_state = sensory_input
            else:
                # Higher levels learn from level below
                lower_state = self.levels[ell-1].summarize_lower()
                # Ensure lower_state dimensions match this level's expected input
                if lower_state.shape[0] != self.levels[ell].dim:
                    if lower_state.shape[0] > self.levels[ell].dim:
                        lower_state = lower_state[:self.levels[ell].dim]
                    else:
                        # Pad with zeros if smaller
                        pad_size = self.levels[ell].dim - lower_state.shape[0]
                        lower_state = torch.cat([lower_state,torch.zeros(pad_size, dtype=lower_state.dtype)])
            
            
            # For level 0, learning happened during forward pass
            # For higher levels, trigger learning now
            if ell > 0:
                self.levels[ell].local_learn(
                    lower_state,
                    delta_da=delta_da,
                    plasticity_gate=plasticity_gate
                )
                self.stats['learning_updates'] += 1
            
            # Track statistics
            error_norm = self.levels[ell].error.norm().item()
            precision_mean = self.levels[ell].Pi.mean().item()
            
            self.stats['level_errors'][ell].append(error_norm)
            self.stats['level_precisions'][ell].append(precision_mean)
            
            # Keep history bounded
            if len(self.stats['level_errors'][ell]) > 1000:
                self.stats['level_errors'][ell].pop(0)
                self.stats['level_precisions'][ell].pop(0)
        
        # 6. COMPUTE CONVERGENCE METRIC
        if len(self.stats['level_errors'][0]) > 1:
            recent_errors = self.stats['level_errors'][0][-10:]
            if len(recent_errors) >= 2:
                convergence = abs(recent_errors[-1] - recent_errors[-2])
                self.stats['convergence_history'].append(convergence)
                if len(self.stats['convergence_history']) > 100:
                    self.stats['convergence_history'].pop(0)
        print(f"[HIERARCHY DEBUG] current_states dimensions: {[s.shape if s is not None else 'None' for s in self.current_states]}")
        print(f"[HIERARCHY DEBUG] Returning state from level {len(self.current_states)-1} with dim {self.current_states[-1].shape[0]}")               
        return self.current_states[-1]
        
    def get_hierarchy_state(self) -> Dict:
        """Get detailed state of entire hierarchy"""
        state = {
            'levels': [],
            'errors': [],
            'precisions': [],
            'convergence': 0.0
        }
        
        for i, level in enumerate(self.levels):
            level_info = {
                'index': i,
                'representation_norm': level.o.norm().item(),
                'prediction_norm': level.o_pred.norm().item() if hasattr(level, 'o_pred') and level.o_pred is not None else 0.0,
                'error_norm': level.error.norm().item() if hasattr(level, 'error') and level.error is not None else 0.0,
                'precision_mean': level.Pi.mean().item() if hasattr(level, 'Pi') else 0.0,
                'timescale': level.timescale,
                'step_counter': level.step_counter
            }
            state['levels'].append(level_info)
            
            if hasattr(level, 'error') and level.error is not None:
                state['errors'].append(level.error.norm().item())
            else:
                state['errors'].append(0.0)
            
            if hasattr(level, 'Pi'):
                state['precisions'].append(level.Pi.mean().item())
            else:
                state['precisions'].append(0.0)
        
        # Compute convergence
        if self.stats['convergence_history']:
            state['convergence'] = sum(self.stats['convergence_history'][-5:]) / 5
        
        return state
    
    def get_detailed_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            **self.stats,
            'current_states': [s.norm().item() if s is not None else 0.0 
                              for s in self.current_states],
            'current_predictions': [p.norm().item() if p is not None else 0.0 
                                   for p in self.current_predictions]
        }
    
    def reset(self):
        """Reset entire hierarchy"""
        for level in self.levels:
            level.reset_state()
        
        self.current_states = [None] * self.L
        self.current_predictions = [None] * self.L
        
        self.stats = {
            'total_steps': 0,
            'level_errors': [[] for _ in range(self.L)],
            'level_precisions': [[] for _ in range(self.L)],
            'convergence_history': [],
            'learning_updates': 0
        }