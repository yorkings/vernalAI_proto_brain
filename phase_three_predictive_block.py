import torch
from typing import Optional
from phase_two_attractor_cotex import AttractorCortex 

class PredictiveBlock:
    def __init__(self, base_cortex: AttractorCortex, dim_rep: int,gen_init_scale: float = 0.05, rec_init_scale: float = 0.05,
                 kappa: float = 0.2,           # Inference step size
                 alpha_G: float = 1e-3,        # Generative learning rate
                 alpha_R: float = 1e-3,        # Recognition learning rate
                 precision_init: float = 1.0, timescale: int = 1,
                 level_idx: int = 0):          # Added: level index for debugging
        """
        base_cortex: Your existing AttractorCortex instance
        dim_rep: Dimensionality of representation
        timescale: Update frequency
        level_idx: Hierarchy level for debugging
        """
        self.base = base_cortex
        self.dim = dim_rep
        self.kappa = kappa
        self.alpha_G = alpha_G
        self.alpha_R = alpha_R
        self.timescale = timescale
        self.level_idx = level_idx  # For debugging
        
        # G: Generative map (top-down prediction)
        self.G = torch.randn(self.dim, self.dim) * gen_init_scale
        
        # R: Recognition map (bottom-up encoding)  
        self.R = torch.randn(self.dim, self.dim) * rec_init_scale
        
        # W_up: Projection of error upward (FIXED: proper initialization)
        self.W_up = torch.randn(self.dim, self.dim) * 0.02
        
        # Π: Precision weights (inverse variance)
        self.Pi = torch.ones(self.dim) * precision_init
        
        # STATE 
        self.o = torch.randn(self.dim) * 0.1  # Changed: small random init instead of zeros
        self.o_pred = torch.zeros(self.dim)
        self.error = torch.zeros(self.dim)
        
        # Bookkeeping
        self.step_counter = 0
        self.last_lower_state = None
        self.error_history = []
        self.precision_history = []
        
        # Normalize initial weights
        self.normalize_weights()

        # Ensure G matrix dimensions match (dim_rep x dim_rep)
        if self.G.shape != (self.dim, self.dim):
            self.G = torch.randn(self.dim, self.dim, dtype=torch.float32) * gen_init_scale
        
        # Ensure R matrix dimensions match (dim_rep x dim_rep)
        if self.R.shape != (self.dim, self.dim):
            self.R = torch.randn(self.dim, self.dim, dtype=torch.float32) * rec_init_scale
        
        # Ensure W_up matrix dimensions match (dim_rep x dim_rep)
        if self.W_up.shape != (self.dim, self.dim):
            self.W_up = torch.randn(self.dim, self.dim, dtype=torch.float32) * 0.02
        
        # Ensure Pi dimension matches dim_rep
        if self.Pi.shape[0] != self.dim:
            self.Pi = torch.ones(self.dim, dtype=torch.float32) * precision_init
            
        
    def normalize_weights(self):
        """Normalize weights for stability"""
        with torch.no_grad():
            # Normalize G and R rows
            for W in [self.G, self.R, self.W_up]:
                row_norms = torch.norm(W, dim=1, keepdim=True) + 1e-8
                W.data = W.data / row_norms
            
            # Ensure Pi is positive
            self.Pi.data = torch.clamp(self.Pi.data, min=1e-3, max=1e3)
    
    def summarize_lower(self) -> torch.Tensor:
        """Get summary from base cortex"""
        if self.base.stable_state is not None:
            return self.base.stable_state.clone().detach()
        elif self.base.last_o is not None:
            return self.base.last_o.clone().detach()
        else:
            # Return random if no state (for initialization)
            return torch.randn(self.dim) * 0.1
    
    def topdown_predict(self, target_dim: Optional[int] = None) -> torch.Tensor:
        """Compute prediction for lower level: ô = G @ o, with optional dimension target"""
        with torch.no_grad():
            # Basic prediction
            self.o_pred = self.G @ self.o
            # Apply non-linearity
            self.o_pred = torch.tanh(self.o_pred)
            
            # If target dimension is specified and doesn't match, adjust
            if target_dim is not None and self.o_pred.shape[0] != target_dim:
                if self.o_pred.shape[0] < target_dim:
                    # Pad with zeros
                    pad_size = target_dim - self.o_pred.shape[0]
                    self.o_pred = torch.cat([self.o_pred, torch.zeros(pad_size, dtype=self.o_pred.dtype)])
                else:
                    # Truncate
                    self.o_pred = self.o_pred[:target_dim]
        
        return self.o_pred

    def compute_error(self, lower_o: torch.Tensor) -> torch.Tensor:
        """Compute prediction error: ε = lower_o - ô, with dimension safety"""
        # Ensure dimensions match
        if self.o_pred.shape[0] != lower_o.shape[0]:
            # Resize prediction
            if self.o_pred.shape[0] < lower_o.shape[0]:
                pad_size = lower_o.shape[0] - self.o_pred.shape[0]
                self.o_pred = torch.cat([
                    self.o_pred, 
                    torch.zeros(pad_size, dtype=self.o_pred.dtype)
                ])
            else:
                self.o_pred = self.o_pred[:lower_o.shape[0]]
        
        self.error = lower_o - self.o_pred
        return self.error
    
    def infer_update(self, lower_o: torch.Tensor) -> torch.Tensor:
        """
        Bottom-up inference update with proper dimension handling
        """
        # Get prediction with correct target dimension
        self.topdown_predict(target_dim=lower_o.shape[0])
        
        # Compute error
        err = self.compute_error(lower_o)
        
        # Clamp error to prevent explosion
        err = torch.clamp(err, -2.0, 2.0)
        # FIX: Ensure prediction matches input dimension
        if self.o_pred.shape[0] != lower_o.shape[0]:
            # Resize prediction to match input
            if self.o_pred.shape[0] < lower_o.shape[0]:
                pad_size = lower_o.shape[0] - self.o_pred.shape[0]
                padding = torch.zeros(pad_size, dtype=self.o_pred.dtype)
                self.o_pred = torch.cat([self.o_pred, padding])
            else:
                self.o_pred = self.o_pred[:lower_o.shape[0]]
        
        # Precision weighting with dimension checking
        Pi_normalized = self.Pi / (self.Pi.sum() + 1e-8)
        
        # Resize Pi if needed
        if Pi_normalized.shape[0] != err.shape[0]:
            if Pi_normalized.shape[0] > err.shape[0]:
                Pi_normalized = Pi_normalized[:err.shape[0]]
            else:
                # Pad with ones (neutral weight)
                pad_size = err.shape[0] - Pi_normalized.shape[0]
                Pi_normalized = torch.cat([Pi_normalized, torch.ones(pad_size, dtype=Pi_normalized.dtype)])
        
        weighted_err = Pi_normalized * err
        
        # Project error upward with dimension safety
        # Ensure W_up has correct dimensions
        if self.W_up.shape[1] != weighted_err.shape[0]:
            # Adjust W_up columns
            if self.W_up.shape[1] < weighted_err.shape[0]:
                # Add columns
                pad_cols = weighted_err.shape[0] - self.W_up.shape[1]
                self.W_up = torch.cat([
                    self.W_up, 
                    torch.zeros((self.W_up.shape[0], pad_cols), dtype=self.W_up.dtype)
                ], dim=1)
            else:
                # Remove extra columns
                self.W_up = self.W_up[:, :weighted_err.shape[0]]
        
        delta = self.W_up @ weighted_err
        
        # Ensure delta matches self.o dimensions
        if delta.shape[0] != self.o.shape[0]:
            # Simple resizing: truncate or pad
            if delta.shape[0] > self.o.shape[0]:
                delta = delta[:self.o.shape[0]]
            else:
                pad_size = self.o.shape[0] - delta.shape[0]
                delta = torch.cat([delta, torch.zeros(pad_size, dtype=delta.dtype)])
        
        # Update with leaky integration
        self.o = 0.9 * self.o + 0.1 * (self.o + self.kappa * delta)
        
        # Apply non-linearity
        self.o = torch.tanh(self.o)
        
        return err
    
    # SIMPLEST FIX: Update local_learn to resize everything
    def local_learn(self, lower_o: torch.Tensor, 
                    delta_da: Optional[float] = None,
                    plasticity_gate: float = 1.0):
        """
        Simplified local learning that handles dimension mismatches
        """
        # Ensure dimensions match by resizing lower_o to our dimension
        if lower_o.shape[0] != self.dim:
            if lower_o.shape[0] > self.dim:
                # Truncate
                lower_o = lower_o[:self.dim]
            else:
                # Pad with zeros
                pad_size = self.dim - lower_o.shape[0]
                lower_o = torch.cat([lower_o, torch.zeros(pad_size, dtype=lower_o.dtype)])
        
        # Get prediction for our dimension
        self.topdown_predict(target_dim=self.dim)
        err = lower_o - self.o_pred
        err = torch.clamp(err, -1.0, 1.0)
        
        # Simple Hebbian learning (skip precision weighting for now)
        dG = self.alpha_G * torch.ger(err, self.o) * plasticity_gate
        dR = self.alpha_R * torch.ger(self.o, err) * plasticity_gate
        
        if delta_da is not None:
            dG *= (1.0 + 0.5 * delta_da)
            dR *= (1.0 + 0.5 * delta_da)
        
        self.G += dG
        self.R += dR
        
        # Simple precision update
        error_mag = err.abs().mean().item()
        self.Pi = 0.99 * self.Pi + 0.01 * (1.0 / (1.0 + error_mag))
        
        return dG.norm().item(), dR.norm().item()
    
    def step(self, x: Optional[torch.Tensor] = None,
             lower_prediction: Optional[torch.Tensor] = None,
             do_learn: bool = True,
             delta_da: Optional[float] = None,
             plasticity_gate: float = 1.0) -> torch.Tensor:
        """
        One hierarchical step with improved stability
        """
        self.step_counter += 1
        
        # Only process on timescale multiples
        if (self.step_counter % self.timescale) != 0:
            return self.o
        
        # UPDATE BASE CORTEX with prediction integration
        if x is not None:
            if lower_prediction is not None:
                # Adaptive weighting based on precision
                if self.Pi.mean() > 0.5:  # High precision -> trust prediction more
                    pred_weight = 0.6
                else:  # Low precision -> trust sensory more
                    pred_weight = 0.2
                
                # Ensure dimensions match
                if lower_prediction.shape[0] != x.shape[0]:
                    # Reshape or interpolate prediction
                    if lower_prediction.shape[0] < x.shape[0]:
                        # Pad prediction
                        pad_size = x.shape[0] - lower_prediction.shape[0]
                        lower_prediction = torch.cat([lower_prediction, 
                                                     torch.zeros(pad_size)])
                    else:
                        # Truncate prediction
                        lower_prediction = lower_prediction[:x.shape[0]]
                
                combined_input = (1 - pred_weight) * x + pred_weight * lower_prediction
                self.base.forward(combined_input)
            else:
                self.base.forward(x)
        
        # GET LOWER STATE
        lower_state = self.summarize_lower()
        self.last_lower_state = lower_state.clone()
        
        # INFERENCE UPDATE
        err = self.infer_update(lower_state)
        
        # LEARNING
        if do_learn:
            self.local_learn(lower_state, delta_da, plasticity_gate)
        
        return self.o
    
    def reset_state(self):
        """Reset dynamic state (keep weights)"""
        self.o = torch.randn(self.dim) * 0.1
        self.o_pred = torch.zeros(self.dim)
        self.error = torch.zeros(self.dim)
        self.last_lower_state = None
        self.step_counter = 0
        self.error_history = []
        self.precision_history = []
        
        # Reset base cortex if it has reset method
        if hasattr(self.base, 'reset_memory'):
            self.base.reset_memory()
    
    def get_debug_info(self):
        """Get debugging information"""
        return {
            'level': self.level_idx,
            'o_norm': self.o.norm().item(),
            'error_norm': self.error.norm().item() if self.error is not None else 0.0,
            'precision_mean': self.Pi.mean().item(),
            'timescale': self.timescale,
            'steps': self.step_counter,
            'recent_errors': self.error_history[-10:] if self.error_history else [],
            'recent_precision': self.precision_history[-10:] if self.precision_history else []
        }