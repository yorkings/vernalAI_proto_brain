import torch
torch.set_default_dtype(torch.float32)
from typing import Optional,Dict
# In phase_two_modulated_neuron.py, update the ProtoNeuronMod class:
class ProtoNeuronMod:
    def __init__(self, input_size: int, learning_rate: float = 0.01,
                 decay: float = 0.9, init_scale: float = 0.1,
                 elig_decay: float = 0.9, 
                 local_lr: float = 0.01,
                 enable_phase5: bool = True,
                 
                 # Phase 5.10: Plasticity Gate parameters
                 enable_plasticity_gates: bool = True,
                 w1_gate: float = 1.0,
                 w2_gate: float = 0.2,
                 w3_gate: float = 0.5,
                 w4_gate: float = 1.0,
                 G_min: float = 0.05,
                 G_max: float = 1.5):
        
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.local_lr = local_lr
        self.gamma = decay
        
        # Weights & bias
        self.weights = ((torch.randn(input_size, dtype=torch.float32) * 2.0) - 1.0) * init_scale
        self.bias = ((torch.randn(1, dtype=torch.float32) * 2.0) - 1.0) * init_scale 
        
        # State
        self.trace = torch.tensor(0.0)
        self.last_input = torch.zeros(input_size)
        self.last_activation = torch.tensor(0.0)
        
        # Eligibility trace
        self.eligibility = torch.zeros_like(self.weights)
        self.elig_decay = elig_decay
        self.elig_trace = torch.zeros_like(self.weights) 
        
        # Phase 5.10: Plasticity Gates
        self.enable_plasticity_gates = enable_plasticity_gates and enable_phase5
        if self.enable_plasticity_gates:
            from phase_five_plastic_gates import PlasticityGate, PlasticityGateState
            
            # Create plasticity gate for this neuron
            self.plasticity_gate = PlasticityGate(
                w1=w1_gate,
                w2=w2_gate,
                w3=w3_gate,
                w4=w4_gate,
                G_min=G_min,
                G_max=G_max,
                debug=False
            )
            
            # Create gate states for each input synapse
            self.gate_states = [PlasticityGateState() for _ in range(input_size)]
            
            # Neuron-level prediction error tracking for confidence
            self.prediction_errors = []  # Store recent prediction errors
            self.max_error_history = 20
            
            print(f"[Neuron] Plasticity gates enabled with {input_size} synapses")
        else:
            self.plasticity_gate = None
            self.gate_states = None
        
        # Original Phase 5 fields (for backward compatibility)
        self.fatigue = 0.0
        self.local_confidence = 1.0
        self.recent_update_mag = 0.0
        self.enable_phase5 = enable_phase5
        
        # Plasticity gate (can be modulated)
        self.plasticity_gate_global = 1.0  # Global gate multiplier
    
    def update_prediction_error(self, error: float):
        """Update neuron-level prediction error for confidence calculation"""
        if not self.enable_plasticity_gates:
            return
        
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > self.max_error_history:
            self.prediction_errors.pop(0)
        
        # Update local confidence based on error history
        if len(self.prediction_errors) >= 2:
            # Simple confidence: inverse of average recent error
            avg_error = sum(self.prediction_errors) / len(self.prediction_errors)
            self.local_confidence = 1.0 / (1.0 + avg_error)
            self.local_confidence = max(0.1, min(1.0, self.local_confidence))
    
    def forward(self, x: torch.tensor):
        if x.shape[0] != self.input_size:
            raise ValueError("Input size mismatch")
        
        self.last_input = x.clone()
        z = torch.dot(self.weights, x) + self.bias.squeeze()
        a = torch.tanh(z)
        
        self.trace = self.gamma * self.trace + a
        self.last_activation = a
        
        # Phase 5: Update eligibility trace with clamping
        if self.enable_phase5:
            update = a * x
            self.eligibility = self.elig_decay * self.eligibility + update
            
            # Clamp eligibility to prevent explosion
            elig_norm = torch.norm(self.eligibility).item()
            if elig_norm > 5.0:
                self.eligibility = self.eligibility / elig_norm * 5.0
            
            self.elig_trace = 0.99 * self.elig_trace + 0.01 * update
            
            trace_norm = torch.norm(self.elig_trace).item()
            if trace_norm > 10.0:
                self.elig_trace = self.elig_trace / trace_norm * 10.0
      
        return a
    
    def learn(self, target=None, delta=None, gate: float = 1.0, mix_local: float = 0.5,
              DA_phasic: Optional[float] = None,
              DA_tonic: Optional[float] = None,
              DA_baseline: float = 0.0,
              prediction_error: Optional[float] = None):
        """
        Enhanced learn method with Phase 5.10 plasticity gates
        
        Args:
            DA_phasic: Phasic dopamine for plasticity gating
            DA_tonic: Tonic dopamine for plasticity gating
            DA_baseline: Dopamine baseline
            prediction_error: Current prediction error for confidence updates
        """
        # Update prediction error for confidence calculation
        if prediction_error is not None and self.enable_plasticity_gates:
            self.update_prediction_error(prediction_error)
        
        if target is not None:
            if not isinstance(target, torch.Tensor):
                target = torch.tensor([float(target)])
            a = self.last_activation
            error = target - a
            delta_w_local = self.local_lr * a * error * self.last_input
            delta_b_local = self.local_lr * error
        else:
            delta_w_local = torch.zeros_like(self.weights)
            delta_b_local = torch.tensor([0.0])
        
        if delta is not None:
            # Phase 5.10: Use plasticity gates if enabled
            if self.enable_plasticity_gates and DA_phasic is not None:
                # Apply plasticity gates to each synapse individually
                delta_w_da = torch.zeros_like(self.weights)
                
                for i in range(self.input_size):
                    # Get synapse-specific eligibility
                    synapse_eligibility = self.eligibility[i].item()
                    
                    # Base dopamine update for this synapse
                    synapse_delta = self.learning_rate * delta * synapse_eligibility * gate
                    
                    # Apply plasticity gate if we have dopamine signals
                    if self.plasticity_gate is not None and self.gate_states is not None:
                        # Use plasticity gate for this synapse
                        applied_delta = self.plasticity_gate.apply_gated_update(
                            synapse=self.weights[i],  # Direct weight access
                            delta_w=synapse_delta,
                            DA_phasic=DA_phasic,
                            DA_tonic=DA_tonic if DA_tonic is not None else 0.0,
                            DA_baseline=DA_baseline,
                            prediction_error=prediction_error if prediction_error is not None else 0.1,
                            lr=1.0,  # Already scaled by learning_rate
                            state=self.gate_states[i]
                        )
                        
                        # Convert back to scalar if needed
                        if isinstance(applied_delta, torch.Tensor):
                            delta_w_da[i] = applied_delta.item() if applied_delta.numel() == 1 else applied_delta
                        else:
                            delta_w_da[i] = applied_delta
                    else:
                        # Fallback to original dopamine update
                        delta_w_da[i] = synapse_delta
                
                # Bias update (simpler, use global gate)
                delta_b_da = self.learning_rate * delta * gate
                
                # Update fatigue (legacy, for backward compatibility)
                update_mag = torch.norm(delta_w_da).item()
                self.fatigue = 0.95 * self.fatigue + 0.05 * update_mag
                self.recent_update_mag = 0.9 * self.recent_update_mag + 0.1 * update_mag
                
            elif self.enable_phase5:
                # Original Phase 5 dopamine update (without detailed gating)
                if abs(delta) > 0.01:
                    local_gate = self.local_confidence * (1.0 / (1.0 + self.fatigue))
                    
                    # Ensure eligibility is bounded
                    bounded_eligibility = self.eligibility.clone()
                    elig_norm = torch.norm(bounded_eligibility).item()
                    if elig_norm > 5.0:
                        bounded_eligibility = bounded_eligibility / elig_norm * 5.0
                    
                    delta_w_da = self.learning_rate * delta * bounded_eligibility * (gate * self.plasticity_gate_global) * local_gate
                    delta_b_da = self.learning_rate * delta * (gate * self.plasticity_gate_global) * local_gate
                    
                    # Update fatigue
                    update_mag = torch.norm(delta_w_da).item()
                    self.fatigue = 0.95 * self.fatigue + 0.05 * update_mag
                    self.recent_update_mag = 0.9 * self.recent_update_mag + 0.1 * update_mag
                else:
                    delta_w_da = torch.zeros_like(self.weights)
                    delta_b_da = torch.tensor([0.0])
            else:
                # Original dopamine update
                delta_w_da = self.learning_rate * (delta * self.eligibility) * (gate * self.plasticity_gate_global)
                delta_b_da = self.learning_rate * delta * (gate * self.plasticity_gate_global)
        else:
            delta_w_da = torch.zeros_like(self.weights)
            delta_b_da = torch.tensor([0.0])
        
        # Blend updates
        w_update = (mix_local * delta_w_local) + ((1 - mix_local) * delta_w_da)
        b_update = (mix_local * delta_b_local) + ((1 - mix_local) * delta_b_da)
        
        # Apply updates
        self.weights += w_update
        self.bias += b_update
        
        # Phase 5: Simple weight normalization
        if self.enable_phase5:
            weight_norm = torch.norm(self.weights).item()
            if weight_norm > 3.0:
                self.weights = self.weights / weight_norm * 3.0
    
    def get_plasticity_gate_stats(self) -> Dict:
        """Get plasticity gate statistics for this neuron"""
        if not self.enable_plasticity_gates or self.plasticity_gate is None:
            return {'enabled': False}
        
        stats = self.plasticity_gate.get_statistics()
        stats['enabled'] = True
        stats['neuron_confidence'] = self.local_confidence
        stats['neuron_fatigue'] = self.fatigue
        
        # Add synapse-specific stats if available
        if self.gate_states:
            confidences = [s.local_confidence for s in self.gate_states]
            fatigues = [s.fatigue for s in self.gate_states]
            
            stats['synapse_stats'] = {
                'avg_confidence': sum(confidences) / len(confidences),
                'avg_fatigue': sum(fatigues) / len(fatigues),
                'confidence_range': (min(confidences), max(confidences)),
                'fatigue_range': (min(fatigues), max(fatigues))
            }
        
        return stats