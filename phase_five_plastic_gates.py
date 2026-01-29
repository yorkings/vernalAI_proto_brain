# phase_five_plasticity_gates.py
"""
Phase 5.10: Plasticity Gates
- Per-synapse/neuron gating based on dopamine, confidence, and fatigue
- Adaptive learning rate based on update history
- Local confidence estimation from prediction reliability
"""

import torch
import math
from typing import Optional, Dict, List
from dataclasses import dataclass

@dataclass
class PlasticityGateState:
    """State for a single plasticity gate (can be per-synapse or per-neuron)"""
    fatigue: float = 0.0
    recent_update_sum: float = 0.0
    local_confidence: float = 1.0
    confidence_history: List[float] = None
    update_history: List[float] = None
    gate_value: float = 1.0
    last_gate_components: Dict = None
    
    def __post_init__(self):
        if self.confidence_history is None:
            self.confidence_history = []
        if self.update_history is None:
            self.update_history = []
        if self.last_gate_components is None:
            self.last_gate_components = {}

class PlasticityGate:
    """
    Implements gating logic for weight updates with fatigue, confidence, and dopamine modulation
    
    Gate equation:
    G = σ(w1*DA_phasic + w2*(DA_tonic - DA_baseline) + w3*local_confidence - w4*fatigue)
    
    Where:
    - DA_phasic: Phasic dopamine (immediate reward/surprise signal)
    - DA_tonic: Tonic dopamine (baseline motivation level)
    - local_confidence: How reliable predictions from this synapse/neuron are
    - fatigue: Accumulated recent update magnitude (prevents over-learning)
    """
    
    def __init__(self,
                 # Gate weight parameters
                 w1: float = 1.0,      # Phasic dopamine weight
                 w2: float = 0.2,      # Tonic dopamine weight  
                 w3: float = 0.5,      # Local confidence weight
                 w4: float = 1.0,      # Fatigue weight
                 
                 # Gate thresholds
                 G_min: float = 0.05,   # Minimum gate value (prevents complete shutdown)
                 G_max: float = 1.5,    # Maximum gate value (prevents explosion)
                 
                 # Fatigue parameters
                 lambda_f: float = 0.95,  # Fatigue decay rate
                 fatigue_scale: float = 1.0,  # Fatigue scaling factor
                 
                 # Confidence parameters
                 confidence_decay: float = 0.99,  # Confidence moving average decay
                 min_confidence: float = 0.1,     # Minimum confidence (prevents collapse)
                 
                 # Learning rate scaling
                 small_lr_scale: float = 0.01,    # Learning rate when gate is very low
                 
                 # Debug
                 debug: bool = False):
        
        # Parameters
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.G_min = G_min
        self.G_max = G_max
        self.lambda_f = lambda_f
        self.fatigue_scale = fatigue_scale
        self.confidence_decay = confidence_decay
        self.min_confidence = min_confidence
        self.small_lr_scale = small_lr_scale
        self.debug = debug
        
        # Default state (can be overridden per synapse/neuron)
        self.default_state = PlasticityGateState()
        
        # Statistics
        self.stats = {
            'gate_values': [],
            'fatigue_values': [],
            'confidence_values': [],
            'update_sizes': []
        }
    
    def compute_gate(self,
                    DA_phasic: float,
                    DA_tonic: float,
                    DA_baseline: float,
                    local_confidence: float,
                    fatigue: float,
                    state: Optional[PlasticityGateState] = None) -> float:
        """
        Compute gate value G ∈ [G_min, G_max]
        """
        # Raw gate computation
        raw_gate = (
            self.w1 * DA_phasic +
            self.w2 * (DA_tonic - DA_baseline) +
            self.w3 * local_confidence -
            self.w4 * fatigue
        )
        
        # Sigmoid activation (bounded between G_min and G_max)
        # Use modified sigmoid: G_min + (G_max - G_min) * σ(raw_gate)
        sigmoid = 1.0 / (1.0 + math.exp(-raw_gate))
        G = self.G_min + (self.G_max - self.G_min) * sigmoid
        
        # Store gate components if state provided
        if state is not None:
            state.last_gate_components = {
                'raw_gate': raw_gate,
                'DA_phasic_component': self.w1 * DA_phasic,
                'DA_tonic_component': self.w2 * (DA_tonic - DA_baseline),
                'confidence_component': self.w3 * local_confidence,
                'fatigue_component': -self.w4 * fatigue,
                'final_gate': G
            }
            state.gate_value = G
        
        return G
    
    def update_local_confidence(self,
                              prediction_error: float,
                              state: PlasticityGateState,
                              window_size: int = 10) -> float:
        """
        Update local confidence based on prediction error
        
        Confidence is high when predictions are consistently accurate
        Confidence decays when predictions are unreliable
        """
        # Store recent error
        state.confidence_history.append(prediction_error)
        if len(state.confidence_history) > window_size:
            state.confidence_history.pop(0)
        
        # Compute confidence: inverse of normalized error variance
        if len(state.confidence_history) >= 2:
            errors = torch.tensor(state.confidence_history) if isinstance(state.confidence_history[0], torch.Tensor) \
                    else torch.tensor(state.confidence_history, dtype=torch.float32)
            
            # Normalize errors to 0-1 range
            if errors.max() > errors.min():
                norm_errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
            else:
                norm_errors = torch.zeros_like(errors)
            
            # Confidence is 1 - (moving average of normalized errors)
            avg_error = norm_errors.mean().item()
            new_confidence = 1.0 - avg_error
            
            # Update with decay
            state.local_confidence = (
                self.confidence_decay * state.local_confidence +
                (1.0 - self.confidence_decay) * new_confidence
            )
        
        # Ensure minimum confidence
        state.local_confidence = max(self.min_confidence, state.local_confidence)
        
        return state.local_confidence
    
    def update_fatigue(self,
                      delta_w_magnitude: float,
                      state: PlasticityGateState) -> float:
        """
        Update fatigue based on recent update magnitudes
        
        Fatigue accumulates with large/rapid updates to prevent over-learning
        """
        # Store update magnitude
        state.update_history.append(delta_w_magnitude)
        if len(state.update_history) > 20:  # Keep last 20 updates
            state.update_history.pop(0)
        
        # Update fatigue with decay
        state.fatigue = (
            self.lambda_f * state.fatigue +
            (1.0 - self.lambda_f) * delta_w_magnitude * self.fatigue_scale
        )
        
        # Also track recent update sum
        state.recent_update_sum = (
            self.lambda_f * state.recent_update_sum +
            (1.0 - self.lambda_f) * delta_w_magnitude
        )
        
        return state.fatigue
    
    def apply_gated_update(self,
                          synapse,
                          delta_w: torch.Tensor,
                          DA_phasic: float,
                          DA_tonic: float,
                          DA_baseline: float,
                          prediction_error: float,
                          lr: float,
                          state: Optional[PlasticityGateState] = None) -> torch.Tensor:
        """
        Apply gated weight update
        
        Args:
            synapse: Object with .w attribute (the weight to update)
            delta_w: Proposed weight update (pre-gating)
            DA_phasic: Current phasic dopamine
            DA_tonic: Current tonic dopamine
            DA_baseline: Dopamine baseline
            prediction_error: Current prediction error for confidence update
            lr: Base learning rate
            state: PlasticityGateState for this synapse (creates new if None)
            
        Returns:
            Applied weight update (after gating)
        """
        # Get or create state
        if state is None:
            state = PlasticityGateState()
        
        # Update local confidence based on prediction error
        confidence = self.update_local_confidence(prediction_error, state)
        
        # Compute delta_w magnitude for fatigue update
        delta_w_mag = torch.norm(delta_w).item() if isinstance(delta_w, torch.Tensor) else abs(delta_w)
        
        # Update fatigue
        fatigue = self.update_fatigue(delta_w_mag, state)
        
        # Compute gate value
        G = self.compute_gate(
            DA_phasic=DA_phasic,
            DA_tonic=DA_tonic,
            DA_baseline=DA_baseline,
            local_confidence=confidence,
            fatigue=fatigue,
            state=state
        )
        
        # Apply gating
        if G < self.G_min * 1.1:  # Very low gate
            # Use minimal learning rate
            applied_delta = self.small_lr_scale * lr * delta_w
            gate_status = "MINIMAL"
        else:
            # Scale by gate value
            applied_delta = lr * G * delta_w
            gate_status = "GATED"
        
        # Apply update to synapse weight
        if hasattr(synapse, 'w'):
            synapse.w += applied_delta
        elif hasattr(synapse, 'weights'):
            synapse.weights += applied_delta
        else:
            # Assume synapse is the weight tensor itself
            synapse += applied_delta
        
        # Update statistics
        self.stats['gate_values'].append(G)
        self.stats['fatigue_values'].append(fatigue)
        self.stats['confidence_values'].append(confidence)
        self.stats['update_sizes'].append(torch.norm(applied_delta).item() if isinstance(applied_delta, torch.Tensor) else abs(applied_delta))
        
        # Debug output
        if self.debug and len(self.stats['gate_values']) % 50 == 0:
            print(f"[PlasticityGate] G={G:.3f}, Conf={confidence:.3f}, "
                  f"Fatigue={fatigue:.4f}, Status={gate_status}")
        
        return applied_delta
    
    def get_statistics(self, recent_window: int = 100) -> Dict:
        """Get plasticity gate statistics"""
        stats = {}
        
        if self.stats['gate_values']:
            gate_vals = self.stats['gate_values'][-recent_window:]
            stats['gate'] = {
                'mean': sum(gate_vals) / len(gate_vals),
                'min': min(gate_vals),
                'max': max(gate_vals),
                'recent': gate_vals[-5:] if len(gate_vals) >= 5 else gate_vals
            }
        
        if self.stats['confidence_values']:
            conf_vals = self.stats['confidence_values'][-recent_window:]
            stats['confidence'] = {
                'mean': sum(conf_vals) / len(conf_vals),
                'min': min(conf_vals),
                'max': max(conf_vals)
            }
        
        if self.stats['fatigue_values']:
            fatigue_vals = self.stats['fatigue_values'][-recent_window:]
            stats['fatigue'] = {
                'mean': sum(fatigue_vals) / len(fatigue_vals),
                'min': min(fatigue_vals),
                'max': max(fatigue_vals)
            }
        
        if self.stats['update_sizes']:
            update_vals = self.stats['update_sizes'][-recent_window:]
            stats['updates'] = {
                'mean': sum(update_vals) / len(update_vals),
                'total_updates': len(self.stats['update_sizes'])
            }
        
        return stats
    
    def reset_stats(self):
        """Reset statistics (keep gate states if managed externally)"""
        self.stats = {
            'gate_values': [],
            'fatigue_values': [],
            'confidence_values': [],
            'update_sizes': []
        }

# Advanced: Per-neuron plasticity gate manager
class NeuronPlasticityGateManager:
    """Manages plasticity gates for all synapses in a neuron"""
    
    def __init__(self, 
                 num_synapses: int,
                 plasticity_gate: Optional[PlasticityGate] = None,
                 share_fatigue: bool = False,  # Share fatigue across synapses?
                 share_confidence: bool = True):  # Share confidence across synapses?
        
        self.num_synapses = num_synapses
        self.share_fatigue = share_fatigue
        self.share_confidence = share_confidence
        
        # Create plasticity gate if not provided
        self.plasticity_gate = plasticity_gate or PlasticityGate()
        
        # Create gate states for each synapse
        self.synapse_states = [PlasticityGateState() for _ in range(num_synapses)]
        
        # Shared state (if sharing enabled)
        self.shared_state = PlasticityGateState() if share_fatigue or share_confidence else None
    
    def apply_gated_updates(self,
                           synapses: List,
                           delta_ws: List[torch.Tensor],
                           DA_phasic: float,
                           DA_tonic: float,
                           DA_baseline: float,
                           prediction_error: float,
                           lr: float) -> List[torch.Tensor]:
        """
        Apply gated updates to multiple synapses
        
        Returns list of applied updates
        """
        applied_updates = []
        
        for i, (synapse, delta_w) in enumerate(zip(synapses, delta_ws)):
            # Get appropriate state
            if self.share_fatigue or self.share_confidence:
                state = self.shared_state
            else:
                state = self.synapse_states[i]
            
            # Apply gated update
            applied = self.plasticity_gate.apply_gated_update(
                synapse=synapse,
                delta_w=delta_w,
                DA_phasic=DA_phasic,
                DA_tonic=DA_tonic,
                DA_baseline=DA_baseline,
                prediction_error=prediction_error,
                lr=lr,
                state=state
            )
            
            applied_updates.append(applied)
        
        return applied_updates
    
    def get_statistics(self) -> Dict:
        """Get manager statistics"""
        gate_stats = self.plasticity_gate.get_statistics()
        
        # Add synapse-specific stats
        if not self.share_fatigue and not self.share_confidence:
            confidences = [s.local_confidence for s in self.synapse_states]
            fatigues = [s.fatigue for s in self.synapse_states]
            
            gate_stats['synapse_variability'] = {
                'confidence_std': torch.std(torch.tensor(confidences)).item() if confidences else 0.0,
                'fatigue_std': torch.std(torch.tensor(fatigues)).item() if fatigues else 0.0,
                'max_confidence': max(confidences) if confidences else 0.0,
                'min_confidence': min(confidences) if confidences else 0.0
            }
        
        return gate_stats

# Test function
def test_plasticity_gates():
    """Test plasticity gate functionality"""
    print("Testing Plasticity Gates...")
    print("="*60)
    
    # Create plasticity gate
    gate = PlasticityGate(
        w1=1.0, w2=0.2, w3=0.5, w4=1.0,
        G_min=0.05, G_max=1.5,
        lambda_f=0.95,
        debug=True
    )
    
    # Test synapse class
    class TestSynapse:
        def __init__(self, w):
            self.w = torch.tensor(w, dtype=torch.float32)
    
    # Create test synapse
    synapse = TestSynapse(0.5)
    state = PlasticityGateState()
    
    print("\n1. Testing gate computation under different conditions:")
    
    test_conditions = [
        ("High DA, high conf, low fatigue", 0.8, 0.3, 0.0, 0.9, 0.1),
        ("Low DA, low conf, high fatigue", 0.1, 0.1, 0.0, 0.3, 0.5),
        ("Negative DA, medium conf, medium fatigue", -0.4, 0.2, 0.0, 0.6, 0.3),
        ("Mixed: positive DA but high fatigue", 0.6, 0.3, 0.0, 0.7, 0.8),
    ]
    
    for name, DA_phasic, DA_tonic, DA_base, confidence, fatigue in test_conditions:
        G = gate.compute_gate(
            DA_phasic=DA_phasic,
            DA_tonic=DA_tonic,
            DA_baseline=DA_base,
            local_confidence=confidence,
            fatigue=fatigue,
            state=state
        )
        
        print(f"  {name}: G={G:.3f}")
        if state.last_gate_components:
            comp = state.last_gate_components
            print(f"    Components: DA_phasic={comp['DA_phasic_component']:.3f}, "
                  f"DA_tonic={comp['DA_tonic_component']:.3f}, "
                  f"Conf={comp['confidence_component']:.3f}, "
                  f"Fatigue={comp['fatigue_component']:.3f}")
    
    print("\n2. Testing gated updates with fatigue accumulation:")
    
    # Reset state
    state = PlasticityGateState()
    
    # Simulate sequence of updates
    for i in range(10):
        # Vary dopamine and error
        DA_phasic = 0.5 if i % 3 == 0 else -0.2
        prediction_error = 0.1 + 0.05 * (i % 4)  # Varying error
        
        # Propose update
        delta_w = torch.tensor(0.01 * (1.0 + 0.2 * (i % 3)))
        
        # Apply gated update
        applied = gate.apply_gated_update(
            synapse=synapse,
            delta_w=delta_w,
            DA_phasic=DA_phasic,
            DA_tonic=0.2,
            DA_baseline=0.0,
            prediction_error=prediction_error,
            lr=0.01,
            state=state
        )
        
        if i % 2 == 0:
            print(f"  Step {i}: DA={DA_phasic:.2f}, Error={prediction_error:.3f}, "
                  f"Conf={state.local_confidence:.3f}, Fatigue={state.fatigue:.4f}, "
                  f"Applied={applied.item():.6f}")
    
    print(f"\n  Final synapse weight: {synapse.w.item():.6f}")
    print(f"  Final confidence: {state.local_confidence:.3f}")
    print(f"  Final fatigue: {state.fatigue:.4f}")
    
    print("\n3. Testing manager with multiple synapses:")
    
    # Create manager for 3 synapses
    manager = NeuronPlasticityGateManager(
        num_synapses=3,
        plasticity_gate=gate,
        share_fatigue=False,
        share_confidence=True
    )
    
    synapses = [TestSynapse(0.1 * i) for i in range(3)]
    delta_ws = [torch.tensor(0.01 * (i+1)) for i in range(3)]
    
    applied = manager.apply_gated_updates(
        synapses=synapses,
        delta_ws=delta_ws,
        DA_phasic=0.4,
        DA_tonic=0.2,
        DA_baseline=0.0,
        prediction_error=0.15,
        lr=0.01
    )
    
    print(f"  Applied updates: {[a.item() for a in applied]}")
    print(f"  Final weights: {[s.w.item() for s in synapses]}")
    
    # Get statistics
    stats = gate.get_statistics()
    print(f"\n4. Gate Statistics:")
    print(f"  Gate mean: {stats.get('gate', {}).get('mean', 0):.3f}")
    print(f"  Confidence mean: {stats.get('confidence', {}).get('mean', 0):.3f}")
    print(f"  Fatigue mean: {stats.get('fatigue', {}).get('mean', 0):.4f}")
    
    # Verify gate is working
    gate_vals = stats.get('gate', {}).get('recent', [])
    if gate_vals and min(gate_vals) > gate.G_min and max(gate_vals) < gate.G_max:
        print("✓ Gate values bounded correctly")
    else:
        print("✗ Gate bounds issue")
    
    if state.fatigue > 0:
        print("✓ Fatigue accumulated with updates")
    else:
        print("✗ Fatigue not accumulating")
    
    print("\n" + "="*60)
    print("PLASTICITY GATES TEST COMPLETE")
    print("="*60)
    
    return gate, manager

if __name__ == "__main__":
    test_plasticity_gates()