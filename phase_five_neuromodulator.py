# phase_five_neuromodulator.py
import torch
import math
from typing import Optional

class TorchNeuromodulator:
    """PyTorch-compatible dopamine neuromodulator for Phase 5 integration"""
    
    def __init__(self, 
                 gamma: float = 0.9, 
                 tau_tonic: float = 0.995, 
                 k: float = 2.0,
                 DA_baseline: float = 0.0, 
                 DA_max: float = 1.0,
                 device: str = 'cpu'):
        
        # State variables as tensors
        self.DA_phasic = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.DA_tonic = torch.tensor(DA_baseline, dtype=torch.float32, device=device)
        self.DA_baseline = torch.tensor(DA_baseline, dtype=torch.float32, device=device)
        self.DA_max = torch.tensor(DA_max, dtype=torch.float32, device=device)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau_tonic = tau_tonic
        self.k = k
        self.device = device
        
        # History for debugging
        self.history = {
            'DA_phasic': [],
            'DA_tonic': [],
            'drive': []
        }
    
    def update_phasic(self, drive: torch.Tensor) -> torch.Tensor:
        """
        Update phasic dopamine from reward/surprise drive signal
        
        Args:
            drive: Scalar or tensor with dopamine drive signal
            
        Returns:
            Updated DA_phasic value
        """
        # Ensure drive is a tensor
        if not isinstance(drive, torch.Tensor):
            drive = torch.tensor(drive, dtype=torch.float32, device=self.device)
        
        # Compute new value: gamma*DA_prev + (1-gamma)*tanh(k*drive)
        val = torch.tanh(self.k * drive)
        self.DA_phasic = self.gamma * self.DA_phasic + (1.0 - self.gamma) * val
        
        # Clamp to bounds
        self.DA_phasic = torch.clamp(self.DA_phasic, -self.DA_max, self.DA_max)
        
        # Record history
        self.history['DA_phasic'].append(self.DA_phasic.item())
        self.history['drive'].append(drive.item() if drive.numel() == 1 else drive.mean().item())
        
        return self.DA_phasic
    
    def update_tonic(self) -> torch.Tensor:
        """Update tonic dopamine (slow moving average)"""
        self.DA_tonic = self.tau_tonic * self.DA_tonic + (1.0 - self.tau_tonic) * self.DA_phasic
        self.DA_tonic = torch.clamp(self.DA_tonic, -self.DA_max, self.DA_max)
        
        # Record history
        self.history['DA_tonic'].append(self.DA_tonic.item())
        
        return self.DA_tonic
    
    def compute_drive(self, 
                     reward: torch.Tensor, 
                     surprise: Optional[torch.Tensor] = None,
                     error_change: Optional[torch.Tensor] = None,
                     alpha: float = 1.0,
                     beta: float = 0.5) -> torch.Tensor:
        """
        Compute dopamine drive signal from reward and surprise
        
        Args:
            reward: External or total reward signal
            surprise: Prediction error magnitude
            error_change: (error_prev - error_now) for learning progress
            alpha: Reward weight
            beta: Surprise/learning progress weight
            
        Returns:
            Dopamine drive signal
        """
        drive = alpha * reward
        
        if surprise is not None:
            # Positive surprise with positive reward = extra dopamine
            # Negative surprise with negative reward = extra negative dopamine
            drive = drive + beta * surprise * torch.sign(reward)
        
        if error_change is not None:
            # Learning progress bonus: positive when error decreases
            drive = drive + beta * error_change
        
        return drive
    
    def exploration_scale(self, k_e: float = 1.0) -> torch.Tensor:
        """
        Compute exploration scaling factor based on tonic dopamine
        
        Higher tonic -> potentially more exploration (or less, tune sign)
        """
        # Sigmoid: 0.5 when tonic=baseline, approaches 1 when tonic high
        scale = 1.0 / (1.0 + torch.exp(-k_e * (self.DA_tonic - self.DA_baseline)))
        return scale
    
    def reset(self, keep_history: bool = False):
        """Reset neuromodulator state"""
        self.DA_phasic = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.DA_tonic = self.DA_baseline.clone()
        
        if not keep_history:
            self.history = {'DA_phasic': [], 'DA_tonic': [], 'drive': []}
    
    def get_stats(self) -> dict:
        """Get statistics about dopamine signals"""
        if not self.history['DA_phasic']:
            return {}
        
        return {
            'DA_phasic_mean': torch.tensor(self.history['DA_phasic']).mean().item(),
            'DA_phasic_std': torch.tensor(self.history['DA_phasic']).std().item(),
            'DA_tonic_mean': torch.tensor(self.history['DA_tonic']).mean().item(),
            'DA_tonic_std': torch.tensor(self.history['DA_tonic']).std().item(),
            'recent_drive': self.history['drive'][-5:] if len(self.history['drive']) >= 5 else self.history['drive']
        }