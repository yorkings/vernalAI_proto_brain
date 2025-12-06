import torch
import torch.nn as nn
from phase_two_column import ProtoColumn
from typing import List, Callable, Dict

class GlobalWorkspace(nn.Module):
    def __init__(self, hidden_dim: int, num_columns: int, threshold: float = 0.6):
        super().__init__()
        self.threshold = threshold
        self.hidden_dim = hidden_dim
        self.num_columns = num_columns
        
        # learns how to combine signals from columns
        self.fusion = nn.Linear(hidden_dim * num_columns, hidden_dim)

    def forward(self, column_outputs: List[torch.Tensor], column_traces: List[float]) -> Dict:
        """
        Process column outputs and check for global ignition.
        
        Args:
            column_outputs: List of column output tensors (each shape [hidden_dim])
            column_traces: List of column trace values
            
        Returns:
            Dict with ignition results
        """
        # Stack column outputs
        latents = torch.stack(column_outputs, dim=0)  # (C, H)
        
        # Calculate metrics
        salience = torch.tensor([torch.norm(output).item() for output in column_outputs])  # (C,)
        reward = torch.tensor(column_traces)  # (C,)
        
        # For now, use placeholder for surprise (0.0)
        surprise = torch.zeros_like(salience)
        
        # Add batch dimension for compatibility
        latents = latents.unsqueeze(0)  # (1, C, H)
        salience = salience.unsqueeze(0)  # (1, C)
        surprise = surprise.unsqueeze(0)  # (1, C)
        reward = reward.unsqueeze(0)  # (1, C)

        # 1. Compute importance score per column
        importance = 0.5 * salience + 0.3 * surprise + 0.2 * reward  # (1, C)

        # 2. Determine which columns participate in ignition
        ignition_mask = (importance > self.threshold).float()  # (1, C)

        if ignition_mask.sum() == 0:
            # No ignition → return "no global event"
            return {
                "ignited": False,
                "global_signal": torch.zeros(1),
                "broadcast_vector": torch.zeros(self.hidden_dim),
                "sync_mask": ignition_mask.squeeze(0)
            }

        # 3. Weighted mean of latent vectors → global broadcast vector
        weights = ignition_mask.unsqueeze(-1)  # (1, C, 1)
        broadcast = (latents * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)  # (1, H)

        # 4. Alternative: use all latents concatenated
        flat_latents = latents.flatten(start_dim=1)  # (1, C*H)
        broadcast = torch.tanh(self.fusion(flat_latents))  # (1, H)

        # 5. Global signal magnitude
        global_signal = importance.max(dim=1).values  # strongest activated column

        return {
            "ignited": True,
            "global_signal": global_signal.squeeze(0),      # scalar
            "broadcast_vector": broadcast.squeeze(0),       # (H,)
            "sync_mask": ignition_mask.squeeze(0)           # (C,)
        }
