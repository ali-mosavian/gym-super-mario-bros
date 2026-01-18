from abc import ABC
from typing import Any
from typing import Dict
from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base class for all models with parameter counting functionality."""
    
    def num_parameters(self) -> Dict[str, int]:
        """Returns dictionary with total, trainable, and non-trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': train_params,
            'non_trainable': total_params - train_params
        }
    
    def parameter_summary(self) -> str:
        """Returns a formatted string with parameter statistics."""
        counts = self.num_parameters()
        summary = [
            f"Model Parameter Summary for {self.__class__.__name__}:",
            f"Total parameters:      {counts['total']:,}",
            f"Trainable parameters:  {counts['trainable']:,}",
            f"Non-trainable params:  {counts['non_trainable']:,}"
        ]
        return '\n'.join(summary)
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model."""
        pass