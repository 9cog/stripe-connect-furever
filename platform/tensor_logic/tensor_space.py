"""
Tensor Space Implementation

Core tensor space for gradient-based learning in Tensor Logic.
Provides differentiable truth values and attention values that can be
optimized through backpropagation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import math


@dataclass
class TensorAtomValue:
    """
    Tensor-valued representation of an atom's truth value and attention.
    
    Unlike traditional truth values (discrete), tensor values support
    gradient-based optimization and can represent continuous confidence.
    """
    # Strength: probability or degree of truth (0.0 to 1.0)
    strength: float = 0.5
    
    # Confidence: reliability of the strength value (0.0 to 1.0)
    confidence: float = 0.5
    
    # Short-term importance (STI): immediate attention weight
    sti: float = 0.0
    
    # Long-term importance (LTI): sustained relevance
    lti: float = 0.0
    
    # Gradient information for backpropagation
    grad_strength: float = 0.0
    grad_confidence: float = 0.0
    grad_sti: float = 0.0
    grad_lti: float = 0.0
    
    # Learning metadata
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0
    
    def __post_init__(self):
        """Clamp values to valid ranges."""
        self.strength = self._clamp(self.strength)
        self.confidence = self._clamp(self.confidence)
        self.sti = self._clamp_attention(self.sti)
        self.lti = self._clamp_attention(self.lti)
    
    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to range [min_val, max_val]."""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def _clamp_attention(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Clamp attention value to range [-1.0, 1.0]."""
        return max(min_val, min(max_val, value))
    
    def update_with_gradient(self, learning_rate: float = 0.01):
        """
        Update tensor values using gradient descent.
        
        Args:
            learning_rate: Step size for gradient descent
        """
        self.strength = self._clamp(
            self.strength - learning_rate * self.grad_strength
        )
        self.confidence = self._clamp(
            self.confidence - learning_rate * self.grad_confidence
        )
        self.sti = self._clamp_attention(
            self.sti - learning_rate * self.grad_sti
        )
        self.lti = self._clamp_attention(
            self.lti - learning_rate * self.grad_lti
        )
        
        self.last_updated = datetime.now()
        self.update_count += 1
        
        # Reset gradients after update
        self.reset_gradients()
    
    def reset_gradients(self):
        """Reset all gradients to zero."""
        self.grad_strength = 0.0
        self.grad_confidence = 0.0
        self.grad_sti = 0.0
        self.grad_lti = 0.0
    
    def accumulate_gradient(
        self,
        grad_strength: float = 0.0,
        grad_confidence: float = 0.0,
        grad_sti: float = 0.0,
        grad_lti: float = 0.0
    ):
        """Accumulate gradients for batch learning."""
        self.grad_strength += grad_strength
        self.grad_confidence += grad_confidence
        self.grad_sti += grad_sti
        self.grad_lti += grad_lti
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to traditional (strength, confidence) tuple."""
        return (self.strength, self.confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'strength': self.strength,
            'confidence': self.confidence,
            'sti': self.sti,
            'lti': self.lti,
            'last_updated': self.last_updated.isoformat(),
            'update_count': self.update_count
        }


class TensorSpace:
    """
    Tensor space for gradient-based learning in knowledge graphs.
    
    Provides:
    - Differentiable truth values and attention
    - Gradient computation and backpropagation
    - Batch learning and optimization
    - Loss function computation
    - Neural-symbolic integration
    """
    
    def __init__(
        self,
        name: str = "tensor_space",
        learning_rate: float = 0.01,
        momentum: float = 0.9
    ):
        self.name = name
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Storage for tensor values
        self.tensor_values: Dict[str, TensorAtomValue] = {}
        
        # Momentum for optimization
        self.velocity: Dict[str, Dict[str, float]] = {}
        
        # Training metadata
        self.epoch: int = 0
        self.total_loss: float = 0.0
        self.loss_history: List[float] = []
        
        self.created_at = datetime.now()
    
    def register_atom(
        self,
        atom_id: str,
        initial_value: Optional[TensorAtomValue] = None
    ) -> TensorAtomValue:
        """
        Register an atom in the tensor space.
        
        Args:
            atom_id: Unique identifier for the atom
            initial_value: Initial tensor value (optional)
        
        Returns:
            TensorAtomValue for the atom
        """
        if atom_id not in self.tensor_values:
            self.tensor_values[atom_id] = initial_value or TensorAtomValue()
            self.velocity[atom_id] = {
                'strength': 0.0,
                'confidence': 0.0,
                'sti': 0.0,
                'lti': 0.0
            }
        return self.tensor_values[atom_id]
    
    def get_value(self, atom_id: str) -> Optional[TensorAtomValue]:
        """Get tensor value for an atom."""
        return self.tensor_values.get(atom_id)
    
    def update_value(
        self,
        atom_id: str,
        strength: Optional[float] = None,
        confidence: Optional[float] = None,
        sti: Optional[float] = None,
        lti: Optional[float] = None
    ):
        """
        Update tensor value for an atom.
        
        Args:
            atom_id: Atom identifier
            strength: New strength value (optional)
            confidence: New confidence value (optional)
            sti: New STI value (optional)
            lti: New LTI value (optional)
        """
        if atom_id not in self.tensor_values:
            self.register_atom(atom_id)
        
        value = self.tensor_values[atom_id]
        if strength is not None:
            value.strength = TensorAtomValue._clamp(strength)
        if confidence is not None:
            value.confidence = TensorAtomValue._clamp(confidence)
        if sti is not None:
            value.sti = TensorAtomValue._clamp_attention(sti)
        if lti is not None:
            value.lti = TensorAtomValue._clamp_attention(lti)
        
        value.last_updated = datetime.now()
        value.update_count += 1
    
    def compute_loss(
        self,
        atom_id: str,
        target_strength: float,
        target_confidence: float,
        loss_type: str = "mse"
    ) -> float:
        """
        Compute loss for an atom's tensor value.
        
        Args:
            atom_id: Atom identifier
            target_strength: Target strength value
            target_confidence: Target confidence value
            loss_type: Type of loss function ("mse", "cross_entropy")
        
        Returns:
            Loss value
        """
        value = self.get_value(atom_id)
        if not value:
            return 0.0
        
        if loss_type == "mse":
            # Mean squared error
            loss_s = (value.strength - target_strength) ** 2
            loss_c = (value.confidence - target_confidence) ** 2
            return (loss_s + loss_c) / 2.0
        
        elif loss_type == "cross_entropy":
            # Binary cross-entropy
            eps = 1e-7  # Small epsilon to avoid log(0)
            loss_s = -(
                target_strength * math.log(value.strength + eps) +
                (1 - target_strength) * math.log(1 - value.strength + eps)
            )
            loss_c = -(
                target_confidence * math.log(value.confidence + eps) +
                (1 - target_confidence) * math.log(1 - value.confidence + eps)
            )
            return (loss_s + loss_c) / 2.0
        
        return 0.0
    
    def compute_gradient(
        self,
        atom_id: str,
        target_strength: float,
        target_confidence: float,
        loss_type: str = "mse"
    ):
        """
        Compute and accumulate gradients for an atom.
        
        Args:
            atom_id: Atom identifier
            target_strength: Target strength value
            target_confidence: Target confidence value
            loss_type: Type of loss function
        """
        value = self.get_value(atom_id)
        if not value:
            return
        
        if loss_type == "mse":
            # Gradient of MSE
            grad_s = 2 * (value.strength - target_strength)
            grad_c = 2 * (value.confidence - target_confidence)
        
        elif loss_type == "cross_entropy":
            # Gradient of cross-entropy
            eps = 1e-7
            grad_s = (
                -target_strength / (value.strength + eps) +
                (1 - target_strength) / (1 - value.strength + eps)
            )
            grad_c = (
                -target_confidence / (value.confidence + eps) +
                (1 - target_confidence) / (1 - value.confidence + eps)
            )
        else:
            grad_s = 0.0
            grad_c = 0.0
        
        value.accumulate_gradient(grad_strength=grad_s, grad_confidence=grad_c)
    
    def step(self, use_momentum: bool = True):
        """
        Perform one optimization step for all atoms.
        
        Args:
            use_momentum: Whether to use momentum for optimization
        """
        for atom_id, value in self.tensor_values.items():
            if use_momentum:
                # Update velocity with momentum
                vel = self.velocity[atom_id]
                vel['strength'] = (
                    self.momentum * vel['strength'] +
                    self.learning_rate * value.grad_strength
                )
                vel['confidence'] = (
                    self.momentum * vel['confidence'] +
                    self.learning_rate * value.grad_confidence
                )
                vel['sti'] = (
                    self.momentum * vel['sti'] +
                    self.learning_rate * value.grad_sti
                )
                vel['lti'] = (
                    self.momentum * vel['lti'] +
                    self.learning_rate * value.grad_lti
                )
                
                # Update values using velocity
                value.strength = TensorAtomValue._clamp(
                    value.strength - vel['strength']
                )
                value.confidence = TensorAtomValue._clamp(
                    value.confidence - vel['confidence']
                )
                value.sti = TensorAtomValue._clamp_attention(
                    value.sti - vel['sti']
                )
                value.lti = TensorAtomValue._clamp_attention(
                    value.lti - vel['lti']
                )
            else:
                # Simple gradient descent
                value.update_with_gradient(self.learning_rate)
            
            value.reset_gradients()
    
    def train_batch(
        self,
        training_data: List[Tuple[str, float, float]],
        loss_type: str = "mse",
        use_momentum: bool = True
    ) -> float:
        """
        Train on a batch of examples.
        
        Args:
            training_data: List of (atom_id, target_strength, target_confidence)
            loss_type: Type of loss function
            use_momentum: Whether to use momentum
        
        Returns:
            Average batch loss
        """
        total_loss = 0.0
        
        # Compute gradients for batch
        for atom_id, target_strength, target_confidence in training_data:
            self.compute_gradient(
                atom_id,
                target_strength,
                target_confidence,
                loss_type
            )
            loss = self.compute_loss(
                atom_id,
                target_strength,
                target_confidence,
                loss_type
            )
            total_loss += loss
        
        # Update all values
        self.step(use_momentum)
        
        # Track statistics
        avg_loss = total_loss / len(training_data) if training_data else 0.0
        self.loss_history.append(avg_loss)
        self.epoch += 1
        
        return avg_loss
    
    def zero_grad(self):
        """Reset all gradients to zero."""
        for value in self.tensor_values.values():
            value.reset_gradients()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the tensor space."""
        if not self.tensor_values:
            return {
                'name': self.name,
                'total_atoms': 0,
                'epoch': self.epoch
            }
        
        strengths = [v.strength for v in self.tensor_values.values()]
        confidences = [v.confidence for v in self.tensor_values.values()]
        
        return {
            'name': self.name,
            'total_atoms': len(self.tensor_values),
            'epoch': self.epoch,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'avg_strength': sum(strengths) / len(strengths),
            'avg_confidence': sum(confidences) / len(confidences),
            'recent_loss': self.loss_history[-1] if self.loss_history else 0.0,
            'created_at': self.created_at.isoformat()
        }
    
    def export_values(self) -> Dict[str, Dict[str, Any]]:
        """Export all tensor values."""
        return {
            atom_id: value.to_dict()
            for atom_id, value in self.tensor_values.items()
        }
