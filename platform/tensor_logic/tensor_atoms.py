"""
Tensor Atoms

Extension of OpenCog Atomspace with tensor-valued atoms that support
gradient-based learning and neural-symbolic reasoning.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from opencog.atomspace import (
    StripeAtomspace,
    AtomType,
    Atom,
    Node as BaseNode,
    Link as BaseLink
)
from .tensor_space import TensorSpace, TensorAtomValue


@dataclass
class TensorAtom(Atom):
    """
    Base class for atoms with tensor values.
    
    Extends traditional atoms with:
    - Gradient-based truth values
    - Learnable attention values
    - Neural network integration
    - Backpropagation support
    """
    tensor_space: Optional[TensorSpace] = None
    tensor_id: Optional[str] = None
    
    def __post_init__(self):
        """Register atom in tensor space."""
        if self.tensor_space and self.tensor_id is None:
            self.tensor_id = self.id
            self.tensor_space.register_atom(self.tensor_id)
    
    def get_tensor_value(self) -> Optional[TensorAtomValue]:
        """Get the tensor value for this atom."""
        if self.tensor_space and self.tensor_id:
            return self.tensor_space.get_value(self.tensor_id)
        return None
    
    def set_tensor_value(
        self,
        strength: Optional[float] = None,
        confidence: Optional[float] = None,
        sti: Optional[float] = None,
        lti: Optional[float] = None
    ):
        """Set the tensor value for this atom."""
        if self.tensor_space and self.tensor_id:
            self.tensor_space.update_value(
                self.tensor_id,
                strength=strength,
                confidence=confidence,
                sti=sti,
                lti=lti
            )
    
    def compute_loss(
        self,
        target_strength: float,
        target_confidence: float,
        loss_type: str = "mse"
    ) -> float:
        """Compute loss for this atom's tensor value."""
        if self.tensor_space and self.tensor_id:
            return self.tensor_space.compute_loss(
                self.tensor_id,
                target_strength,
                target_confidence,
                loss_type
            )
        return 0.0
    
    def compute_gradient(
        self,
        target_strength: float,
        target_confidence: float,
        loss_type: str = "mse"
    ):
        """Compute and accumulate gradients for this atom."""
        if self.tensor_space and self.tensor_id:
            self.tensor_space.compute_gradient(
                self.tensor_id,
                target_strength,
                target_confidence,
                loss_type
            )
    
    def update_with_gradient(self, learning_rate: float = 0.01):
        """Update tensor value using accumulated gradients."""
        tensor_value = self.get_tensor_value()
        if tensor_value:
            tensor_value.update_with_gradient(learning_rate)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including tensor values."""
        base_dict = super().to_dict()
        
        tensor_value = self.get_tensor_value()
        if tensor_value:
            base_dict['tensor_value'] = tensor_value.to_dict()
        
        return base_dict


@dataclass
class TensorNode(TensorAtom, BaseNode):
    """
    A node with tensor values for gradient-based learning.
    
    Represents entities or concepts with learnable truth values
    and attention weights.
    """
    pass


@dataclass
class TensorLink(TensorAtom, BaseLink):
    """
    A link with tensor values for gradient-based learning.
    
    Represents relationships with learnable truth values
    and attention weights.
    """
    
    def propagate_gradient(self, learning_rate: float = 0.01):
        """
        Propagate gradients to outgoing atoms.
        
        Implements backpropagation through the graph structure
        by distributing gradients to connected atoms.
        """
        my_tensor_value = self.get_tensor_value()
        if not my_tensor_value:
            return
        
        # Distribute gradient to outgoing atoms
        num_outgoing = len(self.outgoing)
        if num_outgoing == 0:
            return
        
        for atom in self.outgoing:
            if isinstance(atom, TensorAtom):
                # Propagate a fraction of the gradient
                atom_tensor = atom.get_tensor_value()
                if atom_tensor:
                    # Simple gradient distribution (can be made more sophisticated)
                    atom_tensor.accumulate_gradient(
                        grad_strength=my_tensor_value.grad_strength / num_outgoing,
                        grad_confidence=my_tensor_value.grad_confidence / num_outgoing,
                        grad_sti=my_tensor_value.grad_sti / num_outgoing,
                        grad_lti=my_tensor_value.grad_lti / num_outgoing
                    )


class TensorAtomspace(StripeAtomspace):
    """
    Extended Atomspace with tensor logic support.
    
    Provides:
    - Tensor-valued atoms
    - Gradient-based learning
    - Neural-symbolic integration
    - Batch training
    """
    
    def __init__(self, name: str = "tensor_atomspace", learning_rate: float = 0.01):
        super().__init__(name)
        self.tensor_space = TensorSpace(
            name=f"{name}_tensor_space",
            learning_rate=learning_rate
        )
    
    def add_tensor_node(
        self,
        atom_type: AtomType,
        name: str,
        value: Any = None,
        strength: float = 0.5,
        confidence: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> TensorNode:
        """Add a node with tensor values."""
        # Check if node exists
        existing = self.get_node(atom_type, name)
        if existing:
            # Convert to TensorNode if needed
            if isinstance(existing, TensorNode):
                return existing
            # Create new TensorNode wrapping existing
            tensor_node = TensorNode(
                atom_type=atom_type,
                name=name,
                value=value,
                truth_value=(strength, confidence),
                metadata=metadata or {},
                tensor_space=self.tensor_space
            )
            # Copy ID from existing
            tensor_node.id = existing.id
            tensor_node.tensor_id = existing.id
            # Update in storage
            self.atoms[tensor_node.id] = tensor_node
            return tensor_node
        
        # Create new TensorNode
        tensor_node = TensorNode(
            atom_type=atom_type,
            name=name,
            value=value,
            truth_value=(strength, confidence),
            metadata=metadata or {},
            tensor_space=self.tensor_space
        )
        
        self._add_atom(tensor_node)
        
        # Initialize tensor value
        tensor_node.set_tensor_value(strength=strength, confidence=confidence)
        
        return tensor_node
    
    def add_tensor_link(
        self,
        atom_type: AtomType,
        outgoing: List[Atom],
        name: str = None,
        strength: float = 0.5,
        confidence: float = 0.5,
        metadata: Dict[str, Any] = None
    ) -> TensorLink:
        """Add a link with tensor values."""
        tensor_link = TensorLink(
            atom_type=atom_type,
            name=name or "",
            outgoing=outgoing,
            truth_value=(strength, confidence),
            metadata=metadata or {},
            tensor_space=self.tensor_space
        )
        
        self._add_atom(tensor_link)
        
        # Update incoming index
        for atom in outgoing:
            if atom.id not in self.incoming_index:
                self.incoming_index[atom.id] = set()
            self.incoming_index[atom.id].add(tensor_link.id)
        
        # Initialize tensor value
        tensor_link.set_tensor_value(strength=strength, confidence=confidence)
        
        return tensor_link
    
    def train_atoms(
        self,
        training_data: List[Tuple[str, float, float]],
        epochs: int = 100,
        batch_size: int = 32,
        loss_type: str = "mse"
    ) -> List[float]:
        """
        Train all tensor atoms on examples.
        
        Args:
            training_data: List of (atom_id, target_strength, target_confidence)
            epochs: Number of training epochs
            batch_size: Batch size for training
            loss_type: Type of loss function
        
        Returns:
            List of loss values per epoch
        """
        loss_history = []
        
        for epoch in range(epochs):
            # Shuffle training data
            import random
            shuffled_data = training_data.copy()
            random.shuffle(shuffled_data)
            
            # Train in batches
            epoch_losses = []
            for i in range(0, len(shuffled_data), batch_size):
                batch = shuffled_data[i:i+batch_size]
                batch_loss = self.tensor_space.train_batch(
                    batch,
                    loss_type=loss_type,
                    use_momentum=True
                )
                epoch_losses.append(batch_loss)
            
            # Average loss for epoch
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            loss_history.append(avg_loss)
        
        return loss_history
    
    def backpropagate(self, root_atoms: List[TensorAtom], learning_rate: float = 0.01):
        """
        Perform backpropagation through the graph.
        
        Args:
            root_atoms: Starting atoms for backpropagation
            learning_rate: Learning rate for updates
        """
        # Propagate gradients backward through links
        visited = set()
        queue = list(root_atoms)
        
        while queue:
            atom = queue.pop(0)
            if atom.id in visited:
                continue
            visited.add(atom.id)
            
            # Update this atom
            if isinstance(atom, TensorAtom):
                atom.update_with_gradient(learning_rate)
            
            # Propagate through links
            if isinstance(atom, TensorLink):
                atom.propagate_gradient(learning_rate)
                # Add outgoing atoms to queue
                for outgoing_atom in atom.outgoing:
                    if isinstance(outgoing_atom, TensorAtom):
                        queue.append(outgoing_atom)
    
    def get_tensor_statistics(self) -> Dict[str, Any]:
        """Get statistics about tensor values."""
        base_stats = self.get_statistics()
        tensor_stats = self.tensor_space.get_statistics()
        
        # Count tensor atoms
        tensor_nodes = sum(
            1 for atom in self.atoms.values()
            if isinstance(atom, TensorNode)
        )
        tensor_links = sum(
            1 for atom in self.atoms.values()
            if isinstance(atom, TensorLink)
        )
        
        return {
            **base_stats,
            'tensor_space': tensor_stats,
            'tensor_nodes': tensor_nodes,
            'tensor_links': tensor_links
        }
    
    def export_tensor_values(self) -> Dict[str, Any]:
        """Export all tensor values."""
        return {
            'atomspace_name': self.name,
            'tensor_values': self.tensor_space.export_values(),
            'statistics': self.get_tensor_statistics()
        }
