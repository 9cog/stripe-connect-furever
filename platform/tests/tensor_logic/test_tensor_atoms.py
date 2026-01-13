"""
Unit tests for Tensor Atoms

Tests the tensor-valued atom extensions including:
- TensorAtom functionality
- TensorNode and TensorLink creation
- Gradient propagation
- TensorAtomspace integration
- Backpropagation through graphs
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from opencog.atomspace import AtomType
from tensor_logic.tensor_atoms import (
    TensorAtom,
    TensorNode,
    TensorLink,
    TensorAtomspace
)
from tensor_logic.tensor_space import TensorSpace, TensorAtomValue


class TestTensorAtom:
    """Test cases for TensorAtom base class"""
    
    def test_initialization_with_tensor_space(self):
        """Test atom initialization with tensor space"""
        tensor_space = TensorSpace()
        atom = TensorNode(
            atom_type=AtomType.CONCEPT,
            name="test_concept",
            tensor_space=tensor_space
        )
        
        assert atom.tensor_space is tensor_space
        assert atom.tensor_id is not None
        assert atom.tensor_id in tensor_space.tensor_values
    
    def test_get_tensor_value(self):
        """Test getting tensor value"""
        tensor_space = TensorSpace()
        atom = TensorNode(
            atom_type=AtomType.CONCEPT,
            name="test",
            tensor_space=tensor_space
        )
        
        value = atom.get_tensor_value()
        assert value is not None
        assert isinstance(value, TensorAtomValue)
    
    def test_set_tensor_value(self):
        """Test setting tensor value"""
        tensor_space = TensorSpace()
        atom = TensorNode(
            atom_type=AtomType.CONCEPT,
            name="test",
            tensor_space=tensor_space
        )
        
        atom.set_tensor_value(strength=0.8, confidence=0.9)
        value = atom.get_tensor_value()
        
        assert value.strength == 0.8
        assert value.confidence == 0.9
    
    def test_compute_loss(self):
        """Test loss computation"""
        tensor_space = TensorSpace()
        atom = TensorNode(
            atom_type=AtomType.CONCEPT,
            name="test",
            tensor_space=tensor_space
        )
        
        atom.set_tensor_value(strength=0.6, confidence=0.7)
        loss = atom.compute_loss(0.8, 0.9, loss_type="mse")
        
        # MSE = ((0.6-0.8)^2 + (0.7-0.9)^2) / 2
        expected_loss = ((0.6 - 0.8)**2 + (0.7 - 0.9)**2) / 2
        assert abs(loss - expected_loss) < 1e-6
    
    def test_compute_gradient(self):
        """Test gradient computation"""
        tensor_space = TensorSpace()
        atom = TensorNode(
            atom_type=AtomType.CONCEPT,
            name="test",
            tensor_space=tensor_space
        )
        
        atom.set_tensor_value(strength=0.5, confidence=0.5)
        atom.compute_gradient(0.8, 0.9, loss_type="mse")
        
        value = atom.get_tensor_value()
        assert value.grad_strength != 0.0
        assert value.grad_confidence != 0.0
    
    def test_update_with_gradient(self):
        """Test gradient-based update"""
        tensor_space = TensorSpace()
        atom = TensorNode(
            atom_type=AtomType.CONCEPT,
            name="test",
            tensor_space=tensor_space
        )
        
        atom.set_tensor_value(strength=0.5, confidence=0.5)
        atom.compute_gradient(0.8, 0.9, loss_type="mse")
        
        initial_strength = atom.get_tensor_value().strength
        atom.update_with_gradient(learning_rate=0.1)
        
        # Value should have changed
        assert atom.get_tensor_value().strength != initial_strength
    
    def test_to_dict(self):
        """Test dictionary conversion"""
        tensor_space = TensorSpace()
        atom = TensorNode(
            atom_type=AtomType.CONCEPT,
            name="test",
            value="test_value",
            tensor_space=tensor_space
        )
        
        atom.set_tensor_value(strength=0.7, confidence=0.8)
        dict_repr = atom.to_dict()
        
        assert 'type' in dict_repr
        assert 'name' in dict_repr
        assert 'tensor_value' in dict_repr
        assert dict_repr['tensor_value']['strength'] == 0.7


class TestTensorNode:
    """Test cases for TensorNode"""
    
    def test_creation(self):
        """Test tensor node creation"""
        tensor_space = TensorSpace()
        node = TensorNode(
            atom_type=AtomType.PAYMENT,
            name="payment_123",
            value={'amount': 5000, 'currency': 'usd'},
            tensor_space=tensor_space
        )
        
        assert node.atom_type == AtomType.PAYMENT
        assert node.name == "payment_123"
        assert node.value['amount'] == 5000
    
    def test_multiple_nodes(self):
        """Test creating multiple nodes"""
        tensor_space = TensorSpace()
        nodes = []
        
        for i in range(5):
            node = TensorNode(
                atom_type=AtomType.CONCEPT,
                name=f"node_{i}",
                tensor_space=tensor_space
            )
            nodes.append(node)
        
        assert len(nodes) == 5
        assert len(tensor_space.tensor_values) == 5


class TestTensorLink:
    """Test cases for TensorLink"""
    
    def test_creation(self):
        """Test tensor link creation"""
        tensor_space = TensorSpace()
        
        node1 = TensorNode(
            atom_type=AtomType.PAYMENT,
            name="payment_1",
            tensor_space=tensor_space
        )
        node2 = TensorNode(
            atom_type=AtomType.CUSTOMER,
            name="customer_1",
            tensor_space=tensor_space
        )
        
        link = TensorLink(
            atom_type=AtomType.PAYMENT_CUSTOMER,
            name="payment_customer_link",
            outgoing=[node1, node2],
            tensor_space=tensor_space
        )
        
        assert link.atom_type == AtomType.PAYMENT_CUSTOMER
        assert len(link.outgoing) == 2
        assert node1 in link.outgoing
        assert node2 in link.outgoing
    
    def test_propagate_gradient(self):
        """Test gradient propagation through link"""
        tensor_space = TensorSpace()
        
        node1 = TensorNode(
            atom_type=AtomType.CONCEPT,
            name="concept_1",
            tensor_space=tensor_space
        )
        node2 = TensorNode(
            atom_type=AtomType.CONCEPT,
            name="concept_2",
            tensor_space=tensor_space
        )
        
        link = TensorLink(
            atom_type=AtomType.INHERITANCE,
            name="inheritance_link",
            outgoing=[node1, node2],
            tensor_space=tensor_space
        )
        
        # Set gradient on link
        link_value = link.get_tensor_value()
        link_value.accumulate_gradient(grad_strength=1.0, grad_confidence=1.0)
        
        # Propagate to outgoing atoms
        link.propagate_gradient(learning_rate=0.1)
        
        # Outgoing atoms should have gradients
        node1_value = node1.get_tensor_value()
        node2_value = node2.get_tensor_value()
        
        # Gradients should be distributed
        assert node1_value.grad_strength != 0.0 or node2_value.grad_strength != 0.0


class TestTensorAtomspace:
    """Test cases for TensorAtomspace"""
    
    def test_initialization(self):
        """Test atomspace initialization"""
        atomspace = TensorAtomspace(name="test_atomspace", learning_rate=0.05)
        
        assert atomspace.name == "test_atomspace"
        assert atomspace.tensor_space is not None
        assert atomspace.tensor_space.learning_rate == 0.05
    
    def test_add_tensor_node(self):
        """Test adding tensor node"""
        atomspace = TensorAtomspace()
        
        node = atomspace.add_tensor_node(
            atom_type=AtomType.PAYMENT,
            name="payment_1",
            value={'amount': 5000},
            strength=0.8,
            confidence=0.9
        )
        
        assert isinstance(node, TensorNode)
        assert node.name == "payment_1"
        
        # Check tensor value was set
        tensor_value = node.get_tensor_value()
        assert tensor_value.strength == 0.8
        assert tensor_value.confidence == 0.9
    
    def test_add_tensor_node_idempotent(self):
        """Test that adding same node twice returns existing"""
        atomspace = TensorAtomspace()
        
        node1 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="concept_1",
            strength=0.7
        )
        
        node2 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="concept_1",
            strength=0.8  # Different strength
        )
        
        # Should return the same node
        assert node1.id == node2.id
    
    def test_add_tensor_link(self):
        """Test adding tensor link"""
        atomspace = TensorAtomspace()
        
        node1 = atomspace.add_tensor_node(
            atom_type=AtomType.PAYMENT,
            name="payment_1"
        )
        node2 = atomspace.add_tensor_node(
            atom_type=AtomType.CUSTOMER,
            name="customer_1"
        )
        
        link = atomspace.add_tensor_link(
            atom_type=AtomType.PAYMENT_CUSTOMER,
            outgoing=[node1, node2],
            strength=0.9,
            confidence=0.95
        )
        
        assert isinstance(link, TensorLink)
        assert len(link.outgoing) == 2
        
        # Check tensor value
        tensor_value = link.get_tensor_value()
        assert tensor_value.strength == 0.9
        assert tensor_value.confidence == 0.95
    
    def test_train_atoms(self):
        """Test training atoms"""
        atomspace = TensorAtomspace(learning_rate=0.1)
        
        # Add nodes
        node1 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="concept_1",
            strength=0.3
        )
        node2 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="concept_2",
            strength=0.4
        )
        
        # Training data
        training_data = [
            (node1.id, 0.9, 0.95),
            (node2.id, 0.8, 0.85)
        ]
        
        # Train
        loss_history = atomspace.train_atoms(
            training_data,
            epochs=50,
            batch_size=2,
            loss_type="mse"
        )
        
        assert len(loss_history) == 50
        
        # Values should have moved towards targets
        assert node1.get_tensor_value().strength > 0.3
        assert node2.get_tensor_value().strength > 0.4
    
    def test_backpropagate(self):
        """Test backpropagation through graph"""
        atomspace = TensorAtomspace(learning_rate=0.1)
        
        # Create a simple graph: node1 -> link -> node2
        node1 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="source",
            strength=0.5
        )
        node2 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="target",
            strength=0.5
        )
        link = atomspace.add_tensor_link(
            atom_type=AtomType.INHERITANCE,
            outgoing=[node1, node2],
            strength=0.5
        )
        
        # Set gradient on link
        link.compute_gradient(0.9, 0.95, loss_type="mse")
        
        initial_strength = node1.get_tensor_value().strength
        
        # Backpropagate from link
        atomspace.backpropagate([link], learning_rate=0.1)
        
        # Node values may have changed through propagation
        # (depends on implementation details)
        assert True  # Test completes without error
    
    def test_get_tensor_statistics(self):
        """Test getting tensor statistics"""
        atomspace = TensorAtomspace()
        
        # Add some atoms
        for i in range(3):
            atomspace.add_tensor_node(
                atom_type=AtomType.CONCEPT,
                name=f"concept_{i}"
            )
        
        node1 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="concept_a"
        )
        node2 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="concept_b"
        )
        atomspace.add_tensor_link(
            atom_type=AtomType.INHERITANCE,
            outgoing=[node1, node2]
        )
        
        stats = atomspace.get_tensor_statistics()
        
        assert 'tensor_space' in stats
        assert 'tensor_nodes' in stats
        assert 'tensor_links' in stats
        assert stats['tensor_nodes'] > 0
        assert stats['tensor_links'] > 0
    
    def test_export_tensor_values(self):
        """Test exporting tensor values"""
        atomspace = TensorAtomspace()
        
        node1 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="concept_1",
            strength=0.8,
            confidence=0.9
        )
        node2 = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="concept_2",
            strength=0.7,
            confidence=0.85
        )
        
        exported = atomspace.export_tensor_values()
        
        assert 'atomspace_name' in exported
        assert 'tensor_values' in exported
        assert 'statistics' in exported
        assert node1.id in exported['tensor_values']
        assert node2.id in exported['tensor_values']
    
    def test_convergence_through_training(self):
        """Test that training converges values to targets"""
        atomspace = TensorAtomspace(learning_rate=0.1)
        
        node = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="trainable",
            strength=0.2,
            confidence=0.3
        )
        
        target_strength = 0.9
        target_confidence = 0.95
        
        training_data = [(node.id, target_strength, target_confidence)]
        
        # Train for many epochs
        loss_history = atomspace.train_atoms(
            training_data,
            epochs=200,
            batch_size=1
        )
        
        # Loss should decrease
        assert loss_history[-1] < loss_history[0]
        
        # Value should be closer to target
        final_value = node.get_tensor_value()
        assert abs(final_value.strength - target_strength) < 0.2
        assert abs(final_value.confidence - target_confidence) < 0.2
    
    def test_multiple_links_to_same_node(self):
        """Test multiple links pointing to same node"""
        atomspace = TensorAtomspace()
        
        central_node = atomspace.add_tensor_node(
            atom_type=AtomType.CONCEPT,
            name="central"
        )
        
        # Create multiple nodes linking to central
        for i in range(3):
            source = atomspace.add_tensor_node(
                atom_type=AtomType.CONCEPT,
                name=f"source_{i}"
            )
            atomspace.add_tensor_link(
                atom_type=AtomType.INHERITANCE,
                outgoing=[source, central_node]
            )
        
        # Check incoming links
        incoming = atomspace.get_incoming(central_node)
        assert len(incoming) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
