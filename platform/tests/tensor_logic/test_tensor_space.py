"""
Unit tests for Tensor Space

Tests the core tensor space functionality including:
- Tensor value creation and updates
- Gradient computation and accumulation
- Optimization steps
- Batch training
- Loss computation
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tensor_logic.tensor_space import TensorSpace, TensorAtomValue


class TestTensorAtomValue:
    """Test cases for TensorAtomValue"""
    
    def test_initialization(self):
        """Test basic initialization"""
        value = TensorAtomValue()
        assert value.strength == 0.5
        assert value.confidence == 0.5
        assert value.sti == 0.0
        assert value.lti == 0.0
        assert value.update_count == 0
    
    def test_initialization_with_values(self):
        """Test initialization with custom values"""
        value = TensorAtomValue(strength=0.8, confidence=0.9, sti=0.5, lti=0.3)
        assert value.strength == 0.8
        assert value.confidence == 0.9
        assert value.sti == 0.5
        assert value.lti == 0.3
    
    def test_clamping(self):
        """Test value clamping to valid ranges"""
        value = TensorAtomValue(strength=1.5, confidence=-0.5, sti=2.0, lti=-2.0)
        assert value.strength == 1.0  # Clamped to max
        assert value.confidence == 0.0  # Clamped to min
        assert value.sti == 1.0  # Clamped to max
        assert value.lti == -1.0  # Clamped to min
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation"""
        value = TensorAtomValue()
        value.accumulate_gradient(grad_strength=0.1, grad_confidence=0.2)
        assert value.grad_strength == 0.1
        assert value.grad_confidence == 0.2
        
        value.accumulate_gradient(grad_strength=0.3, grad_confidence=0.4)
        assert value.grad_strength == 0.4  # 0.1 + 0.3
        assert value.grad_confidence == 0.6  # 0.2 + 0.4
    
    def test_gradient_reset(self):
        """Test gradient reset"""
        value = TensorAtomValue()
        value.accumulate_gradient(grad_strength=0.5, grad_confidence=0.6)
        value.reset_gradients()
        assert value.grad_strength == 0.0
        assert value.grad_confidence == 0.0
    
    def test_update_with_gradient(self):
        """Test value update using gradients"""
        value = TensorAtomValue(strength=0.5, confidence=0.5)
        value.accumulate_gradient(grad_strength=1.0, grad_confidence=-1.0)
        
        initial_update_count = value.update_count
        value.update_with_gradient(learning_rate=0.1)
        
        # Values should change based on gradients
        assert value.strength < 0.5  # Decreased due to positive gradient
        assert value.confidence > 0.5  # Increased due to negative gradient
        assert value.update_count == initial_update_count + 1
        
        # Gradients should be reset
        assert value.grad_strength == 0.0
        assert value.grad_confidence == 0.0
    
    def test_to_tuple(self):
        """Test conversion to tuple"""
        value = TensorAtomValue(strength=0.7, confidence=0.8)
        tuple_val = value.to_tuple()
        assert tuple_val == (0.7, 0.8)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        value = TensorAtomValue(strength=0.7, confidence=0.8)
        dict_val = value.to_dict()
        assert dict_val['strength'] == 0.7
        assert dict_val['confidence'] == 0.8
        assert 'last_updated' in dict_val
        assert 'update_count' in dict_val


class TestTensorSpace:
    """Test cases for TensorSpace"""
    
    def test_initialization(self):
        """Test tensor space initialization"""
        space = TensorSpace(name="test_space", learning_rate=0.01)
        assert space.name == "test_space"
        assert space.learning_rate == 0.01
        assert len(space.tensor_values) == 0
        assert space.epoch == 0
    
    def test_register_atom(self):
        """Test atom registration"""
        space = TensorSpace()
        value = space.register_atom("atom1")
        
        assert "atom1" in space.tensor_values
        assert isinstance(value, TensorAtomValue)
        assert "atom1" in space.velocity
    
    def test_register_atom_with_initial_value(self):
        """Test atom registration with initial value"""
        space = TensorSpace()
        initial_value = TensorAtomValue(strength=0.8, confidence=0.9)
        value = space.register_atom("atom1", initial_value)
        
        assert value.strength == 0.8
        assert value.confidence == 0.9
    
    def test_register_atom_idempotent(self):
        """Test that registering same atom twice returns same value"""
        space = TensorSpace()
        value1 = space.register_atom("atom1")
        value1.strength = 0.7
        
        value2 = space.register_atom("atom1")
        assert value2.strength == 0.7  # Same object
        assert value1 is value2
    
    def test_get_value(self):
        """Test getting atom value"""
        space = TensorSpace()
        space.register_atom("atom1")
        
        value = space.get_value("atom1")
        assert value is not None
        assert isinstance(value, TensorAtomValue)
        
        # Non-existent atom
        value = space.get_value("nonexistent")
        assert value is None
    
    def test_update_value(self):
        """Test updating atom value"""
        space = TensorSpace()
        space.register_atom("atom1")
        
        space.update_value("atom1", strength=0.7, confidence=0.8)
        value = space.get_value("atom1")
        
        assert value.strength == 0.7
        assert value.confidence == 0.8
    
    def test_compute_loss_mse(self):
        """Test MSE loss computation"""
        space = TensorSpace()
        space.register_atom("atom1")
        space.update_value("atom1", strength=0.6, confidence=0.7)
        
        loss = space.compute_loss("atom1", target_strength=0.8, target_confidence=0.9, loss_type="mse")
        
        # MSE = ((0.6-0.8)^2 + (0.7-0.9)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
        assert abs(loss - 0.04) < 1e-6
    
    def test_compute_gradient_mse(self):
        """Test gradient computation for MSE"""
        space = TensorSpace()
        space.register_atom("atom1")
        space.update_value("atom1", strength=0.6, confidence=0.7)
        
        space.compute_gradient("atom1", target_strength=0.8, target_confidence=0.9, loss_type="mse")
        value = space.get_value("atom1")
        
        # Gradient of MSE: 2 * (current - target)
        assert abs(value.grad_strength - 2 * (0.6 - 0.8)) < 1e-6  # -0.4
        assert abs(value.grad_confidence - 2 * (0.7 - 0.9)) < 1e-6  # -0.4
    
    def test_step_without_momentum(self):
        """Test optimization step without momentum"""
        space = TensorSpace(learning_rate=0.1)
        space.register_atom("atom1")
        space.update_value("atom1", strength=0.5, confidence=0.5)
        
        space.compute_gradient("atom1", target_strength=0.8, target_confidence=0.8, loss_type="mse")
        space.step(use_momentum=False)
        
        value = space.get_value("atom1")
        # Should move towards target
        assert value.strength > 0.5
        assert value.confidence > 0.5
        # Gradients should be reset
        assert value.grad_strength == 0.0
    
    def test_step_with_momentum(self):
        """Test optimization step with momentum"""
        space = TensorSpace(learning_rate=0.1, momentum=0.9)
        space.register_atom("atom1")
        space.update_value("atom1", strength=0.5, confidence=0.5)
        
        # First step
        space.compute_gradient("atom1", target_strength=0.8, target_confidence=0.8, loss_type="mse")
        space.step(use_momentum=True)
        
        value1 = space.get_value("atom1")
        strength1 = value1.strength
        
        # Second step
        space.compute_gradient("atom1", target_strength=0.8, target_confidence=0.8, loss_type="mse")
        space.step(use_momentum=True)
        
        value2 = space.get_value("atom1")
        strength2 = value2.strength
        
        # With momentum, second step should make larger progress
        assert strength2 > strength1
    
    def test_train_batch(self):
        """Test batch training"""
        space = TensorSpace(learning_rate=0.1)
        
        # Register atoms
        space.register_atom("atom1")
        space.register_atom("atom2")
        space.register_atom("atom3")
        
        # Training data
        training_data = [
            ("atom1", 0.8, 0.9),
            ("atom2", 0.6, 0.7),
            ("atom3", 0.9, 0.95)
        ]
        
        initial_epoch = space.epoch
        loss = space.train_batch(training_data, loss_type="mse")
        
        assert space.epoch == initial_epoch + 1
        assert loss >= 0.0
        assert len(space.loss_history) > 0
    
    def test_zero_grad(self):
        """Test gradient zeroing"""
        space = TensorSpace()
        space.register_atom("atom1")
        
        value = space.get_value("atom1")
        value.accumulate_gradient(grad_strength=0.5, grad_confidence=0.6)
        
        space.zero_grad()
        
        assert value.grad_strength == 0.0
        assert value.grad_confidence == 0.0
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        space = TensorSpace(name="test_space")
        space.register_atom("atom1")
        space.register_atom("atom2")
        
        stats = space.get_statistics()
        
        assert stats['name'] == "test_space"
        assert stats['total_atoms'] == 2
        assert stats['epoch'] == 0
        assert 'avg_strength' in stats
        assert 'avg_confidence' in stats
    
    def test_export_values(self):
        """Test value export"""
        space = TensorSpace()
        space.register_atom("atom1")
        space.update_value("atom1", strength=0.7, confidence=0.8)
        
        exported = space.export_values()
        
        assert "atom1" in exported
        assert exported["atom1"]['strength'] == 0.7
        assert exported["atom1"]['confidence'] == 0.8
    
    def test_convergence(self):
        """Test that training converges to target"""
        space = TensorSpace(learning_rate=0.1)
        space.register_atom("atom1")
        space.update_value("atom1", strength=0.2, confidence=0.3)
        
        target_strength = 0.9
        target_confidence = 0.95
        
        # Train for many epochs
        for _ in range(100):
            space.compute_gradient("atom1", target_strength, target_confidence, loss_type="mse")
            space.step(use_momentum=False)
        
        value = space.get_value("atom1")
        
        # Should be close to target
        assert abs(value.strength - target_strength) < 0.1
        assert abs(value.confidence - target_confidence) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
