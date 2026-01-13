"""
Unit tests for Symbolic-Neural Integration

Tests the bridge between symbolic logic and neural learning including:
- LogicTensor creation and evaluation
- Fuzzy logic operators
- Learnable parameters
- SymbolicNeuralBridge functionality
- Rule learning from data
"""

import pytest
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tensor_logic.tensor_space import TensorSpace, TensorAtomValue
from tensor_logic.symbolic_integration import (
    LogicTensor,
    LogicOperator,
    FuzzyNorm,
    SymbolicNeuralBridge
)


class TestLogicTensor:
    """Test cases for LogicTensor"""
    
    def test_initialization(self):
        """Test basic initialization"""
        tensor = LogicTensor(
            name="test_tensor",
            operator=LogicOperator.AND,
            operands=["atom1", "atom2"]
        )
        assert tensor.name == "test_tensor"
        assert tensor.operator == LogicOperator.AND
        assert len(tensor.operands) == 2
        assert 'weight' in tensor.parameters
        assert 'bias' in tensor.parameters
    
    def test_and_operator_product_norm(self):
        """Test AND operator with product t-norm"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.register_atom("atom2")
        tensor_space.update_value("atom1", strength=0.8, confidence=0.9)
        tensor_space.update_value("atom2", strength=0.6, confidence=0.9)
        
        logic_tensor = LogicTensor(
            name="and_test",
            operator=LogicOperator.AND,
            operands=["atom1", "atom2"],
            norm=FuzzyNorm.PRODUCT
        )
        logic_tensor.parameters = {'weight': 1.0, 'bias': 0.0}  # No transformation
        
        # Need to account for sigmoid activation
        result = logic_tensor.evaluate(tensor_space)
        # With weight=1, bias=0, input should be 0.8 * 0.6 = 0.48
        # After sigmoid: 1 / (1 + exp(-0.48)) ≈ 0.618
        assert 0.6 < result < 0.65
    
    def test_or_operator_product_norm(self):
        """Test OR operator with product t-norm"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.register_atom("atom2")
        tensor_space.update_value("atom1", strength=0.8, confidence=0.9)
        tensor_space.update_value("atom2", strength=0.6, confidence=0.9)
        
        logic_tensor = LogicTensor(
            name="or_test",
            operator=LogicOperator.OR,
            operands=["atom1", "atom2"],
            norm=FuzzyNorm.PRODUCT
        )
        logic_tensor.parameters = {'weight': 1.0, 'bias': 0.0}
        
        result = logic_tensor.evaluate(tensor_space)
        # OR with product: 1 - (1-0.8)*(1-0.6) = 1 - 0.08 = 0.92
        # After sigmoid: 1 / (1 + exp(-0.92)) ≈ 0.715
        assert 0.7 < result < 0.75
    
    def test_not_operator(self):
        """Test NOT operator"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.update_value("atom1", strength=0.7, confidence=0.9)
        
        logic_tensor = LogicTensor(
            name="not_test",
            operator=LogicOperator.NOT,
            operands=["atom1"]
        )
        logic_tensor.parameters = {'weight': 1.0, 'bias': 0.0}
        
        result = logic_tensor.evaluate(tensor_space)
        # NOT: 1 - 0.7 = 0.3
        # After sigmoid: 1 / (1 + exp(-0.3)) ≈ 0.574
        assert 0.55 < result < 0.60
    
    def test_implies_operator(self):
        """Test IMPLIES operator"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.register_atom("atom2")
        tensor_space.update_value("atom1", strength=0.8, confidence=0.9)
        tensor_space.update_value("atom2", strength=0.9, confidence=0.9)
        
        logic_tensor = LogicTensor(
            name="implies_test",
            operator=LogicOperator.IMPLIES,
            operands=["atom1", "atom2"]
        )
        
        result = logic_tensor.evaluate(tensor_space)
        # A → B = ¬A ∨ B = 0.2 ∨ 0.9
        # Should be high since consequent is high
        assert result > 0.5
    
    def test_godel_norm(self):
        """Test Gödel (minimum) t-norm"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.register_atom("atom2")
        tensor_space.update_value("atom1", strength=0.8, confidence=0.9)
        tensor_space.update_value("atom2", strength=0.6, confidence=0.9)
        
        logic_tensor = LogicTensor(
            name="and_godel",
            operator=LogicOperator.AND,
            operands=["atom1", "atom2"],
            norm=FuzzyNorm.GODEL
        )
        logic_tensor.parameters = {'weight': 1.0, 'bias': 0.0}
        
        result = logic_tensor.evaluate(tensor_space)
        # Min(0.8, 0.6) = 0.6, then sigmoid
        # 1 / (1 + exp(-0.6)) ≈ 0.646
        assert 0.62 < result < 0.67
    
    def test_lukasiewicz_norm(self):
        """Test Lukasiewicz t-norm"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.register_atom("atom2")
        tensor_space.update_value("atom1", strength=0.8, confidence=0.9)
        tensor_space.update_value("atom2", strength=0.6, confidence=0.9)
        
        logic_tensor = LogicTensor(
            name="and_lukasiewicz",
            operator=LogicOperator.AND,
            operands=["atom1", "atom2"],
            norm=FuzzyNorm.LUKASIEWICZ
        )
        logic_tensor.parameters = {'weight': 1.0, 'bias': 0.0}
        
        result = logic_tensor.evaluate(tensor_space)
        # max(0, 0.8 + 0.6 - 1) = 0.4, then sigmoid
        assert 0.55 < result < 0.65
    
    def test_nested_logic_tensors(self):
        """Test nested logic tensor evaluation"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.register_atom("atom2")
        tensor_space.register_atom("atom3")
        tensor_space.update_value("atom1", strength=0.9, confidence=0.9)
        tensor_space.update_value("atom2", strength=0.8, confidence=0.9)
        tensor_space.update_value("atom3", strength=0.7, confidence=0.9)
        
        # (atom1 AND atom2) OR atom3
        and_tensor = LogicTensor(
            name="and_part",
            operator=LogicOperator.AND,
            operands=["atom1", "atom2"]
        )
        
        or_tensor = LogicTensor(
            name="or_part",
            operator=LogicOperator.OR,
            operands=[and_tensor, "atom3"]
        )
        
        result = or_tensor.evaluate(tensor_space)
        # Should be relatively high since all values are high
        assert result > 0.5
    
    def test_cache_invalidation(self):
        """Test cache invalidation"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.update_value("atom1", strength=0.8, confidence=0.9)
        
        logic_tensor = LogicTensor(
            name="test",
            operator=LogicOperator.AND,
            operands=["atom1"]
        )
        
        # First evaluation
        result1 = logic_tensor.evaluate(tensor_space)
        assert logic_tensor._cache_valid
        
        # Invalidate cache
        logic_tensor.invalidate_cache()
        assert not logic_tensor._cache_valid
        
        # Second evaluation should recompute
        result2 = logic_tensor.evaluate(tensor_space)
        assert logic_tensor._cache_valid
        assert result1 == result2
    
    def test_compute_gradient(self):
        """Test gradient computation"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.update_value("atom1", strength=0.5, confidence=0.9)
        
        logic_tensor = LogicTensor(
            name="test",
            operator=LogicOperator.AND,
            operands=["atom1"]
        )
        
        gradients = logic_tensor.compute_gradient(
            tensor_space,
            target_value=0.8
        )
        
        assert 'weight' in gradients
        assert 'bias' in gradients
        assert isinstance(gradients['weight'], float)
        assert isinstance(gradients['bias'], float)
    
    def test_update_parameters(self):
        """Test parameter updates"""
        tensor_space = TensorSpace()
        tensor_space.register_atom("atom1")
        tensor_space.update_value("atom1", strength=0.5, confidence=0.9)
        
        logic_tensor = LogicTensor(
            name="test",
            operator=LogicOperator.AND,
            operands=["atom1"]
        )
        
        initial_weight = logic_tensor.parameters['weight']
        gradients = {'weight': 0.1, 'bias': 0.05}
        
        logic_tensor.update_parameters(gradients, learning_rate=0.1)
        
        # Weight should have changed
        assert logic_tensor.parameters['weight'] != initial_weight
        # Cache should be invalidated
        assert not logic_tensor._cache_valid


class TestSymbolicNeuralBridge:
    """Test cases for SymbolicNeuralBridge"""
    
    def test_initialization(self):
        """Test bridge initialization"""
        tensor_space = TensorSpace()
        bridge = SymbolicNeuralBridge(tensor_space)
        
        assert bridge.tensor_space is tensor_space
        assert len(bridge.logic_tensors) == 0
        assert len(bridge.training_history) == 0
    
    def test_create_logic_tensor(self):
        """Test logic tensor creation"""
        tensor_space = TensorSpace()
        bridge = SymbolicNeuralBridge(tensor_space)
        
        logic_tensor = bridge.create_logic_tensor(
            name="test_rule",
            operator=LogicOperator.AND,
            operands=["atom1", "atom2"]
        )
        
        assert "test_rule" in bridge.logic_tensors
        assert logic_tensor.name == "test_rule"
        assert logic_tensor.operator == LogicOperator.AND
    
    def test_evaluate_formula(self):
        """Test formula evaluation"""
        tensor_space = TensorSpace()
        bridge = SymbolicNeuralBridge(tensor_space)
        
        tensor_space.register_atom("atom1")
        tensor_space.update_value("atom1", strength=0.8, confidence=0.9)
        
        bridge.create_logic_tensor(
            name="simple",
            operator=LogicOperator.AND,
            operands=["atom1"]
        )
        
        result = bridge.evaluate_formula("simple")
        assert 0.0 <= result <= 1.0
    
    def test_train_formula(self):
        """Test formula training"""
        tensor_space = TensorSpace()
        bridge = SymbolicNeuralBridge(tensor_space)
        
        tensor_space.register_atom("atom1")
        tensor_space.register_atom("atom2")
        tensor_space.update_value("atom1", strength=0.5, confidence=0.9)
        tensor_space.update_value("atom2", strength=0.5, confidence=0.9)
        
        bridge.create_logic_tensor(
            name="trainable",
            operator=LogicOperator.AND,
            operands=["atom1", "atom2"]
        )
        
        training_examples = [
            ({"atom1": "atom1", "atom2": "atom2"}, 0.9),
            ({"atom1": "atom1", "atom2": "atom2"}, 0.8)
        ]
        
        loss_history = bridge.train_formula(
            "trainable",
            training_examples,
            epochs=50,
            learning_rate=0.01
        )
        
        assert len(loss_history) == 50
        # Loss should generally decrease
        assert loss_history[-1] < loss_history[0]
        assert len(bridge.training_history) > 0
    
    def test_learn_rule(self):
        """Test rule learning from data"""
        tensor_space = TensorSpace()
        bridge = SymbolicNeuralBridge(tensor_space)
        
        # Register atoms
        for atom in ["atom1", "atom2", "atom3"]:
            tensor_space.register_atom(atom)
            tensor_space.update_value(atom, strength=0.5, confidence=0.9)
        
        training_data = [
            {
                'bindings': {"x": "atom1", "y": "atom3"},
                'target': 0.9
            },
            {
                'bindings': {"x": "atom2", "y": "atom3"},
                'target': 0.8
            }
        ]
        
        rule = bridge.learn_rule(
            rule_name="learned_rule",
            antecedent_atoms=["atom1"],
            consequent_atoms=["atom3"],
            training_data=training_data
        )
        
        assert rule.name == "learned_rule"
        assert "learned_rule" in bridge.logic_tensors
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        tensor_space = TensorSpace()
        bridge = SymbolicNeuralBridge(tensor_space)
        
        bridge.create_logic_tensor(
            name="rule1",
            operator=LogicOperator.AND,
            operands=["atom1", "atom2"]
        )
        
        stats = bridge.get_statistics()
        
        assert stats['total_logic_tensors'] == 1
        assert 'rule1' in stats['logic_tensor_names']
        assert 'tensor_space_stats' in stats
    
    def test_variable_bindings(self):
        """Test evaluation with variable bindings"""
        tensor_space = TensorSpace()
        bridge = SymbolicNeuralBridge(tensor_space)
        
        tensor_space.register_atom("concrete_atom")
        tensor_space.update_value("concrete_atom", strength=0.8, confidence=0.9)
        
        bridge.create_logic_tensor(
            name="with_variable",
            operator=LogicOperator.AND,
            operands=["x"]  # Variable
        )
        
        # Bind variable to concrete atom
        result = bridge.evaluate_formula(
            "with_variable",
            variable_bindings={"x": "concrete_atom"}
        )
        
        assert 0.0 <= result <= 1.0
    
    def test_multiple_formula_training(self):
        """Test training multiple formulas"""
        tensor_space = TensorSpace()
        bridge = SymbolicNeuralBridge(tensor_space)
        
        for i in range(3):
            tensor_space.register_atom(f"atom{i}")
            tensor_space.update_value(f"atom{i}", strength=0.5, confidence=0.9)
        
        # Create multiple formulas
        for i in range(3):
            bridge.create_logic_tensor(
                name=f"formula{i}",
                operator=LogicOperator.AND,
                operands=[f"atom{i}"]
            )
        
        # Train each
        for i in range(3):
            training_examples = [({"x": f"atom{i}"}, 0.8)]
            bridge.train_formula(f"formula{i}", training_examples, epochs=10)
        
        assert len(bridge.training_history) == 3
    
    def test_complex_rule_learning(self):
        """Test learning complex multi-atom rules"""
        tensor_space = TensorSpace()
        bridge = SymbolicNeuralBridge(tensor_space)
        
        # Setup atoms
        atoms = ["cond1", "cond2", "result1", "result2"]
        for atom in atoms:
            tensor_space.register_atom(atom)
            tensor_space.update_value(atom, strength=0.5, confidence=0.9)
        
        training_data = [
            {
                'bindings': {},
                'target': 0.9
            }
        ]
        
        rule = bridge.learn_rule(
            rule_name="complex_rule",
            antecedent_atoms=["cond1", "cond2"],
            consequent_atoms=["result1", "result2"],
            training_data=training_data,
            operator=LogicOperator.IMPLIES
        )
        
        assert rule is not None
        assert rule.operator == LogicOperator.IMPLIES


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
