"""
Unit tests for Gradient Reasoner

Tests gradient-based reasoning including:
- TensorInferenceRule creation and application
- Rule learning from data
- Forward chaining with gradients
- Probabilistic inference
- Rule induction
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from opencog.atomspace import AtomType
from tensor_logic.tensor_atoms import TensorAtomspace
from tensor_logic.symbolic_integration import (
    LogicTensor,
    LogicOperator,
    SymbolicNeuralBridge
)
from tensor_logic.gradient_reasoner import (
    GradientReasoner,
    TensorInferenceRule,
    OptimizationMethod
)


class TestTensorInferenceRule:
    """Test cases for TensorInferenceRule"""
    
    def test_initialization(self):
        """Test rule initialization"""
        antecedent = LogicTensor("ante", LogicOperator.AND, ["atom1"])
        consequent = LogicTensor("cons", LogicOperator.AND, ["atom2"])
        
        rule = TensorInferenceRule(
            name="test_rule",
            antecedent=antecedent,
            consequent=consequent,
            confidence_threshold=0.7,
            learning_rate=0.01
        )
        
        assert rule.name == "test_rule"
        assert rule.confidence_threshold == 0.7
        assert rule.applications == 0
        assert rule.successful_applications == 0
    
    def test_rule_application_fires(self):
        """Test rule fires when antecedent is satisfied"""
        atomspace = TensorAtomspace()
        tensor_space = atomspace.tensor_space
        
        # Setup atoms
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom1", strength=0.9)
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom2", strength=0.5)
        
        antecedent = LogicTensor("ante", LogicOperator.AND, ["atom1"])
        consequent = LogicTensor("cons", LogicOperator.AND, ["atom2"])
        
        rule = TensorInferenceRule(
            name="test",
            antecedent=antecedent,
            consequent=consequent,
            confidence_threshold=0.5  # Low threshold
        )
        
        fired, confidence = rule.apply(tensor_space)
        
        assert fired
        assert rule.applications == 1
        assert rule.successful_applications == 1
    
    def test_rule_application_doesnt_fire(self):
        """Test rule doesn't fire when antecedent not satisfied"""
        atomspace = TensorAtomspace()
        tensor_space = atomspace.tensor_space
        
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom1", strength=0.3)
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom2", strength=0.5)
        
        antecedent = LogicTensor("ante", LogicOperator.AND, ["atom1"])
        consequent = LogicTensor("cons", LogicOperator.AND, ["atom2"])
        
        rule = TensorInferenceRule(
            name="test",
            antecedent=antecedent,
            consequent=consequent,
            confidence_threshold=0.8  # High threshold
        )
        
        fired, confidence = rule.apply(tensor_space)
        
        assert not fired
        assert rule.applications == 1
        assert rule.successful_applications == 0
    
    def test_learn_from_feedback(self):
        """Test rule learning from feedback"""
        atomspace = TensorAtomspace()
        tensor_space = atomspace.tensor_space
        
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom1", strength=0.5)
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom2", strength=0.5)
        
        antecedent = LogicTensor("ante", LogicOperator.AND, ["atom1"])
        consequent = LogicTensor("cons", LogicOperator.AND, ["atom2"])
        
        rule = TensorInferenceRule(
            name="learnable",
            antecedent=antecedent,
            consequent=consequent,
            learning_rate=0.1
        )
        
        initial_params = consequent.parameters.copy()
        
        rule.learn_from_feedback(
            tensor_space,
            {"atom1": "atom1", "atom2": "atom2"},
            expected_outcome=0.9
        )
        
        # Parameters should have changed
        assert consequent.parameters != initial_params
        assert rule.total_loss > 0
    
    def test_get_success_rate(self):
        """Test success rate calculation"""
        atomspace = TensorAtomspace()
        tensor_space = atomspace.tensor_space
        
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom1", strength=0.8)
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom2", strength=0.5)
        
        antecedent = LogicTensor("ante", LogicOperator.AND, ["atom1"])
        consequent = LogicTensor("cons", LogicOperator.AND, ["atom2"])
        
        rule = TensorInferenceRule(
            name="test",
            antecedent=antecedent,
            consequent=consequent,
            confidence_threshold=0.7
        )
        
        # Apply multiple times
        for i in range(10):
            atomspace.update_value(
                atomspace.get_node(AtomType.CONCEPT, "atom1").id,
                strength=0.6 + i * 0.04  # Gradually increase
            )
            rule.apply(tensor_space)
        
        success_rate = rule.get_success_rate()
        assert 0.0 <= success_rate <= 1.0
        assert rule.applications == 10


class TestGradientReasoner:
    """Test cases for GradientReasoner"""
    
    def test_initialization(self):
        """Test reasoner initialization"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        assert reasoner.tensor_atomspace is atomspace
        assert reasoner.tensor_space is atomspace.tensor_space
        assert len(reasoner.inference_rules) == 0
        assert reasoner.iteration == 0
    
    def test_add_inference_rule(self):
        """Test adding inference rules"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        antecedent = LogicTensor("ante", LogicOperator.AND, ["atom1"])
        consequent = LogicTensor("cons", LogicOperator.AND, ["atom2"])
        
        rule = reasoner.add_inference_rule(
            name="rule1",
            antecedent=antecedent,
            consequent=consequent
        )
        
        assert len(reasoner.inference_rules) == 1
        assert rule.name == "rule1"
    
    def test_forward_chain(self):
        """Test forward chaining inference"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Setup atoms
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom1", strength=0.9)
        atomspace.add_tensor_node(AtomType.CONCEPT, "atom2", strength=0.5)
        
        # Add rule: atom1 -> atom2
        antecedent = LogicTensor("ante", LogicOperator.AND, ["atom1"])
        consequent = LogicTensor("cons", LogicOperator.AND, ["atom2"])
        
        reasoner.add_inference_rule(
            name="rule1",
            antecedent=antecedent,
            consequent=consequent,
            confidence_threshold=0.5
        )
        
        # Run forward chaining
        results = reasoner.forward_chain(max_iterations=3)
        
        assert len(results) > 0
        # Each result is (rule_name, fired, confidence)
        for rule_name, fired, confidence in results:
            assert isinstance(rule_name, str)
            assert isinstance(fired, bool)
            assert 0.0 <= confidence <= 1.0
    
    def test_learn_inference_rule(self):
        """Test learning inference rule from data"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Setup atoms
        atomspace.add_tensor_node(AtomType.CONCEPT, "cond1", strength=0.5)
        atomspace.add_tensor_node(AtomType.CONCEPT, "cond2", strength=0.5)
        atomspace.add_tensor_node(AtomType.CONCEPT, "result", strength=0.5)
        
        training_examples = [
            {
                'bindings': {},
                'expected_output': 0.9
            },
            {
                'bindings': {},
                'expected_output': 0.85
            }
        ]
        
        rule = reasoner.learn_inference_rule(
            rule_name="learned_rule",
            training_examples=training_examples,
            antecedent_atoms=["cond1", "cond2"],
            consequent_atoms=["result"],
            epochs=50,
            learning_rate=0.01
        )
        
        assert rule.name == "learned_rule"
        assert len(reasoner.loss_history) > 0
    
    def test_induce_rules_from_data(self):
        """Test rule induction from data"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Create atoms
        for i in range(6):
            atomspace.add_tensor_node(
                AtomType.CONCEPT,
                f"atom{i}",
                strength=0.5
            )
        
        data = [
            {
                'atoms': ['atom0', 'atom1', 'atom2', 'atom3'],
                'target': 0.9
            },
            {
                'atoms': ['atom2', 'atom3', 'atom4', 'atom5'],
                'target': 0.8
            }
        ]
        
        induced_rules = reasoner.induce_rules_from_data(
            data,
            max_rules=2,
            min_confidence=0.6
        )
        
        assert len(induced_rules) <= 2
        assert all(isinstance(rule, TensorInferenceRule) for rule in induced_rules)
    
    def test_optimize_atomspace(self):
        """Test atomspace optimization"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Add nodes
        node1 = atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "concept1",
            strength=0.3
        )
        node2 = atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "concept2",
            strength=0.4
        )
        
        training_data = [
            (node1.id, 0.9, 0.95),
            (node2.id, 0.8, 0.85)
        ]
        
        loss_history = reasoner.optimize_atomspace(
            training_data,
            epochs=50,
            batch_size=2
        )
        
        assert len(loss_history) == 50
        # Loss should generally decrease
        assert loss_history[-1] <= loss_history[0] * 1.5  # Allow some variance
    
    def test_probabilistic_query(self):
        """Test probabilistic query with evidence"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Setup atoms
        for atom_name in ["evidence1", "evidence2", "query1", "query2"]:
            atomspace.add_tensor_node(
                AtomType.CONCEPT,
                atom_name,
                strength=0.5
            )
        
        # Add simple rule
        antecedent = LogicTensor(
            "ante",
            LogicOperator.AND,
            ["evidence1", "evidence2"]
        )
        consequent = LogicTensor(
            "cons",
            LogicOperator.AND,
            ["query1"]
        )
        reasoner.add_inference_rule(
            "inference_rule",
            antecedent,
            consequent
        )
        
        # Query with evidence
        evidence = {
            atomspace.get_node(AtomType.CONCEPT, "evidence1").id: 0.9,
            atomspace.get_node(AtomType.CONCEPT, "evidence2").id: 0.8
        }
        
        query_atoms = [
            atomspace.get_node(AtomType.CONCEPT, "query1").id
        ]
        
        results = reasoner.probabilistic_query(
            query_atoms,
            evidence,
            num_samples=10
        )
        
        assert len(results) == len(query_atoms)
        for atom_id, value in results.items():
            assert 0.0 <= value <= 1.0
    
    def test_explain_inference(self):
        """Test inference explanation"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Create simple structure
        node1 = atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "concept1",
            strength=0.8
        )
        node2 = atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "concept2",
            strength=0.7
        )
        link = atomspace.add_tensor_link(
            AtomType.INHERITANCE,
            [node1, node2],
            strength=0.9
        )
        
        # Add rule involving concept2
        antecedent = LogicTensor("ante", LogicOperator.AND, ["concept1"])
        consequent = LogicTensor("cons", LogicOperator.AND, ["concept2"])
        reasoner.add_inference_rule("rule1", antecedent, consequent)
        
        # Apply rule
        reasoner.forward_chain(max_iterations=2)
        
        # Explain
        explanation = reasoner.explain_inference(node2.id, max_depth=2)
        
        assert 'atom_id' in explanation
        assert 'value' in explanation
        assert 'rules_applied' in explanation
        assert explanation['atom_id'] == node2.id
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Add some rules
        for i in range(3):
            antecedent = LogicTensor(f"ante{i}", LogicOperator.AND, [f"atom{i}"])
            consequent = LogicTensor(f"cons{i}", LogicOperator.AND, [f"result{i}"])
            reasoner.add_inference_rule(f"rule{i}", antecedent, consequent)
        
        stats = reasoner.get_statistics()
        
        assert stats['total_rules'] == 3
        assert 'optimization_method' in stats
        assert 'rules' in stats
        assert len(stats['rules']) == 3
        assert 'tensorspace_stats' in stats
    
    def test_multiple_iterations_convergence(self):
        """Test that multiple forward chaining iterations work"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Create chain: atom1 -> atom2 -> atom3
        for i in range(1, 4):
            atomspace.add_tensor_node(
                AtomType.CONCEPT,
                f"atom{i}",
                strength=0.5
            )
        
        # Set initial condition
        atomspace.update_value(
            atomspace.get_node(AtomType.CONCEPT, "atom1").id,
            strength=0.9
        )
        
        # Add chained rules
        for i in range(1, 3):
            antecedent = LogicTensor(
                f"ante{i}",
                LogicOperator.AND,
                [f"atom{i}"]
            )
            consequent = LogicTensor(
                f"cons{i}",
                LogicOperator.AND,
                [f"atom{i+1}"]
            )
            reasoner.add_inference_rule(
                f"rule{i}",
                antecedent,
                consequent,
                confidence_threshold=0.5
            )
        
        # Run forward chaining
        results = reasoner.forward_chain(max_iterations=5)
        
        # Should fire multiple times
        fired_count = sum(1 for _, fired, _ in results if fired)
        assert fired_count > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
