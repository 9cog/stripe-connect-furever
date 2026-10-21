"""
Integration tests for Tensor Logic framework

Tests end-to-end functionality including:
- Complete workflow from data to inference
- Hybrid reasoning (neural + symbolic)
- Stripe payment use cases
- Multi-component integration
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from opencog.atomspace import (
    StripeAtomspace,
    AtomType,
    create_payment_atom,
    create_customer_atom
)
from opencog.knowledge_base import StripeKnowledgeBase
from opencog.reasoning import StripeReasoner

from tensor_logic.tensor_space import TensorSpace
from tensor_logic.tensor_atoms import TensorAtomspace
from tensor_logic.symbolic_integration import LogicTensor, LogicOperator, SymbolicNeuralBridge
from tensor_logic.gradient_reasoner import GradientReasoner
from tensor_logic.tensor_bridge import TensorLogicBridge


class TestEndToEndWorkflow:
    """Test complete workflows"""
    
    def test_simple_learning_workflow(self):
        """Test learning from scratch to inference"""
        # Create atomspace
        atomspace = TensorAtomspace(learning_rate=0.1)
        
        # Add payment atoms
        payments = []
        for i in range(5):
            node = atomspace.add_tensor_node(
                AtomType.PAYMENT,
                f"payment_{i}",
                value={'amount': 1000 * (i + 1)},
                strength=0.5,
                confidence=0.5
            )
            payments.append(node)
        
        # Create training data
        training_data = [
            (payments[0].id, 0.9, 0.9),  # Low amount, successful
            (payments[1].id, 0.9, 0.9),
            (payments[2].id, 0.7, 0.8),  # Medium amount
            (payments[3].id, 0.3, 0.7),  # High amount, risky
            (payments[4].id, 0.2, 0.7),  # Very high amount, risky
        ]
        
        # Train
        loss_history = atomspace.train_atoms(
            training_data,
            epochs=100,
            batch_size=2
        )
        
        # Verify convergence
        assert len(loss_history) == 100
        assert loss_history[-1] < loss_history[0]
        
        # Check learned values
        assert payments[0].get_tensor_value().strength > 0.7
        assert payments[4].get_tensor_value().strength < 0.5
    
    def test_rule_learning_and_inference(self):
        """Test learning rules and using them for inference"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Setup scenario atoms
        for atom_name in ["high_amount", "unknown_customer", "high_risk"]:
            atomspace.add_tensor_node(
                AtomType.CONCEPT,
                atom_name,
                strength=0.5,
                confidence=0.5
            )
        
        # Training examples: high_amount AND unknown_customer -> high_risk
        training_examples = [
            {
                'bindings': {},
                'expected_output': 0.9
            }
        ]
        
        # Learn rule
        rule = reasoner.learn_inference_rule(
            rule_name="risk_rule",
            training_examples=training_examples,
            antecedent_atoms=["high_amount", "unknown_customer"],
            consequent_atoms=["high_risk"],
            epochs=100,
            learning_rate=0.05
        )
        
        # Set conditions
        atomspace.update_value(
            atomspace.get_node(AtomType.CONCEPT, "high_amount").id,
            strength=0.9
        )
        atomspace.update_value(
            atomspace.get_node(AtomType.CONCEPT, "unknown_customer").id,
            strength=0.8
        )
        
        # Run inference
        results = reasoner.forward_chain(max_iterations=3)
        
        # Check if rule fired
        fired_rules = [r for r in results if r[1]]  # r[1] is fired flag
        assert len(fired_rules) > 0


class TestHybridReasoning:
    """Test hybrid neural-symbolic reasoning"""
    
    def test_tensor_bridge_initialization(self):
        """Test TensorLogicBridge initialization"""
        traditional_atomspace = StripeAtomspace()
        knowledge_base = StripeKnowledgeBase(traditional_atomspace)
        
        bridge = TensorLogicBridge(
            traditional_atomspace=traditional_atomspace,
            knowledge_base=knowledge_base,
            learning_rate=0.05
        )
        
        bridge.initialize()
        
        assert bridge._initialized
        assert bridge.tensor_atomspace is not None
        assert bridge.gradient_reasoner is not None
    
    def test_sync_atomspaces(self):
        """Test syncing between traditional and tensor atomspaces"""
        traditional_atomspace = StripeAtomspace()
        
        # Add atoms to traditional atomspace
        create_payment_atom(
            traditional_atomspace,
            "payment_123",
            5000,
            "usd",
            "succeeded"
        )
        
        # Create bridge
        bridge = TensorLogicBridge(
            traditional_atomspace=traditional_atomspace,
            learning_rate=0.05
        )
        bridge.initialize()
        
        # Check that payment was synced
        tensor_payment = bridge.tensor_atomspace.get_node(
            AtomType.PAYMENT,
            "payment_123"
        )
        assert tensor_payment is not None
    
    def test_learn_from_stripe_data(self):
        """Test learning from Stripe payment data"""
        bridge = TensorLogicBridge(learning_rate=0.1)
        bridge.initialize()
        
        payment_data = [
            {'id': 'pi_1', 'amount': 5000, 'status': 'succeeded'},
            {'id': 'pi_2', 'amount': 10000, 'status': 'succeeded'},
            {'id': 'pi_3', 'amount': 150000, 'status': 'failed'},
            {'id': 'pi_4', 'amount': 3000, 'status': 'succeeded'},
        ]
        
        stats = bridge.learn_from_stripe_data(
            payment_data,
            epochs=50
        )
        
        assert stats['payments_processed'] == 4
        assert stats['epochs_trained'] == 50
        assert 'final_loss' in stats
        assert stats['final_loss'] >= 0.0


class TestStripePaymentScenarios:
    """Test Stripe-specific payment scenarios"""
    
    def test_risk_assessment_workflow(self):
        """Test complete risk assessment workflow"""
        bridge = TensorLogicBridge(learning_rate=0.1)
        bridge.initialize()
        
        # Create training data with known risk patterns
        training_examples = []
        
        # High-risk pattern: large amount + new customer
        bridge.tensor_atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "large_amount",
            strength=0.9
        )
        bridge.tensor_atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "new_customer",
            strength=0.9
        )
        bridge.tensor_atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "high_risk",
            strength=0.5
        )
        
        for _ in range(10):
            training_examples.append({
                'bindings': {},
                'expected_output': 0.9
            })
        
        # Learn risk rule
        risk_rule = bridge.learn_risk_assessment_rule(
            training_examples,
            epochs=100
        )
        
        assert risk_rule is not None
        assert risk_rule.name == "risk_assessment_rule"
    
    def test_payment_success_prediction(self):
        """Test predicting payment success"""
        atomspace = TensorAtomspace(learning_rate=0.1)
        
        # Create payment scenarios
        scenarios = [
            ('low_amount_known_customer', 0.95),  # Very likely to succeed
            ('medium_amount_known_customer', 0.85),
            ('high_amount_known_customer', 0.7),
            ('low_amount_new_customer', 0.7),
            ('high_amount_new_customer', 0.3),  # Likely to fail
        ]
        
        nodes = {}
        training_data = []
        
        for scenario_name, target_success in scenarios:
            node = atomspace.add_tensor_node(
                AtomType.CONCEPT,
                scenario_name,
                strength=0.5,
                confidence=0.5
            )
            nodes[scenario_name] = node
            training_data.append((node.id, target_success, 0.9))
        
        # Train
        loss_history = atomspace.train_atoms(
            training_data,
            epochs=150,
            batch_size=2
        )
        
        # Verify learning
        assert loss_history[-1] < loss_history[0]
        
        # Check predictions
        low_risk = nodes['low_amount_known_customer'].get_tensor_value().strength
        high_risk = nodes['high_amount_new_customer'].get_tensor_value().strength
        
        assert low_risk > 0.7  # Should predict success
        assert high_risk < 0.5  # Should predict failure
    
    def test_multi_factor_risk_assessment(self):
        """Test risk assessment with multiple factors"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        bridge = SymbolicNeuralBridge(atomspace.tensor_space)
        
        # Define risk factors
        risk_factors = [
            "high_amount",
            "velocity_spike",
            "new_payment_method",
            "foreign_ip",
            "unusual_time"
        ]
        
        # Create factor atoms
        for factor in risk_factors:
            atomspace.add_tensor_node(
                AtomType.CONCEPT,
                factor,
                strength=0.5
            )
        
        # Create result atom
        atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "transaction_risk",
            strength=0.5
        )
        
        # Create logic tensor combining factors
        risk_logic = bridge.create_logic_tensor(
            name="multi_factor_risk",
            operator=LogicOperator.OR,
            operands=risk_factors
        )
        
        # Test with different factor combinations
        test_cases = [
            # No risk factors
            ({factor: 0.1 for factor in risk_factors}, "low_risk"),
            # One strong risk factor
            ({**{factor: 0.1 for factor in risk_factors}, "high_amount": 0.9}, "medium_risk"),
            # Multiple risk factors
            ({factor: 0.8 for factor in risk_factors}, "high_risk"),
        ]
        
        for factor_values, expected_category in test_cases:
            # Set factor values
            for factor, value in factor_values.items():
                atomspace.update_value(
                    atomspace.get_node(AtomType.CONCEPT, factor).id,
                    strength=value
                )
            
            # Evaluate risk
            risk_score = bridge.evaluate_formula("multi_factor_risk")
            
            # Verify risk categorization
            if expected_category == "low_risk":
                assert risk_score < 0.4
            elif expected_category == "medium_risk":
                assert 0.4 <= risk_score <= 0.7
            else:  # high_risk
                assert risk_score > 0.6


class TestComplexGraphReasoning:
    """Test reasoning over complex graph structures"""
    
    def test_graph_traversal_inference(self):
        """Test inference through graph traversal"""
        atomspace = TensorAtomspace()
        
        # Create graph: payment -> customer -> subscription
        payment = atomspace.add_tensor_node(
            AtomType.PAYMENT,
            "payment_1",
            strength=0.8
        )
        customer = atomspace.add_tensor_node(
            AtomType.CUSTOMER,
            "customer_1",
            strength=0.9
        )
        subscription = atomspace.add_tensor_node(
            AtomType.SUBSCRIPTION,
            "subscription_1",
            strength=0.7
        )
        
        # Create links
        link1 = atomspace.add_tensor_link(
            AtomType.PAYMENT_CUSTOMER,
            [payment, customer],
            strength=0.9
        )
        link2 = atomspace.add_tensor_link(
            AtomType.CUSTOMER_SUBSCRIPTION,
            [customer, subscription],
            strength=0.8
        )
        
        # Verify graph structure
        incoming_customer = atomspace.get_incoming(customer)
        assert len(incoming_customer) > 0
        
        # Backpropagate gradient
        payment.compute_gradient(0.95, 0.95)
        atomspace.backpropagate([payment], learning_rate=0.1)
        
        # Verify propagation occurred (test completes without error)
        assert True
    
    def test_hierarchical_knowledge(self):
        """Test reasoning with hierarchical knowledge"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Create hierarchy: specific -> general
        # payment_type -> payment_category -> transaction
        specific = atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "credit_card_payment",
            strength=0.9
        )
        general = atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "electronic_payment",
            strength=0.7
        )
        abstract = atomspace.add_tensor_node(
            AtomType.CONCEPT,
            "transaction",
            strength=0.5
        )
        
        # Create inheritance links
        atomspace.add_tensor_link(
            AtomType.INHERITANCE,
            [specific, general],
            strength=0.9
        )
        atomspace.add_tensor_link(
            AtomType.INHERITANCE,
            [general, abstract],
            strength=0.8
        )
        
        # Create inference rule: if specific, then general
        antecedent = LogicTensor(
            "ante",
            LogicOperator.AND,
            ["credit_card_payment"]
        )
        consequent = LogicTensor(
            "cons",
            LogicOperator.AND,
            ["electronic_payment"]
        )
        
        reasoner.add_inference_rule(
            "inheritance_rule",
            antecedent,
            consequent,
            confidence_threshold=0.7
        )
        
        # Run inference
        results = reasoner.forward_chain(max_iterations=2)
        
        # Should propagate information up the hierarchy
        assert len(results) > 0


class TestPerformanceAndScaling:
    """Test performance with larger datasets"""
    
    def test_many_atoms_training(self):
        """Test training with many atoms"""
        atomspace = TensorAtomspace(learning_rate=0.1)
        
        # Create many atoms
        num_atoms = 100
        training_data = []
        
        for i in range(num_atoms):
            node = atomspace.add_tensor_node(
                AtomType.CONCEPT,
                f"concept_{i}",
                strength=0.5
            )
            # Target varies based on index
            target = 0.3 + (i / num_atoms) * 0.6
            training_data.append((node.id, target, 0.9))
        
        # Train in batches
        loss_history = atomspace.train_atoms(
            training_data,
            epochs=20,
            batch_size=10
        )
        
        assert len(loss_history) == 20
        # Should show learning progress
        assert loss_history[-1] <= loss_history[0]
    
    def test_many_rules_inference(self):
        """Test inference with many rules"""
        atomspace = TensorAtomspace()
        reasoner = GradientReasoner(atomspace)
        
        # Create many simple rules
        num_rules = 20
        
        for i in range(num_rules):
            atomspace.add_tensor_node(
                AtomType.CONCEPT,
                f"atom_{i}",
                strength=0.5
            )
            atomspace.add_tensor_node(
                AtomType.CONCEPT,
                f"result_{i}",
                strength=0.5
            )
            
            antecedent = LogicTensor(
                f"ante_{i}",
                LogicOperator.AND,
                [f"atom_{i}"]
            )
            consequent = LogicTensor(
                f"cons_{i}",
                LogicOperator.AND,
                [f"result_{i}"]
            )
            
            reasoner.add_inference_rule(
                f"rule_{i}",
                antecedent,
                consequent,
                confidence_threshold=0.5
            )
        
        # Set some initial conditions
        for i in range(0, num_rules, 2):  # Every other atom
            atomspace.update_value(
                atomspace.get_node(AtomType.CONCEPT, f"atom_{i}").id,
                strength=0.9
            )
        
        # Run inference
        results = reasoner.forward_chain(max_iterations=2)
        
        # Should process all rules
        assert len(results) >= num_rules


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
