"""
Gradient-Based Reasoner

Reasoning engine that uses gradient descent to learn and optimize
inference rules in the Tensor Logic framework.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .tensor_space import TensorSpace, TensorAtomValue
from .symbolic_integration import LogicTensor, LogicOperator, SymbolicNeuralBridge
from .tensor_atoms import TensorAtomspace, TensorAtom, TensorNode, TensorLink


class OptimizationMethod(Enum):
    """Optimization methods for gradient-based learning."""
    SGD = "sgd"  # Stochastic gradient descent
    MOMENTUM = "momentum"  # SGD with momentum
    ADAM = "adam"  # Adaptive moment estimation
    RMSPROP = "rmsprop"  # Root mean square propagation


@dataclass
class TensorInferenceRule:
    """
    An inference rule with learnable parameters.
    
    Combines symbolic logic patterns with neural learning to create
    rules that can be optimized based on data.
    """
    name: str
    antecedent: LogicTensor
    consequent: LogicTensor
    confidence_threshold: float = 0.7
    learning_rate: float = 0.01
    
    # Rule statistics
    applications: int = 0
    successful_applications: int = 0
    total_loss: float = 0.0
    
    def apply(
        self,
        tensor_space: TensorSpace,
        variable_bindings: Optional[Dict[str, str]] = None
    ) -> Tuple[bool, float]:
        """
        Apply the inference rule.
        
        Args:
            tensor_space: Tensor space
            variable_bindings: Variable bindings
        
        Returns:
            Tuple of (rule_fired, confidence)
        """
        self.applications += 1
        
        # Evaluate antecedent
        antecedent_value = self.antecedent.evaluate(tensor_space, variable_bindings)
        
        # Check if rule should fire
        if antecedent_value >= self.confidence_threshold:
            # Evaluate consequent
            consequent_value = self.consequent.evaluate(tensor_space, variable_bindings)
            
            self.successful_applications += 1
            return (True, consequent_value)
        
        return (False, 0.0)
    
    def learn_from_feedback(
        self,
        tensor_space: TensorSpace,
        variable_bindings: Dict[str, str],
        expected_outcome: float
    ):
        """
        Update rule parameters based on feedback.
        
        Args:
            tensor_space: Tensor space
            variable_bindings: Variable bindings
            expected_outcome: Expected consequent value
        """
        # Compute gradients for consequent
        gradients = self.consequent.compute_gradient(
            tensor_space,
            expected_outcome,
            variable_bindings
        )
        
        # Update consequent parameters
        self.consequent.update_parameters(gradients, self.learning_rate)
        
        # Compute loss
        predicted = self.consequent.evaluate(tensor_space, variable_bindings)
        loss = (predicted - expected_outcome) ** 2
        self.total_loss += loss
    
    def get_success_rate(self) -> float:
        """Get the success rate of this rule."""
        if self.applications == 0:
            return 0.0
        return self.successful_applications / self.applications
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for this rule."""
        return {
            'name': self.name,
            'applications': self.applications,
            'successful_applications': self.successful_applications,
            'success_rate': self.get_success_rate(),
            'total_loss': self.total_loss,
            'avg_loss': self.total_loss / max(1, self.successful_applications)
        }


class GradientReasoner:
    """
    Gradient-based reasoning engine for Tensor Logic.
    
    Provides:
    - Learning inference rules from data
    - Gradient-based optimization of reasoning
    - Neural-symbolic rule induction
    - Differentiable forward chaining
    - Probabilistic inference with learning
    """
    
    def __init__(
        self,
        tensor_atomspace: TensorAtomspace,
        symbolic_bridge: Optional[SymbolicNeuralBridge] = None,
        optimization_method: OptimizationMethod = OptimizationMethod.MOMENTUM
    ):
        self.tensor_atomspace = tensor_atomspace
        self.tensor_space = tensor_atomspace.tensor_space
        self.symbolic_bridge = symbolic_bridge or SymbolicNeuralBridge(self.tensor_space)
        self.optimization_method = optimization_method
        
        self.inference_rules: List[TensorInferenceRule] = []
        self.logger = logging.getLogger(__name__)
        
        # Optimization state
        self.iteration = 0
        self.loss_history: List[float] = []
    
    def add_inference_rule(
        self,
        name: str,
        antecedent: LogicTensor,
        consequent: LogicTensor,
        confidence_threshold: float = 0.7,
        learning_rate: float = 0.01
    ) -> TensorInferenceRule:
        """
        Add a learnable inference rule.
        
        Args:
            name: Rule name
            antecedent: Antecedent logic tensor
            consequent: Consequent logic tensor
            confidence_threshold: Minimum confidence to fire
            learning_rate: Learning rate for rule
        
        Returns:
            Created TensorInferenceRule
        """
        rule = TensorInferenceRule(
            name=name,
            antecedent=antecedent,
            consequent=consequent,
            confidence_threshold=confidence_threshold,
            learning_rate=learning_rate
        )
        self.inference_rules.append(rule)
        return rule
    
    def forward_chain(
        self,
        max_iterations: int = 10,
        variable_bindings: Optional[Dict[str, str]] = None
    ) -> List[Tuple[str, bool, float]]:
        """
        Perform differentiable forward chaining inference.
        
        Args:
            max_iterations: Maximum inference iterations
            variable_bindings: Variable bindings
        
        Returns:
            List of (rule_name, fired, confidence) tuples
        """
        results = []
        
        for iteration in range(max_iterations):
            iteration_changed = False
            
            for rule in self.inference_rules:
                fired, confidence = rule.apply(
                    self.tensor_space,
                    variable_bindings
                )
                
                results.append((rule.name, fired, confidence))
                
                if fired:
                    iteration_changed = True
            
            # Stop if no rules fired
            if not iteration_changed:
                break
        
        return results
    
    def learn_inference_rule(
        self,
        rule_name: str,
        training_examples: List[Dict[str, Any]],
        antecedent_atoms: List[str],
        consequent_atoms: List[str],
        epochs: int = 100,
        learning_rate: float = 0.01
    ) -> TensorInferenceRule:
        """
        Learn an inference rule from training examples.
        
        Args:
            rule_name: Name for the learned rule
            training_examples: Training data
            antecedent_atoms: Atoms in antecedent
            consequent_atoms: Atoms in consequent
            epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            Learned TensorInferenceRule
        """
        # Create logic tensors for antecedent and consequent
        if len(antecedent_atoms) > 1:
            antecedent = LogicTensor(
                name=f"{rule_name}_antecedent",
                operator=LogicOperator.AND,
                operands=antecedent_atoms
            )
        else:
            antecedent = LogicTensor(
                name=f"{rule_name}_antecedent",
                operator=LogicOperator.AND,
                operands=antecedent_atoms if antecedent_atoms else ["true"]
            )
        
        if len(consequent_atoms) > 1:
            consequent = LogicTensor(
                name=f"{rule_name}_consequent",
                operator=LogicOperator.AND,
                operands=consequent_atoms
            )
        else:
            consequent = LogicTensor(
                name=f"{rule_name}_consequent",
                operator=LogicOperator.AND,
                operands=consequent_atoms if consequent_atoms else ["true"]
            )
        
        # Create rule
        rule = self.add_inference_rule(
            name=rule_name,
            antecedent=antecedent,
            consequent=consequent,
            learning_rate=learning_rate
        )
        
        # Train the rule
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for example in training_examples:
                bindings = example.get('bindings', {})
                expected_output = example.get('expected_output', 1.0)
                
                # Learn from this example
                rule.learn_from_feedback(
                    self.tensor_space,
                    bindings,
                    expected_output
                )
                
                # Track loss
                predicted = consequent.evaluate(self.tensor_space, bindings)
                epoch_loss += (predicted - expected_output) ** 2
            
            avg_loss = epoch_loss / len(training_examples) if training_examples else 0.0
            self.loss_history.append(avg_loss)
        
        return rule
    
    def induce_rules_from_data(
        self,
        data: List[Dict[str, Any]],
        max_rules: int = 10,
        min_confidence: float = 0.6
    ) -> List[TensorInferenceRule]:
        """
        Automatically induce inference rules from data.
        
        Args:
            data: Training data
            max_rules: Maximum number of rules to induce
            min_confidence: Minimum confidence for induced rules
        
        Returns:
            List of induced rules
        """
        induced_rules = []
        
        # Simple rule induction: find frequent patterns
        # This is a placeholder for more sophisticated induction
        for i in range(min(max_rules, len(data))):
            example = data[i]
            
            # Extract atoms from example
            atoms = example.get('atoms', [])
            target = example.get('target', 1.0)
            
            if len(atoms) >= 2:
                # Create simple rule: first half → second half
                mid = len(atoms) // 2
                antecedent_atoms = atoms[:mid]
                consequent_atoms = atoms[mid:]
                
                rule = self.learn_inference_rule(
                    rule_name=f"induced_rule_{i}",
                    training_examples=[example],
                    antecedent_atoms=antecedent_atoms,
                    consequent_atoms=consequent_atoms,
                    epochs=50
                )
                
                induced_rules.append(rule)
        
        return induced_rules
    
    def optimize_atomspace(
        self,
        training_data: List[Tuple[str, float, float]],
        epochs: int = 100,
        batch_size: int = 32
    ) -> List[float]:
        """
        Optimize the entire tensor atomspace.
        
        Args:
            training_data: List of (atom_id, target_strength, target_confidence)
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Loss history
        """
        return self.tensor_atomspace.train_atoms(
            training_data,
            epochs=epochs,
            batch_size=batch_size
        )
    
    def probabilistic_query(
        self,
        query_atoms: List[str],
        evidence: Dict[str, float],
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Perform probabilistic query with evidence.
        
        Args:
            query_atoms: Atoms to query
            evidence: Evidence as atom_id -> truth_value
            num_samples: Number of samples for inference
        
        Returns:
            Dictionary of query results
        """
        # Set evidence
        for atom_id, truth_value in evidence.items():
            self.tensor_space.update_value(
                atom_id,
                strength=truth_value,
                confidence=0.9
            )
        
        # Run forward chaining multiple times
        results = {atom_id: 0.0 for atom_id in query_atoms}
        
        for _ in range(num_samples):
            self.forward_chain(max_iterations=5)
            
            # Accumulate results
            for atom_id in query_atoms:
                value = self.tensor_space.get_value(atom_id)
                if value:
                    results[atom_id] += value.strength
        
        # Average over samples
        for atom_id in results:
            results[atom_id] /= num_samples
        
        return results
    
    def explain_inference(
        self,
        atom_id: str,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Explain how an atom's value was inferred.
        
        Args:
            atom_id: Atom to explain
            max_depth: Maximum explanation depth
        
        Returns:
            Explanation dictionary
        """
        explanation = {
            'atom_id': atom_id,
            'value': None,
            'rules_applied': [],
            'dependencies': []
        }
        
        # Get current value
        value = self.tensor_space.get_value(atom_id)
        if value:
            explanation['value'] = value.to_dict()
        
        # Find rules that produced this atom
        for rule in self.inference_rules:
            if atom_id in rule.consequent.operands:
                explanation['rules_applied'].append({
                    'rule_name': rule.name,
                    'success_rate': rule.get_success_rate(),
                    'applications': rule.applications
                })
        
        # Get atom from atomspace
        atom = self.tensor_atomspace.atoms.get(atom_id)
        if atom and isinstance(atom, TensorLink):
            # Get dependencies from outgoing atoms
            for outgoing in atom.outgoing:
                if max_depth > 0:
                    sub_explanation = self.explain_inference(
                        outgoing.id,
                        max_depth - 1
                    )
                    explanation['dependencies'].append(sub_explanation)
        
        return explanation
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reasoner."""
        rule_stats = [rule.get_statistics() for rule in self.inference_rules]
        
        return {
            'total_rules': len(self.inference_rules),
            'optimization_method': self.optimization_method.value,
            'iteration': self.iteration,
            'rules': rule_stats,
            'loss_history': self.loss_history[-100:],  # Last 100 losses
            'tensorspace_stats': self.tensor_space.get_statistics(),
            'atomspace_stats': self.tensor_atomspace.get_tensor_statistics()
        }
