"""
Symbolic-Neural Integration

Bridge between neural networks and symbolic reasoning in Tensor Logic.
Implements LogicTensors that combine logical formulas with learnable parameters.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from .tensor_space import TensorSpace, TensorAtomValue


class LogicOperator(Enum):
    """Logical operators for symbolic reasoning."""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    EQUIVALENT = "equivalent"
    EXISTS = "exists"
    FORALL = "forall"


class FuzzyNorm(Enum):
    """Fuzzy logic norms for combining truth values."""
    PRODUCT = "product"  # Product t-norm
    LUKASIEWICZ = "lukasiewicz"  # Lukasiewicz t-norm
    GODEL = "godel"  # Gödel (minimum) t-norm


@dataclass
class LogicTensor:
    """
    A tensor representing a logical formula with learnable parameters.
    
    Combines symbolic logic with gradient-based learning by representing
    logical formulas as differentiable operations on tensor values.
    """
    name: str
    operator: LogicOperator
    operands: List[Union[str, 'LogicTensor']] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    norm: FuzzyNorm = FuzzyNorm.PRODUCT
    
    # Cached evaluation result
    _cached_value: Optional[float] = None
    _cache_valid: bool = False
    
    def __post_init__(self):
        """Initialize default parameters if needed."""
        if not self.parameters:
            # Default weights for weighted combinations
            self.parameters = {'weight': 1.0, 'bias': 0.0}
    
    def invalidate_cache(self):
        """Invalidate cached evaluation result."""
        self._cache_valid = False
        self._cached_value = None
    
    def evaluate(
        self,
        tensor_space: TensorSpace,
        variable_bindings: Optional[Dict[str, str]] = None
    ) -> float:
        """
        Evaluate the logic tensor to a truth value.
        
        Args:
            tensor_space: Tensor space containing atom values
            variable_bindings: Bindings for logical variables
        
        Returns:
            Truth value in [0, 1]
        """
        if self._cache_valid and self._cached_value is not None:
            return self._cached_value
        
        variable_bindings = variable_bindings or {}
        
        # Evaluate operands
        operand_values = []
        for operand in self.operands:
            if isinstance(operand, str):
                # Atom reference or variable
                atom_id = variable_bindings.get(operand, operand)
                value = tensor_space.get_value(atom_id)
                if value:
                    operand_values.append(value.strength)
                else:
                    operand_values.append(0.5)  # Unknown = 0.5
            elif isinstance(operand, LogicTensor):
                # Nested logic tensor
                operand_values.append(
                    operand.evaluate(tensor_space, variable_bindings)
                )
        
        # Apply logical operator with fuzzy logic
        result = self._apply_operator(operand_values)
        
        # Apply learnable parameters
        result = self._apply_parameters(result)
        
        self._cached_value = result
        self._cache_valid = True
        
        return result
    
    def _apply_operator(self, values: List[float]) -> float:
        """Apply the logical operator to operand values."""
        if not values:
            return 0.5
        
        if self.operator == LogicOperator.AND:
            return self._fuzzy_and(values)
        elif self.operator == LogicOperator.OR:
            return self._fuzzy_or(values)
        elif self.operator == LogicOperator.NOT:
            return 1.0 - values[0] if values else 0.5
        elif self.operator == LogicOperator.IMPLIES:
            if len(values) >= 2:
                # A → B = ¬A ∨ B
                return self._fuzzy_or([1.0 - values[0], values[1]])
            return 0.5
        elif self.operator == LogicOperator.EQUIVALENT:
            if len(values) >= 2:
                # A ↔ B = (A → B) ∧ (B → A)
                impl1 = self._fuzzy_or([1.0 - values[0], values[1]])
                impl2 = self._fuzzy_or([1.0 - values[1], values[0]])
                return self._fuzzy_and([impl1, impl2])
            return 0.5
        elif self.operator == LogicOperator.EXISTS:
            # Exists: max of all values
            return max(values) if values else 0.0
        elif self.operator == LogicOperator.FORALL:
            # Forall: min of all values
            return min(values) if values else 1.0
        
        return 0.5
    
    def _fuzzy_and(self, values: List[float]) -> float:
        """Fuzzy AND using selected t-norm."""
        if not values:
            return 1.0
        
        if self.norm == FuzzyNorm.PRODUCT:
            # Product t-norm
            result = 1.0
            for v in values:
                result *= v
            return result
        
        elif self.norm == FuzzyNorm.LUKASIEWICZ:
            # Lukasiewicz t-norm: max(0, sum - n + 1)
            return max(0.0, sum(values) - len(values) + 1.0)
        
        elif self.norm == FuzzyNorm.GODEL:
            # Gödel (minimum) t-norm
            return min(values)
        
        return min(values)
    
    def _fuzzy_or(self, values: List[float]) -> float:
        """Fuzzy OR using selected t-conorm."""
        if not values:
            return 0.0
        
        if self.norm == FuzzyNorm.PRODUCT:
            # Product t-conorm: 1 - ∏(1 - vi)
            result = 1.0
            for v in values:
                result *= (1.0 - v)
            return 1.0 - result
        
        elif self.norm == FuzzyNorm.LUKASIEWICZ:
            # Lukasiewicz t-conorm: min(1, sum)
            return min(1.0, sum(values))
        
        elif self.norm == FuzzyNorm.GODEL:
            # Gödel (maximum) t-conorm
            return max(values)
        
        return max(values)
    
    def _apply_parameters(self, value: float) -> float:
        """Apply learnable parameters to the result."""
        weight = self.parameters.get('weight', 1.0)
        bias = self.parameters.get('bias', 0.0)
        
        # Weighted combination with sigmoid activation
        result = weight * value + bias
        
        # Clamp to [0, 1] using sigmoid
        result = 1.0 / (1.0 + math.exp(-result))
        
        return result
    
    def compute_gradient(
        self,
        tensor_space: TensorSpace,
        target_value: float,
        variable_bindings: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Compute gradients for learnable parameters.
        
        Args:
            tensor_space: Tensor space
            target_value: Target truth value
            variable_bindings: Variable bindings
        
        Returns:
            Dictionary of parameter gradients
        """
        # Evaluate current value
        current_value = self.evaluate(tensor_space, variable_bindings)
        
        # Loss gradient (MSE)
        loss_grad = 2 * (current_value - target_value)
        
        # Parameter gradients (chain rule)
        gradients = {}
        
        # Gradient w.r.t. weight
        weight = self.parameters.get('weight', 1.0)
        bias = self.parameters.get('bias', 0.0)
        
        # Compute base value before parameters
        operand_values = []
        for operand in self.operands:
            if isinstance(operand, str):
                atom_id = variable_bindings.get(operand, operand) if variable_bindings else operand
                value = tensor_space.get_value(atom_id)
                if value:
                    operand_values.append(value.strength)
                else:
                    operand_values.append(0.5)
            elif isinstance(operand, LogicTensor):
                operand_values.append(
                    operand.evaluate(tensor_space, variable_bindings)
                )
        
        base_value = self._apply_operator(operand_values)
        
        # Sigmoid derivative
        sigmoid_val = current_value
        sigmoid_grad = sigmoid_val * (1.0 - sigmoid_val)
        
        # Chain rule: loss_grad * sigmoid_grad * input_grad
        gradients['weight'] = loss_grad * sigmoid_grad * base_value
        gradients['bias'] = loss_grad * sigmoid_grad
        
        return gradients
    
    def update_parameters(self, gradients: Dict[str, float], learning_rate: float = 0.01):
        """
        Update learnable parameters using gradients.
        
        Args:
            gradients: Dictionary of parameter gradients
            learning_rate: Learning rate for updates
        """
        for param_name, grad in gradients.items():
            if param_name in self.parameters:
                self.parameters[param_name] -= learning_rate * grad
        
        self.invalidate_cache()


class SymbolicNeuralBridge:
    """
    Bridge between symbolic reasoning and neural learning.
    
    Provides:
    - Conversion between symbolic formulas and logic tensors
    - Training of logic tensors with gradient descent
    - Integration with traditional knowledge bases
    - Rule learning from data
    """
    
    def __init__(self, tensor_space: TensorSpace):
        self.tensor_space = tensor_space
        self.logic_tensors: Dict[str, LogicTensor] = {}
        self.training_history: List[Dict[str, Any]] = []
    
    def create_logic_tensor(
        self,
        name: str,
        operator: LogicOperator,
        operands: List[Union[str, LogicTensor]],
        norm: FuzzyNorm = FuzzyNorm.PRODUCT
    ) -> LogicTensor:
        """
        Create a new logic tensor.
        
        Args:
            name: Name of the logic tensor
            operator: Logical operator
            operands: List of operand atom IDs or logic tensors
            norm: Fuzzy logic norm to use
        
        Returns:
            Created LogicTensor
        """
        logic_tensor = LogicTensor(
            name=name,
            operator=operator,
            operands=operands,
            norm=norm
        )
        self.logic_tensors[name] = logic_tensor
        return logic_tensor
    
    def evaluate_formula(
        self,
        formula_name: str,
        variable_bindings: Optional[Dict[str, str]] = None
    ) -> float:
        """
        Evaluate a symbolic formula.
        
        Args:
            formula_name: Name of the logic tensor
            variable_bindings: Variable bindings
        
        Returns:
            Truth value
        """
        logic_tensor = self.logic_tensors.get(formula_name)
        if not logic_tensor:
            return 0.5
        
        return logic_tensor.evaluate(self.tensor_space, variable_bindings)
    
    def train_formula(
        self,
        formula_name: str,
        training_examples: List[Tuple[Dict[str, str], float]],
        epochs: int = 100,
        learning_rate: float = 0.01
    ) -> List[float]:
        """
        Train a logic tensor formula on examples.
        
        Args:
            formula_name: Name of the logic tensor to train
            training_examples: List of (variable_bindings, target_value) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            List of loss values per epoch
        """
        logic_tensor = self.logic_tensors.get(formula_name)
        if not logic_tensor:
            return []
        
        loss_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for bindings, target_value in training_examples:
                # Compute gradients
                gradients = logic_tensor.compute_gradient(
                    self.tensor_space,
                    target_value,
                    bindings
                )
                
                # Update parameters
                logic_tensor.update_parameters(gradients, learning_rate)
                
                # Compute loss
                predicted = logic_tensor.evaluate(self.tensor_space, bindings)
                epoch_loss += (predicted - target_value) ** 2
            
            avg_loss = epoch_loss / len(training_examples) if training_examples else 0.0
            loss_history.append(avg_loss)
        
        self.training_history.append({
            'formula_name': formula_name,
            'epochs': epochs,
            'final_loss': loss_history[-1] if loss_history else 0.0
        })
        
        return loss_history
    
    def learn_rule(
        self,
        rule_name: str,
        antecedent_atoms: List[str],
        consequent_atoms: List[str],
        training_data: List[Dict[str, Any]],
        operator: LogicOperator = LogicOperator.IMPLIES
    ) -> LogicTensor:
        """
        Learn a logical rule from data.
        
        Args:
            rule_name: Name for the learned rule
            antecedent_atoms: Atoms in the antecedent
            consequent_atoms: Atoms in the consequent
            training_data: Training examples
            operator: Logical operator (default: IMPLIES)
        
        Returns:
            Learned LogicTensor rule
        """
        # Create antecedent (AND of all antecedent atoms)
        if len(antecedent_atoms) > 1:
            antecedent = LogicTensor(
                name=f"{rule_name}_antecedent",
                operator=LogicOperator.AND,
                operands=antecedent_atoms
            )
        else:
            antecedent = antecedent_atoms[0] if antecedent_atoms else ""
        
        # Create consequent (AND of all consequent atoms)
        if len(consequent_atoms) > 1:
            consequent = LogicTensor(
                name=f"{rule_name}_consequent",
                operator=LogicOperator.AND,
                operands=consequent_atoms
            )
        else:
            consequent = consequent_atoms[0] if consequent_atoms else ""
        
        # Create rule (antecedent → consequent)
        rule = LogicTensor(
            name=rule_name,
            operator=operator,
            operands=[antecedent, consequent] if isinstance(antecedent, LogicTensor) and isinstance(consequent, LogicTensor)
                    else [antecedent_atoms[0] if antecedent_atoms else "", consequent_atoms[0] if consequent_atoms else ""]
        )
        
        self.logic_tensors[rule_name] = rule
        
        # Train the rule on data
        training_examples = []
        for example in training_data:
            bindings = example.get('bindings', {})
            target = example.get('target', 1.0)
            training_examples.append((bindings, target))
        
        if training_examples:
            self.train_formula(rule_name, training_examples)
        
        return rule
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the bridge."""
        return {
            'total_logic_tensors': len(self.logic_tensors),
            'logic_tensor_names': list(self.logic_tensors.keys()),
            'training_sessions': len(self.training_history),
            'tensor_space_stats': self.tensor_space.get_statistics()
        }
