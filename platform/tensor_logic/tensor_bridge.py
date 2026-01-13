"""
Tensor Logic Bridge

Integrates Tensor Logic with the existing OpenCog platform,
enabling hybrid neural-symbolic reasoning for the Stripe ecosystem.
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from opencog.atomspace import StripeAtomspace, AtomType
from opencog.reasoning import StripeReasoner, ReasoningResult, ReasoningType
from opencog.knowledge_base import StripeKnowledgeBase

from .tensor_space import TensorSpace, TensorAtomValue
from .tensor_atoms import TensorAtomspace, TensorNode, TensorLink
from .symbolic_integration import SymbolicNeuralBridge, LogicTensor, LogicOperator
from .gradient_reasoner import GradientReasoner, TensorInferenceRule


class TensorLogicBridge:
    """
    Bridge between Tensor Logic and OpenCog platforms.
    
    Provides:
    - Conversion between traditional and tensor atomspaces
    - Hybrid reasoning (symbolic + neural)
    - Knowledge transfer between systems
    - Unified inference interface
    """
    
    def __init__(
        self,
        traditional_atomspace: Optional[StripeAtomspace] = None,
        knowledge_base: Optional[StripeKnowledgeBase] = None,
        reasoner: Optional[StripeReasoner] = None,
        learning_rate: float = 0.01
    ):
        # Traditional OpenCog components
        self.traditional_atomspace = traditional_atomspace or StripeAtomspace()
        self.knowledge_base = knowledge_base or StripeKnowledgeBase(self.traditional_atomspace)
        self.reasoner = reasoner or StripeReasoner(self.traditional_atomspace, self.knowledge_base)
        
        # Tensor Logic components
        self.tensor_atomspace = TensorAtomspace(
            name="tensor_bridge_atomspace",
            learning_rate=learning_rate
        )
        self.tensor_space = self.tensor_atomspace.tensor_space
        self.symbolic_bridge = SymbolicNeuralBridge(self.tensor_space)
        self.gradient_reasoner = GradientReasoner(
            self.tensor_atomspace,
            self.symbolic_bridge
        )
        
        self.logger = logging.getLogger(__name__)
        self._initialized = False
    
    def initialize(self):
        """Initialize the bridge and sync knowledge bases."""
        if self._initialized:
            return
        
        # Initialize knowledge base
        if not self.knowledge_base._initialized:
            self.knowledge_base.initialize()
        
        # Sync traditional atomspace to tensor atomspace
        self._sync_to_tensor_atomspace()
        
        self._initialized = True
        self.logger.info("TensorLogicBridge initialized successfully")
    
    def _sync_to_tensor_atomspace(self):
        """Sync atoms from traditional to tensor atomspace."""
        for atom_id, atom in self.traditional_atomspace.atoms.items():
            # Convert to tensor atom
            if hasattr(atom, 'outgoing'):
                # It's a link
                # First ensure outgoing atoms exist in tensor atomspace
                tensor_outgoing = []
                for outgoing_atom in atom.outgoing:
                    tensor_atom = self.tensor_atomspace.atoms.get(outgoing_atom.id)
                    if not tensor_atom:
                        # Create tensor version
                        if hasattr(outgoing_atom, 'value'):
                            tensor_atom = self.tensor_atomspace.add_tensor_node(
                                outgoing_atom.atom_type,
                                outgoing_atom.name,
                                value=outgoing_atom.value,
                                strength=outgoing_atom.truth_value[0],
                                confidence=outgoing_atom.truth_value[1],
                                metadata=outgoing_atom.metadata
                            )
                    if tensor_atom:
                        tensor_outgoing.append(tensor_atom)
                
                # Create tensor link
                if tensor_outgoing:
                    self.tensor_atomspace.add_tensor_link(
                        atom.atom_type,
                        tensor_outgoing,
                        name=atom.name,
                        strength=atom.truth_value[0],
                        confidence=atom.truth_value[1],
                        metadata=atom.metadata
                    )
            else:
                # It's a node
                if hasattr(atom, 'value'):
                    self.tensor_atomspace.add_tensor_node(
                        atom.atom_type,
                        atom.name,
                        value=atom.value,
                        strength=atom.truth_value[0],
                        confidence=atom.truth_value[1],
                        metadata=atom.metadata
                    )
    
    def _sync_from_tensor_atomspace(self):
        """Sync learned values back to traditional atomspace."""
        for atom_id, tensor_atom in self.tensor_atomspace.atoms.items():
            if isinstance(tensor_atom, (TensorNode, TensorLink)):
                tensor_value = tensor_atom.get_tensor_value()
                if tensor_value:
                    # Update traditional atom if it exists
                    traditional_atom = self.traditional_atomspace.atoms.get(atom_id)
                    if traditional_atom:
                        self.traditional_atomspace.update_truth_value(
                            traditional_atom,
                            tensor_value.strength,
                            tensor_value.confidence
                        )
    
    def hybrid_reason(
        self,
        query: str,
        use_neural: bool = True,
        use_symbolic: bool = True,
        max_iterations: int = 5
    ) -> ReasoningResult:
        """
        Perform hybrid reasoning using both neural and symbolic approaches.
        
        Args:
            query: Query string
            use_neural: Use neural/gradient-based reasoning
            use_symbolic: Use traditional symbolic reasoning
            max_iterations: Maximum inference iterations
        
        Returns:
            ReasoningResult combining both approaches
        """
        neural_results = []
        symbolic_results = []
        
        # Neural reasoning with gradient-based inference
        if use_neural:
            chain_results = self.gradient_reasoner.forward_chain(
                max_iterations=max_iterations
            )
            for rule_name, fired, confidence in chain_results:
                if fired:
                    neural_results.append(
                        f"Neural rule '{rule_name}' fired with confidence {confidence:.3f}"
                    )
        
        # Symbolic reasoning
        if use_symbolic:
            # Run traditional forward chaining
            symbolic_atoms = self.reasoner.run_forward_chaining()
            for atom in symbolic_atoms:
                symbolic_results.append(
                    f"Symbolic inference created: {atom.name}"
                )
        
        # Combine results
        all_evidence = neural_results + symbolic_results
        
        # Determine conclusion
        if neural_results and symbolic_results:
            conclusion = f"Hybrid reasoning: Neural found {len(neural_results)} patterns, Symbolic created {len(symbolic_atoms)} atoms"
            from opencog.reasoning import ConfidenceLevel
            confidence = ConfidenceLevel.HIGH
        elif neural_results:
            conclusion = f"Neural reasoning: Found {len(neural_results)} patterns"
            from opencog.reasoning import ConfidenceLevel
            confidence = ConfidenceLevel.MEDIUM
        elif symbolic_results:
            conclusion = f"Symbolic reasoning: Created {len(symbolic_atoms)} atoms"
            from opencog.reasoning import ConfidenceLevel
            confidence = ConfidenceLevel.MEDIUM
        else:
            conclusion = "No conclusions drawn"
            from opencog.reasoning import ConfidenceLevel
            confidence = ConfidenceLevel.UNCERTAIN
        
        return ReasoningResult(
            query=query,
            reasoning_type=ReasoningType.DEDUCTION,
            conclusion=conclusion,
            confidence=confidence,
            evidence=all_evidence,
            metadata={
                'neural_patterns': len(neural_results),
                'symbolic_atoms': len(symbolic_results)
            }
        )
    
    def learn_from_stripe_data(
        self,
        payment_data: List[Dict[str, Any]],
        epochs: int = 100
    ) -> Dict[str, Any]:
        """
        Learn patterns from Stripe payment data.
        
        Args:
            payment_data: List of payment dictionaries
            epochs: Number of training epochs
        
        Returns:
            Learning statistics
        """
        training_data = []
        
        # Convert payment data to training examples
        for payment in payment_data:
            payment_id = payment.get('id', '')
            amount = payment.get('amount', 0)
            status = payment.get('status', '')
            
            # Create or get payment atom
            payment_atom = self.tensor_atomspace.add_tensor_node(
                AtomType.PAYMENT,
                payment_id,
                value={'amount': amount, 'status': status}
            )
            
            # Determine target values based on status
            if status == 'succeeded':
                target_strength = 0.9
                target_confidence = 0.9
            elif status == 'failed':
                target_strength = 0.1
                target_confidence = 0.8
            else:
                target_strength = 0.5
                target_confidence = 0.5
            
            training_data.append((payment_atom.id, target_strength, target_confidence))
        
        # Train the tensor atomspace
        loss_history = self.tensor_atomspace.train_atoms(
            training_data,
            epochs=epochs,
            batch_size=32
        )
        
        # Sync learned values back
        self._sync_from_tensor_atomspace()
        
        return {
            'payments_processed': len(payment_data),
            'epochs_trained': epochs,
            'final_loss': loss_history[-1] if loss_history else 0.0,
            'loss_history': loss_history[-10:]  # Last 10 losses
        }
    
    def learn_risk_assessment_rule(
        self,
        training_examples: List[Dict[str, Any]],
        epochs: int = 100
    ) -> TensorInferenceRule:
        """
        Learn a risk assessment rule from examples.
        
        Args:
            training_examples: Training data with risk assessments
            epochs: Number of epochs
        
        Returns:
            Learned risk assessment rule
        """
        # Extract patterns from examples
        antecedent_atoms = ['high_amount', 'unknown_customer', 'suspicious_ip']
        consequent_atoms = ['high_risk']
        
        # Create atoms if they don't exist
        for atom_name in antecedent_atoms + consequent_atoms:
            if not self.tensor_atomspace.get_node(AtomType.CONCEPT, atom_name):
                self.tensor_atomspace.add_tensor_node(
                    AtomType.CONCEPT,
                    atom_name,
                    strength=0.5,
                    confidence=0.5
                )
        
        # Learn the rule
        rule = self.gradient_reasoner.learn_inference_rule(
            rule_name="risk_assessment_rule",
            training_examples=training_examples,
            antecedent_atoms=antecedent_atoms,
            consequent_atoms=consequent_atoms,
            epochs=epochs,
            learning_rate=0.01
        )
        
        return rule
    
    def assess_payment_risk_hybrid(
        self,
        payment_id: str,
        payment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess payment risk using hybrid reasoning.
        
        Args:
            payment_id: Payment identifier
            payment_data: Payment data
        
        Returns:
            Risk assessment result
        """
        # Traditional symbolic reasoning
        symbolic_result = self.reasoner.assess_payment_risk(payment_id)
        
        # Create tensor atoms for this payment
        amount = payment_data.get('amount', 0)
        customer_id = payment_data.get('customer_id', '')
        
        payment_atom = self.tensor_atomspace.add_tensor_node(
            AtomType.PAYMENT,
            payment_id,
            value={'amount': amount},
            strength=0.5,
            confidence=0.5
        )
        
        # Neural reasoning with gradient-based inference
        if amount > 100000:  # High amount
            self.tensor_atomspace.add_tensor_node(
                AtomType.CONCEPT,
                'high_amount',
                strength=0.9,
                confidence=0.9
            )
        
        # Run gradient-based reasoning
        neural_results = self.gradient_reasoner.forward_chain(max_iterations=3)
        
        # Combine results
        neural_risk = 0.0
        for rule_name, fired, confidence in neural_results:
            if 'risk' in rule_name.lower() and fired:
                neural_risk = max(neural_risk, confidence)
        
        symbolic_risk = symbolic_result.metadata.get('risk_score', 0.0)
        
        # Weighted combination
        combined_risk = 0.6 * neural_risk + 0.4 * symbolic_risk
        
        return {
            'payment_id': payment_id,
            'combined_risk_score': combined_risk,
            'neural_risk': neural_risk,
            'symbolic_risk': symbolic_risk,
            'symbolic_conclusion': symbolic_result.conclusion,
            'neural_patterns': len([r for r in neural_results if r[1]]),
            'recommendation': 'high_risk' if combined_risk > 0.7 else 'low_risk'
        }
    
    def export_learned_knowledge(self) -> Dict[str, Any]:
        """Export all learned knowledge from tensor atomspace."""
        return {
            'tensor_values': self.tensor_atomspace.export_tensor_values(),
            'logic_tensors': self.symbolic_bridge.get_statistics(),
            'inference_rules': self.gradient_reasoner.get_statistics(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'traditional_atomspace': self.traditional_atomspace.get_statistics(),
            'tensor_atomspace': self.tensor_atomspace.get_tensor_statistics(),
            'symbolic_bridge': self.symbolic_bridge.get_statistics(),
            'gradient_reasoner': self.gradient_reasoner.get_statistics(),
            'initialized': self._initialized
        }
