"""
Tensor Logic Framework

A deep unification of deep learning and symbolic AI that combines the scalability
and gradient-based learning of neural networks with the transparency and reliability
of symbolic knowledge representation and reasoning.

Based on: https://tensor-logic.org/ and https://arxiv.org/abs/2510.12269
"""

from .tensor_space import TensorSpace, TensorAtomValue
from .symbolic_integration import SymbolicNeuralBridge, LogicTensor
from .tensor_atoms import TensorAtom, TensorNode, TensorLink
from .gradient_reasoner import GradientReasoner, TensorInferenceRule
from .tensor_bridge import TensorLogicBridge

__all__ = [
    'TensorSpace',
    'TensorAtomValue',
    'SymbolicNeuralBridge',
    'LogicTensor',
    'TensorAtom',
    'TensorNode',
    'TensorLink',
    'GradientReasoner',
    'TensorInferenceRule',
    'TensorLogicBridge',
]

__version__ = '1.0.0'
