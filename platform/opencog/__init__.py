"""
OpenCog Atomspace Integration for Stripe Ecosystem

This module provides knowledge representation and reasoning capabilities
using OpenCog's Atomspace for the unified Stripe platform.
"""

from .atomspace import StripeAtomspace, AtomType
from .knowledge_base import StripeKnowledgeBase
from .reasoning import StripeReasoner

__all__ = [
    'StripeAtomspace',
    'AtomType',
    'StripeKnowledgeBase',
    'StripeReasoner',
]
