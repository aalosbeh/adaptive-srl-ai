"""
Adaptive Multi-Modal AI Framework for Self-Regulated Learning

This package provides a comprehensive framework for implementing federated deep reinforcement
learning approaches to support personalized self-regulated learning in educational and
workplace settings.

Key Components:
- Federated Deep Reinforcement Learning
- Multi-Modal Learning Analytics
- Metacognitive State Estimation
- Privacy-Preserving AI
- Cross-Domain Transfer Learning

Authors: Manus AI Research Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Manus AI Research Team"
__email__ = "research@manus.ai"
__license__ = "MIT"

# Core modules
from .core import (
    FederatedDRLAgent,
    MetacognitiveEstimator,
    MultiModalFusion,
    PrivacyPreserver,
    KnowledgeGraph,
)

# Models
from .models import (
    SRLStateEncoder,
    PolicyNetwork,
    ValueNetwork,
    AttentionFusion,
    TemporalConvNet,
    GraphNeuralNet,
)

# Federated learning
from .federated import (
    FederatedLearningCoordinator,
    LocalClient,
    PrivacyEngine,
    ModelAggregator,
)

# Data processing
from .data import (
    SRLDataset,
    MultiModalDataLoader,
    DataPreprocessor,
    SyntheticDataGenerator,
)

# Evaluation
from .evaluation import (
    SRLMetrics,
    PerformanceEvaluator,
    PrivacyAnalyzer,
    TransferLearningEvaluator,
)

# Utilities
from .utils import (
    ConfigManager,
    Logger,
    Visualizer,
    ModelCheckpoint,
)

__all__ = [
    # Core
    "FederatedDRLAgent",
    "MetacognitiveEstimator", 
    "MultiModalFusion",
    "PrivacyPreserver",
    "KnowledgeGraph",
    # Models
    "SRLStateEncoder",
    "PolicyNetwork",
    "ValueNetwork",
    "AttentionFusion",
    "TemporalConvNet",
    "GraphNeuralNet",
    # Federated
    "FederatedLearningCoordinator",
    "LocalClient",
    "PrivacyEngine",
    "ModelAggregator",
    # Data
    "SRLDataset",
    "MultiModalDataLoader",
    "DataPreprocessor",
    "SyntheticDataGenerator",
    # Evaluation
    "SRLMetrics",
    "PerformanceEvaluator",
    "PrivacyAnalyzer",
    "TransferLearningEvaluator",
    # Utils
    "ConfigManager",
    "Logger",
    "Visualizer",
    "ModelCheckpoint",
]

# Package metadata
__package_info__ = {
    "name": "adaptive-srl-ai",
    "version": __version__,
    "description": "Adaptive Multi-Modal AI Framework for Personalized Self-Regulated Learning",
    "url": "https://github.com/manus-ai/adaptive-srl-ai",
    "author": __author__,
    "author_email": __email__,
    "license": __license__,
    "keywords": [
        "artificial intelligence",
        "education",
        "self-regulated learning", 
        "federated learning",
        "deep reinforcement learning",
        "multi-modal AI",
        "privacy-preserving",
        "personalized learning",
    ],
}

