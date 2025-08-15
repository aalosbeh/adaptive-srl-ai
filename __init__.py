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

