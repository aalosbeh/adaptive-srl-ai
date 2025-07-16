
### Prerequisites

- Python 3.8 or higher
- Git
- PyTorch 1.12 or higher

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/aslosbeh/adaptive-srl-ai.git
cd adaptive-srl-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Install the package in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_federated_agent.py

# Run with coverage
pytest --cov=adaptive_srl_ai

# Run integration tests
pytest tests/integration/
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Run all quality checks
pre-commit run --all-files
```

## Coding Standards

### Example Code Style

```python
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from adaptive_srl_ai.core.base import BaseAgent


class ExampleAgent(BaseAgent):
    """Example agent demonstrating coding standards.
    
    This class shows the expected code style including proper
    docstrings, type hints, and formatting.
    
    Args:
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        learning_rate: Learning rate for optimization
        device: Device for computation ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Initialize networks
        self.policy_network = self._build_policy_network()
        self.optimizer = Adam(self.policy_network.parameters(), lr=learning_rate)
    
    def _build_policy_network(self) -> nn.Module:
        """Build the policy network architecture.
        
        Returns:
            PyTorch neural network module
        """
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Softmax(dim=-1)
        )
    
    def select_action(
        self, 
        state: torch.Tensor, 
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select action based on current state.
        
        Args:
            state: Current state tensor
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_probability)
        """
        with torch.no_grad() if not training else torch.enable_grad():
            action_probs = self.policy_network(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
        return action, log_prob
```

### Documentation Standards

#### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: int, param2: str, param3: Optional[bool] = None) -> Dict[str, Any]:
    """Brief description of the function.
    
    Longer description if needed. This can span multiple lines
    and provide more detailed information about the function's
    purpose and behavior.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        param3: Description of optional param3. Defaults to None.
        
    Returns:
        Dictionary containing the results with keys:
            - 'result': The main result
            - 'metadata': Additional information
            
    Raises:
        ValueError: If param1 is negative
        TypeError: If param2 is not a string
        
    Example:
        >>> result = example_function(42, "hello")
        >>> print(result['result'])
        42
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    
    return {
        'result': param1,
        'metadata': {'param2': param2, 'param3': param3}
    }
```

### Test Structure

Organize tests in the `tests/` directory:

```
tests/
├── unit/                 # Unit tests
│   ├── test_agents.py
│   ├── test_estimators.py
│   └── test_utils.py
├── integration/          # Integration tests
│   ├── test_federated_learning.py
│   └── test_end_to_end.py
├── fixtures/            # Test fixtures and data
│   ├── sample_data.py
│   └── mock_models.py
└── conftest.py          # Pytest configuration
```

### Writing Tests

```python
import pytest
import torch
from unittest.mock import Mock, patch

from adaptive_srl_ai.core import FederatedDRLAgent


class TestFederatedDRLAgent:
    """Test suite for FederatedDRLAgent."""
    
    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        return FederatedDRLAgent(
            state_dim=10,
            action_dim=4,
            hidden_dim=64,
            learning_rate=1e-3
        )
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.state_dim == 10
        assert agent.action_dim == 4
        assert agent.hidden_dim == 64
        assert agent.learning_rate == 1e-3
    
    def test_action_selection(self, agent):
        """Test action selection functionality."""
        state = torch.randn(10)
        action, log_prob = agent.select_action(state)
        
        assert action.shape == torch.Size([])
        assert log_prob.shape == torch.Size([])
        assert 0 <= action.item() < 4
    
    @patch('adaptive_srl_ai.core.federated_drl_agent.torch.save')
    def test_save_model(self, mock_save, agent):
        """Test model saving functionality."""
        agent.save_model("test_path.pth")
        mock_save.assert_called_once()
    
    def test_invalid_state_dim(self):
        """Test error handling for invalid state dimension."""
        with pytest.raises(ValueError):
            FederatedDRLAgent(state_dim=-1, action_dim=4)
```

### Memory Management

```python
# Good: Use context managers for temporary computations
with torch.no_grad():
    predictions = model(batch_data)

# Good: Delete large tensors when no longer needed
del large_tensor
torch.cuda.empty_cache()  # If using GPU

# Good: Use generators for large datasets
def data_generator():
    for batch in dataset:
        yield process_batch(batch)
```

### GPU Utilization

```python
# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
else:
    device = torch.device("cpu")

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input_data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Security Best Practices

```python
# Good: Validate inputs
def process_user_input(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process user input with validation."""
    required_fields = ['learner_id', 'timestamp', 'features']
    
    for field in required_fields:
        if field not in user_data:
            raise ValueError(f"Missing required field: {field}")
    
    # Sanitize and validate data
    learner_id = str(user_data['learner_id'])[:50]  # Limit length
    timestamp = float(user_data['timestamp'])
    
    return {
        'learner_id': learner_id,
        'timestamp': timestamp,
        'features': user_data['features']
    }
```

## Research Contributions

### Adding New Algorithms
When contributing new algorithms:

1. **Literature Review**: Ensure the algorithm is novel or significantly improved
2. **Theoretical Foundation**: Provide mathematical formulation
3. **Implementation**: Follow coding standards and include tests
4. **Evaluation**: Compare against existing baselines
5. **Documentation**: Include detailed explanation and examples

### Dataset Contributions

When contributing datasets:

1. **Privacy Compliance**: Ensure all privacy requirements are met
2. **Data Quality**: Validate data integrity and completeness
3. **Documentation**: Provide detailed dataset description
4. **Licensing**: Ensure appropriate licensing for sharing
5. **Reproducibility**: Include data generation scripts if synthetic

### Experimental Contributions

When contributing experiments:

1. **Reproducibility**: Include all code and configuration files
2. **Statistical Rigor**: Use appropriate statistical tests
3. **Baseline Comparisons**: Compare against relevant baselines
4. **Ablation Studies**: Analyze component contributions
5. **Visualization**: Create clear, informative plots

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Actual Behavior**
A clear description of what actually happened.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python Version: [e.g. 3.8.10]
- PyTorch Version: [e.g. 1.12.0]
- CUDA Version: [e.g. 11.6]
- Package Version: [e.g. 1.0.0]

**Additional Context**
Add any other context about the problem here.

**Code Example**
```python
# Minimal code example that reproduces the bug
import adaptive_srl_ai
# ... rest of the example
```

