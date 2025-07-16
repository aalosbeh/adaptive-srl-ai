"""
Policy Network for Deep Reinforcement Learning

This module implements the policy network that learns to select optimal
interventions for self-regulated learning support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Policy Network for SRL Intervention Selection
    
    This network learns a policy that maps encoded learner states to
    probability distributions over possible interventions.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize the Policy Network
        
        Args:
            input_dim: Dimension of input state encoding
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function type
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (action probabilities)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network
        
        Args:
            state_encoding: Encoded learner state [batch_size, input_dim]
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        x = state_encoding
        
        # Input layer
        x = self.input_layer(x)
        x = self.layer_norms[0](x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Hidden layers
        for i, (hidden_layer, layer_norm) in enumerate(zip(self.hidden_layers, self.layer_norms[1:])):
            residual = x
            x = hidden_layer(x)
            x = layer_norm(x)
            x = self.activation(x)
            x = self.dropout(x)
            
            # Residual connection for deeper networks
            if x.shape == residual.shape:
                x = x + residual
        
        # Output layer
        logits = self.output_layer(x)
        
        # Apply softmax to get probabilities
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs
    
    def get_action_distribution(self, state_encoding: torch.Tensor) -> torch.distributions.Categorical:
        """
        Get action distribution for sampling
        
        Args:
            state_encoding: Encoded learner state
            
        Returns:
            Categorical distribution over actions
        """
        action_probs = self.forward(state_encoding)
        return torch.distributions.Categorical(action_probs)
    
    def sample_action(self, state_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state_encoding: Encoded learner state
            
        Returns:
            Tuple of (sampled_action, log_probability)
        """
        action_dist = self.get_action_distribution(state_encoding)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action, log_prob
    
    def get_log_prob(self, state_encoding: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of given actions
        
        Args:
            state_encoding: Encoded learner state
            actions: Actions to evaluate
            
        Returns:
            Log probabilities of actions
        """
        action_dist = self.get_action_distribution(state_encoding)
        return action_dist.log_prob(actions)
    
    def get_entropy(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """
        Get entropy of action distribution (for exploration bonus)
        
        Args:
            state_encoding: Encoded learner state
            
        Returns:
            Entropy of action distribution
        """
        action_dist = self.get_action_distribution(state_encoding)
        return action_dist.entropy()


class MultiHeadPolicyNetwork(nn.Module):
    """
    Multi-Head Policy Network for different intervention types
    
    This network has separate heads for different types of interventions
    (content, strategy, feedback, social) allowing for more specialized policies.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dims: dict,  # {"content": 4, "strategy": 4, "feedback": 4, "social": 4}
        hidden_dim: int = 256,
        shared_layers: int = 2,
        head_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize Multi-Head Policy Network
        
        Args:
            input_dim: Dimension of input state encoding
            action_dims: Dictionary mapping intervention types to action dimensions
            hidden_dim: Hidden layer dimension
            shared_layers: Number of shared layers
            head_layers: Number of layers per head
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dims = action_dims
        self.hidden_dim = hidden_dim
        
        # Shared backbone
        shared_layers_list = []
        shared_layers_list.append(nn.Linear(input_dim, hidden_dim))
        shared_layers_list.append(nn.ReLU())
        shared_layers_list.append(nn.Dropout(dropout))
        
        for _ in range(shared_layers - 1):
            shared_layers_list.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.shared_backbone = nn.Sequential(*shared_layers_list)
        
        # Intervention-specific heads
        self.heads = nn.ModuleDict()
        for intervention_type, action_dim in action_dims.items():
            head_layers_list = []
            
            for i in range(head_layers):
                if i == 0:
                    head_layers_list.append(nn.Linear(hidden_dim, hidden_dim // 2))
                else:
                    head_layers_list.append(nn.Linear(hidden_dim // 2, hidden_dim // 2))
                
                if i < head_layers - 1:
                    head_layers_list.extend([
                        nn.LayerNorm(hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
            
            # Output layer for this head
            head_layers_list.append(nn.Linear(hidden_dim // 2, action_dim))
            
            self.heads[intervention_type] = nn.Sequential(*head_layers_list)
        
        # Intervention type selector
        self.intervention_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, len(action_dims)),
            nn.Softmax(dim=-1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state_encoding: torch.Tensor) -> dict:
        """
        Forward pass through multi-head policy network
        
        Args:
            state_encoding: Encoded learner state
            
        Returns:
            Dictionary of action probabilities for each intervention type
        """
        # Shared feature extraction
        shared_features = self.shared_backbone(state_encoding)
        
        # Intervention type probabilities
        intervention_probs = self.intervention_selector(shared_features)
        
        # Head-specific action probabilities
        head_outputs = {}
        for intervention_type, head in self.heads.items():
            logits = head(shared_features)
            action_probs = F.softmax(logits, dim=-1)
            head_outputs[intervention_type] = action_probs
        
        return {
            "intervention_probs": intervention_probs,
            "action_probs": head_outputs
        }
    
    def sample_hierarchical_action(self, state_encoding: torch.Tensor) -> dict:
        """
        Sample action using hierarchical policy
        
        Args:
            state_encoding: Encoded learner state
            
        Returns:
            Dictionary containing intervention type and specific action
        """
        outputs = self.forward(state_encoding)
        
        # Sample intervention type
        intervention_dist = torch.distributions.Categorical(outputs["intervention_probs"])
        intervention_idx = intervention_dist.sample()
        intervention_type = list(self.action_dims.keys())[intervention_idx]
        
        # Sample specific action for chosen intervention type
        action_probs = outputs["action_probs"][intervention_type]
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return {
            "intervention_type": intervention_type,
            "intervention_idx": intervention_idx,
            "action": action,
            "intervention_log_prob": intervention_dist.log_prob(intervention_idx),
            "action_log_prob": action_dist.log_prob(action)
        }


class AttentionPolicyNetwork(nn.Module):
    """
    Attention-based Policy Network
    
    Uses attention mechanisms to focus on relevant aspects of the learner state
    when making intervention decisions.
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize Attention-based Policy Network
        
        Args:
            input_dim: Dimension of input state encoding
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding (for sequence modeling if needed)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention-based policy network
        
        Args:
            state_encoding: Encoded learner state [batch_size, input_dim]
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        batch_size = state_encoding.shape[0]
        
        # Project input to hidden dimension
        x = self.input_projection(state_encoding)  # [batch_size, hidden_dim]
        
        # Add sequence dimension and positional encoding
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x = x + self.positional_encoding
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, 1, hidden_dim]
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # Output projection
        logits = self.output_projection(x)
        
        # Apply softmax
        action_probs = F.softmax(logits, dim=-1)
        
        return action_probs

