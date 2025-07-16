"""
Federated Deep Reinforcement Learning Agent for Self-Regulated Learning

This module implements the core federated DRL agent that coordinates learning
across multiple educational institutions while preserving privacy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import deque
import copy

from ..models.policy_network import PolicyNetwork
from ..models.value_network import ValueNetwork
from ..models.srl_state_encoder import SRLStateEncoder
from ..privacy.differential_privacy import DifferentialPrivacy
from ..utils.logger import Logger


@dataclass
class SRLState:
    """Self-Regulated Learning State representation"""
    metacognitive: torch.Tensor  # Metacognitive awareness, monitoring, control
    cognitive: torch.Tensor      # Attention, memory, processing
    behavioral: torch.Tensor     # Engagement, persistence, strategy use
    emotional: torch.Tensor      # Motivation, anxiety, confidence
    knowledge: torch.Tensor      # Domain knowledge, skill level
    timestamp: float
    learner_id: str


@dataclass
class SRLAction:
    """Self-Regulated Learning Action representation"""
    content_recommendation: torch.Tensor
    strategy_suggestion: torch.Tensor
    feedback_delivery: torch.Tensor
    social_facilitation: torch.Tensor
    intervention_type: str
    confidence: float


@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: SRLState
    action: SRLAction
    reward: float
    next_state: SRLState
    done: bool
    advantage: Optional[float] = None


class FederatedDRLAgent:
    """
    Federated Deep Reinforcement Learning Agent for Self-Regulated Learning
    
    This agent implements a federated learning approach where multiple educational
    institutions can collaboratively train a shared model while keeping their
    data private and secure.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        privacy_budget: float = 1.0,
        device: str = "cpu",
        institution_id: str = "default",
        **kwargs
    ):
        """
        Initialize the Federated DRL Agent
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Hidden layer dimension for neural networks
            learning_rate: Learning rate for optimization
            gamma: Discount factor for future rewards
            epsilon: Exploration parameter for epsilon-greedy
            privacy_budget: Privacy budget for differential privacy
            device: Computing device (cpu/cuda)
            institution_id: Unique identifier for the institution
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = torch.device(device)
        self.institution_id = institution_id
        
        # Initialize logger
        self.logger = Logger(f"FederatedDRL_{institution_id}")
        
        # Initialize neural networks
        self.state_encoder = SRLStateEncoder(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            device=device
        ).to(self.device)
        
        self.policy_network = PolicyNetwork(
            input_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.value_network = ValueNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            list(self.state_encoder.parameters()) + 
            list(self.policy_network.parameters()),
            lr=learning_rate
        )
        
        self.value_optimizer = optim.Adam(
            list(self.state_encoder.parameters()) + 
            list(self.value_network.parameters()),
            lr=learning_rate
        )
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        
        # Privacy preservation
        self.privacy_engine = DifferentialPrivacy(
            epsilon=privacy_budget,
            delta=1e-5,
            sensitivity=1.0
        )
        
        # Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_reward": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "privacy_cost": 0.0
        }
        
        self.logger.info(f"Initialized FederatedDRLAgent for institution {institution_id}")
    
    def encode_state(self, state: SRLState) -> torch.Tensor:
        """
        Encode the SRL state into a latent representation
        
        Args:
            state: SRL state object
            
        Returns:
            Encoded state tensor
        """
        # Concatenate all state components
        state_vector = torch.cat([
            state.metacognitive,
            state.cognitive,
            state.behavioral,
            state.emotional,
            state.knowledge
        ], dim=-1)
        
        # Encode using the state encoder
        encoded_state = self.state_encoder(state_vector)
        return encoded_state
    
    def select_action(
        self, 
        state: SRLState, 
        training: bool = True
    ) -> Tuple[SRLAction, torch.Tensor]:
        """
        Select an action based on the current state
        
        Args:
            state: Current SRL state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action and action probabilities
        """
        encoded_state = self.encode_state(state)
        
        with torch.no_grad():
            action_probs = self.policy_network(encoded_state)
            
            if training and np.random.random() < self.epsilon:
                # Exploration: random action
                action_idx = np.random.randint(0, self.action_dim)
                action_tensor = torch.zeros(self.action_dim)
                action_tensor[action_idx] = 1.0
            else:
                # Exploitation: sample from policy
                action_dist = torch.distributions.Categorical(action_probs)
                action_idx = action_dist.sample()
                action_tensor = torch.zeros(self.action_dim)
                action_tensor[action_idx] = 1.0
        
        # Convert to SRL action format
        action = self._tensor_to_srl_action(action_tensor, action_probs)
        
        return action, action_probs
    
    def _tensor_to_srl_action(
        self, 
        action_tensor: torch.Tensor, 
        action_probs: torch.Tensor
    ) -> SRLAction:
        """Convert action tensor to SRL action object"""
        # Split action tensor into components
        quarter_dim = self.action_dim // 4
        
        content_rec = action_tensor[:quarter_dim]
        strategy_sug = action_tensor[quarter_dim:2*quarter_dim]
        feedback_del = action_tensor[2*quarter_dim:3*quarter_dim]
        social_fac = action_tensor[3*quarter_dim:]
        
        # Determine intervention type
        intervention_types = ["content", "strategy", "feedback", "social"]
        intervention_idx = torch.argmax(action_tensor).item() // quarter_dim
        intervention_type = intervention_types[min(intervention_idx, 3)]
        
        # Calculate confidence
        confidence = torch.max(action_probs).item()
        
        return SRLAction(
            content_recommendation=content_rec,
            strategy_suggestion=strategy_sug,
            feedback_delivery=feedback_del,
            social_facilitation=social_fac,
            intervention_type=intervention_type,
            confidence=confidence
        )
    
    def compute_reward(
        self, 
        state: SRLState, 
        action: SRLAction, 
        next_state: SRLState
    ) -> float:
        """
        Compute reward for the state-action-next_state transition
        
        Args:
            state: Current state
            action: Taken action
            next_state: Resulting state
            
        Returns:
            Computed reward value
        """
        # Multi-objective reward function
        performance_reward = self._compute_performance_reward(state, next_state)
        engagement_reward = self._compute_engagement_reward(state, next_state)
        metacognitive_reward = self._compute_metacognitive_reward(state, next_state)
        efficiency_reward = self._compute_efficiency_reward(state, action, next_state)
        
        # Weighted combination
        total_reward = (
            0.4 * performance_reward +
            0.3 * engagement_reward +
            0.2 * metacognitive_reward +
            0.1 * efficiency_reward
        )
        
        return total_reward
    
    def _compute_performance_reward(self, state: SRLState, next_state: SRLState) -> float:
        """Compute performance improvement reward"""
        # Extract knowledge components
        current_knowledge = torch.mean(state.knowledge).item()
        next_knowledge = torch.mean(next_state.knowledge).item()
        
        # Performance improvement
        delta_performance = next_knowledge - current_knowledge
        
        # Normalized reward with target-based exponential decay
        target_performance = 0.8  # Target knowledge level
        performance_reward = (delta_performance / 0.2) * np.exp(
            -abs(next_knowledge - target_performance)
        )
        
        return performance_reward
    
    def _compute_engagement_reward(self, state: SRLState, next_state: SRLState) -> float:
        """Compute engagement level reward"""
        # Extract behavioral engagement
        current_engagement = torch.mean(state.behavioral).item()
        next_engagement = torch.mean(next_state.behavioral).item()
        
        # Sigmoid-based engagement reward
        engagement_threshold = 0.5
        engagement_reward = 1.0 / (1.0 + np.exp(-5 * (next_engagement - engagement_threshold)))
        
        return engagement_reward
    
    def _compute_metacognitive_reward(self, state: SRLState, next_state: SRLState) -> float:
        """Compute metacognitive development reward"""
        # Extract metacognitive components
        current_metacog = torch.mean(state.metacognitive).item()
        next_metacog = torch.mean(next_state.metacognitive).item()
        
        # Metacognitive improvement
        metacog_improvement = next_metacog - current_metacog
        
        return max(0, metacog_improvement)
    
    def _compute_efficiency_reward(
        self, 
        state: SRLState, 
        action: SRLAction, 
        next_state: SRLState
    ) -> float:
        """Compute learning efficiency reward"""
        # Time-based efficiency (assuming timestamp difference)
        time_spent = next_state.timestamp - state.timestamp
        
        # Knowledge gained
        knowledge_gained = torch.mean(next_state.knowledge - state.knowledge).item()
        
        # Efficiency calculation
        if time_spent > 0:
            efficiency = knowledge_gained / time_spent
        else:
            efficiency = 0.0
        
        return min(1.0, max(0.0, efficiency))
    
    def store_experience(
        self, 
        state: SRLState, 
        action: SRLAction, 
        reward: float, 
        next_state: SRLState, 
        done: bool
    ):
        """Store experience in replay buffer"""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.replay_buffer.append(experience)
    
    def update_policy(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Update policy and value networks using PPO algorithm
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            Training statistics
        """
        if len(self.replay_buffer) < batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        
        # Sample batch from replay buffer
        batch = np.random.choice(self.replay_buffer, size=batch_size, replace=False)
        
        # Prepare batch data
        states = [exp.state for exp in batch]
        actions = [exp.action for exp in batch]
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
        next_states = [exp.next_state for exp in batch]
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.bool)
        
        # Encode states
        encoded_states = torch.stack([self.encode_state(state) for state in states])
        encoded_next_states = torch.stack([self.encode_state(state) for state in next_states])
        
        # Compute advantages
        with torch.no_grad():
            values = self.value_network(encoded_states).squeeze()
            next_values = self.value_network(encoded_next_states).squeeze()
            
            # TD targets
            td_targets = rewards + self.gamma * next_values * (~dones)
            advantages = td_targets - values
        
        # Convert actions to tensors
        action_tensors = torch.stack([
            self._srl_action_to_tensor(action) for action in actions
        ])
        
        # Policy update (PPO)
        policy_loss = self._compute_policy_loss(
            encoded_states, action_tensors, advantages
        )
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        
        # Apply differential privacy noise to gradients
        self.privacy_engine.add_noise_to_gradients(
            list(self.state_encoder.parameters()) + 
            list(self.policy_network.parameters())
        )
        
        self.policy_optimizer.step()
        
        # Value update
        values = self.value_network(encoded_states).squeeze()
        value_loss = nn.MSELoss()(values, td_targets.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update statistics
        self.training_stats["policy_loss"] = policy_loss.item()
        self.training_stats["value_loss"] = value_loss.item()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }
    
    def _srl_action_to_tensor(self, action: SRLAction) -> torch.Tensor:
        """Convert SRL action to tensor format"""
        return torch.cat([
            action.content_recommendation,
            action.strategy_suggestion,
            action.feedback_delivery,
            action.social_facilitation
        ])
    
    def _compute_policy_loss(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute PPO policy loss"""
        # Current policy probabilities
        action_probs = self.policy_network(states)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Action log probabilities
        action_indices = torch.argmax(actions, dim=-1)
        log_probs = action_dist.log_prob(action_indices)
        
        # PPO clipped objective
        ratio = torch.exp(log_probs - log_probs.detach())
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        return policy_loss
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated aggregation"""
        parameters = {}
        
        # State encoder parameters
        for name, param in self.state_encoder.named_parameters():
            parameters[f"state_encoder.{name}"] = param.data.clone()
        
        # Policy network parameters
        for name, param in self.policy_network.named_parameters():
            parameters[f"policy_network.{name}"] = param.data.clone()
        
        # Value network parameters
        for name, param in self.value_network.named_parameters():
            parameters[f"value_network.{name}"] = param.data.clone()
        
        # Apply differential privacy
        parameters = self.privacy_engine.add_noise_to_parameters(parameters)
        
        return parameters
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters from federated aggregation"""
        # Update state encoder
        state_encoder_params = {
            name.replace("state_encoder.", ""): param 
            for name, param in parameters.items() 
            if name.startswith("state_encoder.")
        }
        self.state_encoder.load_state_dict(state_encoder_params, strict=False)
        
        # Update policy network
        policy_params = {
            name.replace("policy_network.", ""): param 
            for name, param in parameters.items() 
            if name.startswith("policy_network.")
        }
        self.policy_network.load_state_dict(policy_params, strict=False)
        
        # Update value network
        value_params = {
            name.replace("value_network.", ""): param 
            for name, param in parameters.items() 
            if name.startswith("value_network.")
        }
        self.value_network.load_state_dict(value_params, strict=False)
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            "state_encoder": self.state_encoder.state_dict(),
            "policy_network": self.policy_network.state_dict(),
            "value_network": self.value_network.state_dict(),
            "training_stats": self.training_stats,
            "institution_id": self.institution_id
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.training_stats = checkpoint.get("training_stats", self.training_stats)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get current training statistics"""
        return self.training_stats.copy()
    
    def reset_epsilon(self, new_epsilon: float):
        """Update exploration parameter"""
        self.epsilon = new_epsilon
        self.logger.info(f"Epsilon updated to {new_epsilon}")

