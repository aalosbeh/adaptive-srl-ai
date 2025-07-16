"""
Metacognitive State Estimation Engine

This module implements the metacognitive state estimation engine that monitors
and assesses learners' metacognitive awareness, monitoring, and control processes
in real-time using multi-modal data streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import deque

from ..models.attention_fusion import AttentionFusion
from ..models.temporal_conv_net import TemporalConvNet
from ..utils.logger import Logger


@dataclass
class MetacognitiveComponents:
    """Metacognitive state components"""
    awareness: float      # Metacognitive awareness (0-1)
    monitoring: float     # Metacognitive monitoring (0-1)
    control: float        # Metacognitive control (0-1)
    mas_score: float      # Overall Metacognitive Awareness Score
    confidence: float     # Confidence in estimation
    timestamp: float


@dataclass
class MultiModalInput:
    """Multi-modal input data structure"""
    text_features: torch.Tensor      # NLP features from journals/reflections
    visual_features: torch.Tensor    # Computer vision features from engagement
    temporal_features: torch.Tensor  # Time series features from learning patterns
    graph_features: torch.Tensor     # Graph features from knowledge relationships
    metadata: Dict[str, Any]         # Additional metadata


class HierarchicalAttention(nn.Module):
    """Hierarchical attention mechanism for multi-modal fusion"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical attention"""
        # Multi-head attention
        attended, attention_weights = self.attention(query, key, value)
        
        # Residual connection and layer norm
        attended = self.layer_norm(attended + query)
        
        # Feed-forward network
        output = self.ffn(attended)
        output = self.layer_norm(output + attended)
        
        return output


class MetacognitiveEstimator:
    """
    Metacognitive State Estimation Engine
    
    This engine continuously monitors and estimates learners' metacognitive states
    using multi-modal data streams and hierarchical attention mechanisms.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 512,
        temporal_dim: int = 256,
        graph_dim: int = 128,
        hidden_dim: int = 256,
        num_attention_heads: int = 8,
        temporal_window: int = 50,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize the Metacognitive Estimator
        
        Args:
            text_dim: Dimension of text features
            visual_dim: Dimension of visual features
            temporal_dim: Dimension of temporal features
            graph_dim: Dimension of graph features
            hidden_dim: Hidden dimension for neural networks
            num_attention_heads: Number of attention heads
            temporal_window: Size of temporal window for pattern recognition
            device: Computing device
        """
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.temporal_dim = temporal_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window
        self.device = torch.device(device)
        
        # Initialize logger
        self.logger = Logger("MetacognitiveEstimator")
        
        # Feature projection layers
        self.text_projection = nn.Linear(text_dim, hidden_dim).to(self.device)
        self.visual_projection = nn.Linear(visual_dim, hidden_dim).to(self.device)
        self.temporal_projection = nn.Linear(temporal_dim, hidden_dim).to(self.device)
        self.graph_projection = nn.Linear(graph_dim, hidden_dim).to(self.device)
        
        # Temporal Convolutional Network for pattern recognition
        self.temporal_conv_net = TemporalConvNet(
            num_inputs=hidden_dim,
            num_channels=[hidden_dim, hidden_dim, hidden_dim],
            kernel_size=3,
            dropout=0.1
        ).to(self.device)
        
        # Hierarchical attention mechanisms
        self.cross_modal_attention = HierarchicalAttention(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads
        ).to(self.device)
        
        # Metacognitive component estimators
        self.awareness_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.monitoring_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.control_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Temporal buffer for pattern recognition
        self.temporal_buffer = deque(maxlen=temporal_window)
        
        # Historical metacognitive states
        self.metacognitive_history = deque(maxlen=100)
        
        self.logger.info("MetacognitiveEstimator initialized successfully")
    
    def process_multimodal_input(self, input_data: MultiModalInput) -> torch.Tensor:
        """
        Process multi-modal input data and extract fused features
        
        Args:
            input_data: Multi-modal input data
            
        Returns:
            Fused feature representation
        """
        # Project features to common dimension
        text_features = self.text_projection(input_data.text_features)
        visual_features = self.visual_projection(input_data.visual_features)
        temporal_features = self.temporal_projection(input_data.temporal_features)
        graph_features = self.graph_projection(input_data.graph_features)
        
        # Apply temporal convolution to temporal features
        if len(self.temporal_buffer) > 0:
            # Stack temporal features with history
            temporal_sequence = torch.stack(list(self.temporal_buffer) + [temporal_features])
            temporal_features = self.temporal_conv_net(temporal_sequence.unsqueeze(0))[-1]
        
        # Update temporal buffer
        self.temporal_buffer.append(temporal_features.detach())
        
        # Cross-modal attention fusion
        # Combine text and visual for awareness
        text_visual_combined = torch.stack([text_features, visual_features], dim=0).unsqueeze(0)
        awareness_features = self.cross_modal_attention(
            text_visual_combined, text_visual_combined, text_visual_combined
        ).mean(dim=1)
        
        # Combine temporal and graph for monitoring
        temporal_graph_combined = torch.stack([temporal_features, graph_features], dim=0).unsqueeze(0)
        monitoring_features = self.cross_modal_attention(
            temporal_graph_combined, temporal_graph_combined, temporal_graph_combined
        ).mean(dim=1)
        
        # Combined features for control
        all_features = torch.stack([
            text_features, visual_features, temporal_features, graph_features
        ], dim=0).unsqueeze(0)
        control_features = self.cross_modal_attention(
            all_features, all_features, all_features
        ).mean(dim=1)
        
        return awareness_features, monitoring_features, control_features
    
    def estimate_metacognitive_state(
        self, 
        input_data: MultiModalInput,
        previous_state: Optional[MetacognitiveComponents] = None
    ) -> MetacognitiveComponents:
        """
        Estimate current metacognitive state from multi-modal input
        
        Args:
            input_data: Multi-modal input data
            previous_state: Previous metacognitive state (optional)
            
        Returns:
            Estimated metacognitive components
        """
        # Process multi-modal input
        awareness_features, monitoring_features, control_features = self.process_multimodal_input(input_data)
        
        # Estimate metacognitive components
        awareness = self.awareness_estimator(awareness_features).item()
        monitoring = self.monitoring_estimator(monitoring_features).item()
        control = self.control_estimator(control_features.squeeze(0)).item()
        
        # Compute overall MAS score
        mas_score = (awareness + monitoring + control) / 3.0
        
        # Estimate confidence
        combined_features = torch.cat([
            awareness_features.squeeze(0), 
            monitoring_features.squeeze(0), 
            control_features.squeeze(0)
        ], dim=-1).mean(dim=0, keepdim=True)
        confidence = self.confidence_estimator(combined_features).item()
        
        # Apply temporal smoothing if previous state exists
        if previous_state is not None:
            smoothing_factor = 0.3
            awareness = smoothing_factor * awareness + (1 - smoothing_factor) * previous_state.awareness
            monitoring = smoothing_factor * monitoring + (1 - smoothing_factor) * previous_state.monitoring
            control = smoothing_factor * control + (1 - smoothing_factor) * previous_state.control
            mas_score = (awareness + monitoring + control) / 3.0
        
        # Create metacognitive components
        metacognitive_state = MetacognitiveComponents(
            awareness=awareness,
            monitoring=monitoring,
            control=control,
            mas_score=mas_score,
            confidence=confidence,
            timestamp=input_data.metadata.get("timestamp", 0.0)
        )
        
        # Update history
        self.metacognitive_history.append(metacognitive_state)
        
        return metacognitive_state
    
    def detect_metacognitive_patterns(self) -> Dict[str, Any]:
        """
        Detect patterns in metacognitive development over time
        
        Returns:
            Dictionary containing detected patterns and insights
        """
        if len(self.metacognitive_history) < 10:
            return {"patterns": [], "insights": "Insufficient data for pattern detection"}
        
        # Extract time series data
        awareness_series = [state.awareness for state in self.metacognitive_history]
        monitoring_series = [state.monitoring for state in self.metacognitive_history]
        control_series = [state.control for state in self.metacognitive_history]
        mas_series = [state.mas_score for state in self.metacognitive_history]
        
        patterns = {}
        
        # Trend analysis
        patterns["awareness_trend"] = self._compute_trend(awareness_series)
        patterns["monitoring_trend"] = self._compute_trend(monitoring_series)
        patterns["control_trend"] = self._compute_trend(control_series)
        patterns["mas_trend"] = self._compute_trend(mas_series)
        
        # Variability analysis
        patterns["awareness_variability"] = np.std(awareness_series)
        patterns["monitoring_variability"] = np.std(monitoring_series)
        patterns["control_variability"] = np.std(control_series)
        patterns["mas_variability"] = np.std(mas_series)
        
        # Correlation analysis
        patterns["awareness_monitoring_correlation"] = np.corrcoef(
            awareness_series, monitoring_series
        )[0, 1]
        patterns["monitoring_control_correlation"] = np.corrcoef(
            monitoring_series, control_series
        )[0, 1]
        
        # Generate insights
        insights = self._generate_insights(patterns)
        
        return {
            "patterns": patterns,
            "insights": insights,
            "history_length": len(self.metacognitive_history)
        }
    
    def _compute_trend(self, series: List[float]) -> str:
        """Compute trend direction for a time series"""
        if len(series) < 3:
            return "insufficient_data"
        
        # Linear regression slope
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_insights(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate insights based on detected patterns"""
        insights = []
        
        # Trend insights
        if patterns["mas_trend"] == "increasing":
            insights.append("Metacognitive awareness is improving over time")
        elif patterns["mas_trend"] == "decreasing":
            insights.append("Metacognitive awareness may need attention")
        
        # Variability insights
        if patterns["mas_variability"] > 0.3:
            insights.append("High variability in metacognitive states detected")
        
        # Correlation insights
        if patterns["awareness_monitoring_correlation"] > 0.7:
            insights.append("Strong positive correlation between awareness and monitoring")
        
        if patterns["monitoring_control_correlation"] < 0.3:
            insights.append("Weak correlation between monitoring and control - may indicate intervention need")
        
        return insights
    
    def predict_future_state(
        self, 
        steps_ahead: int = 5
    ) -> List[MetacognitiveComponents]:
        """
        Predict future metacognitive states based on historical patterns
        
        Args:
            steps_ahead: Number of time steps to predict
            
        Returns:
            List of predicted metacognitive states
        """
        if len(self.metacognitive_history) < 10:
            return []
        
        # Extract recent trends
        recent_states = list(self.metacognitive_history)[-10:]
        
        # Simple linear extrapolation for each component
        awareness_values = [state.awareness for state in recent_states]
        monitoring_values = [state.monitoring for state in recent_states]
        control_values = [state.control for state in recent_states]
        
        # Fit linear trends
        x = np.arange(len(recent_states))
        awareness_trend = np.polyfit(x, awareness_values, 1)
        monitoring_trend = np.polyfit(x, monitoring_values, 1)
        control_trend = np.polyfit(x, control_values, 1)
        
        predictions = []
        last_timestamp = recent_states[-1].timestamp
        
        for i in range(1, steps_ahead + 1):
            # Predict values
            pred_awareness = np.clip(
                awareness_trend[0] * (len(recent_states) + i) + awareness_trend[1],
                0.0, 1.0
            )
            pred_monitoring = np.clip(
                monitoring_trend[0] * (len(recent_states) + i) + monitoring_trend[1],
                0.0, 1.0
            )
            pred_control = np.clip(
                control_trend[0] * (len(recent_states) + i) + control_trend[1],
                0.0, 1.0
            )
            
            pred_mas = (pred_awareness + pred_monitoring + pred_control) / 3.0
            
            predicted_state = MetacognitiveComponents(
                awareness=pred_awareness,
                monitoring=pred_monitoring,
                control=pred_control,
                mas_score=pred_mas,
                confidence=0.5,  # Lower confidence for predictions
                timestamp=last_timestamp + i
            )
            
            predictions.append(predicted_state)
        
        return predictions
    
    def get_intervention_recommendations(
        self, 
        current_state: MetacognitiveComponents
    ) -> List[Dict[str, Any]]:
        """
        Generate intervention recommendations based on current metacognitive state
        
        Args:
            current_state: Current metacognitive state
            
        Returns:
            List of intervention recommendations
        """
        recommendations = []
        
        # Low awareness interventions
        if current_state.awareness < 0.4:
            recommendations.append({
                "type": "awareness_intervention",
                "priority": "high",
                "description": "Provide metacognitive awareness training",
                "strategies": [
                    "Self-reflection prompts",
                    "Think-aloud protocols",
                    "Metacognitive strategy instruction"
                ]
            })
        
        # Low monitoring interventions
        if current_state.monitoring < 0.4:
            recommendations.append({
                "type": "monitoring_intervention",
                "priority": "high",
                "description": "Enhance monitoring skills",
                "strategies": [
                    "Progress tracking tools",
                    "Self-assessment rubrics",
                    "Real-time feedback systems"
                ]
            })
        
        # Low control interventions
        if current_state.control < 0.4:
            recommendations.append({
                "type": "control_intervention",
                "priority": "high",
                "description": "Improve metacognitive control",
                "strategies": [
                    "Strategy selection training",
                    "Goal-setting exercises",
                    "Self-regulation practice"
                ]
            })
        
        # Overall low MAS interventions
        if current_state.mas_score < 0.5:
            recommendations.append({
                "type": "comprehensive_intervention",
                "priority": "critical",
                "description": "Comprehensive metacognitive support needed",
                "strategies": [
                    "Intensive metacognitive training program",
                    "One-on-one coaching sessions",
                    "Peer collaboration activities"
                ]
            })
        
        return recommendations
    
    def save_state(self, filepath: str):
        """Save estimator state to file"""
        state = {
            "temporal_buffer": list(self.temporal_buffer),
            "metacognitive_history": list(self.metacognitive_history),
            "model_state_dicts": {
                "text_projection": self.text_projection.state_dict(),
                "visual_projection": self.visual_projection.state_dict(),
                "temporal_projection": self.temporal_projection.state_dict(),
                "graph_projection": self.graph_projection.state_dict(),
                "temporal_conv_net": self.temporal_conv_net.state_dict(),
                "cross_modal_attention": self.cross_modal_attention.state_dict(),
                "awareness_estimator": self.awareness_estimator.state_dict(),
                "monitoring_estimator": self.monitoring_estimator.state_dict(),
                "control_estimator": self.control_estimator.state_dict(),
                "confidence_estimator": self.confidence_estimator.state_dict(),
            }
        }
        
        torch.save(state, filepath)
        self.logger.info(f"MetacognitiveEstimator state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load estimator state from file"""
        state = torch.load(filepath, map_location=self.device)
        
        # Restore buffers and history
        self.temporal_buffer = deque(state["temporal_buffer"], maxlen=self.temporal_window)
        self.metacognitive_history = deque(state["metacognitive_history"], maxlen=100)
        
        # Restore model states
        model_states = state["model_state_dicts"]
        self.text_projection.load_state_dict(model_states["text_projection"])
        self.visual_projection.load_state_dict(model_states["visual_projection"])
        self.temporal_projection.load_state_dict(model_states["temporal_projection"])
        self.graph_projection.load_state_dict(model_states["graph_projection"])
        self.temporal_conv_net.load_state_dict(model_states["temporal_conv_net"])
        self.cross_modal_attention.load_state_dict(model_states["cross_modal_attention"])
        self.awareness_estimator.load_state_dict(model_states["awareness_estimator"])
        self.monitoring_estimator.load_state_dict(model_states["monitoring_estimator"])
        self.control_estimator.load_state_dict(model_states["control_estimator"])
        self.confidence_estimator.load_state_dict(model_states["confidence_estimator"])
        
        self.logger.info(f"MetacognitiveEstimator state loaded from {filepath}")

