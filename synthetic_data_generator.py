"""
Synthetic Data Generator for Self-Regulated Learning

This module generates realistic synthetic datasets for training and evaluating
the adaptive multi-modal AI framework for self-regulated learning.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import random
from datetime import datetime, timedelta
import os

from ..core.federated_drl_agent import SRLState, SRLAction
from ..core.metacognitive_estimator import MetacognitiveComponents, MultiModalInput
from ..utils.logger import Logger


@dataclass
class LearnerProfile:
    """Learner profile for synthetic data generation"""
    learner_id: str
    age: int
    education_level: str  # "high_school", "undergraduate", "graduate", "professional"
    domain: str          # "mathematics", "science", "language", "programming", etc.
    baseline_ability: float     # 0-1 scale
    learning_style: str         # "visual", "auditory", "kinesthetic", "mixed"
    motivation_level: float     # 0-1 scale
    metacognitive_maturity: float  # 0-1 scale
    personality_traits: Dict[str, float]  # Big Five traits
    learning_goals: List[str]
    preferred_strategies: List[str]


@dataclass
class LearningSession:
    """Individual learning session data"""
    session_id: str
    learner_id: str
    timestamp: datetime
    duration_minutes: int
    content_topic: str
    difficulty_level: float  # 0-1 scale
    initial_state: SRLState
    final_state: SRLState
    actions_taken: List[SRLAction]
    performance_metrics: Dict[str, float]
    engagement_level: float
    metacognitive_events: List[MetacognitiveComponents]
    multimodal_data: List[MultiModalInput]


class SyntheticDataGenerator:
    """
    Synthetic Data Generator for Self-Regulated Learning
    
    Generates realistic synthetic datasets that capture the complexity of
    self-regulated learning processes across different educational contexts.
    """
    
    def __init__(
        self,
        num_learners: int = 1000,
        num_institutions: int = 10,
        simulation_days: int = 180,
        random_seed: int = 42,
        output_dir: str = "data/synthetic",
        **kwargs
    ):
        """
        Initialize the Synthetic Data Generator
        
        Args:
            num_learners: Number of synthetic learners to generate
            num_institutions: Number of educational institutions
            simulation_days: Number of days to simulate
            random_seed: Random seed for reproducibility
            output_dir: Output directory for generated data
        """
        self.num_learners = num_learners
        self.num_institutions = num_institutions
        self.simulation_days = simulation_days
        self.output_dir = output_dir
        
        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        
        # Initialize logger
        self.logger = Logger("SyntheticDataGenerator")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Learning domains and topics
        self.domains = {
            "mathematics": [
                "algebra", "calculus", "statistics", "geometry", "trigonometry",
                "linear_algebra", "differential_equations", "probability"
            ],
            "science": [
                "physics", "chemistry", "biology", "earth_science", "astronomy",
                "environmental_science", "neuroscience", "genetics"
            ],
            "programming": [
                "python", "javascript", "data_structures", "algorithms", "web_development",
                "machine_learning", "databases", "software_engineering"
            ],
            "language": [
                "grammar", "vocabulary", "reading_comprehension", "writing",
                "literature", "linguistics", "translation", "communication"
            ]
        }
        
        # Learning strategies
        self.learning_strategies = [
            "note_taking", "summarization", "elaboration", "organization",
            "critical_thinking", "self_questioning", "mnemonics", "visualization",
            "practice_testing", "spaced_repetition", "interleaving", "reflection"
        ]
        
        # Intervention types
        self.intervention_types = [
            "content_recommendation", "strategy_suggestion", "feedback_delivery",
            "social_facilitation", "metacognitive_prompt", "motivation_boost",
            "difficulty_adjustment", "pace_modification"
        ]
        
        self.logger.info(f"SyntheticDataGenerator initialized for {num_learners} learners")
    
    def generate_learner_profiles(self) -> List[LearnerProfile]:
        """Generate diverse learner profiles"""
        profiles = []
        
        for i in range(self.num_learners):
            # Basic demographics
            age = np.random.choice([16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 
                                 p=[0.1, 0.15, 0.2, 0.15, 0.1, 0.1, 0.08, 0.06, 0.04, 0.02])
            
            education_levels = ["high_school", "undergraduate", "graduate", "professional"]
            education_weights = [0.3, 0.5, 0.15, 0.05]
            education_level = np.random.choice(education_levels, p=education_weights)
            
            # Learning characteristics
            domain = np.random.choice(list(self.domains.keys()))
            baseline_ability = np.random.beta(2, 2)  # Bell-shaped distribution
            learning_style = np.random.choice(["visual", "auditory", "kinesthetic", "mixed"],
                                            p=[0.4, 0.3, 0.2, 0.1])
            motivation_level = np.random.beta(2, 1.5)  # Slightly skewed toward higher motivation
            metacognitive_maturity = np.random.beta(1.5, 2)  # Skewed toward lower maturity
            
            # Big Five personality traits
            personality_traits = {
                "openness": np.random.beta(2, 2),
                "conscientiousness": np.random.beta(2, 2),
                "extraversion": np.random.beta(2, 2),
                "agreeableness": np.random.beta(2, 2),
                "neuroticism": np.random.beta(2, 2)
            }
            
            # Learning goals (2-4 goals per learner)
            num_goals = np.random.randint(2, 5)
            learning_goals = np.random.choice(
                self.domains[domain], size=num_goals, replace=False
            ).tolist()
            
            # Preferred strategies (3-6 strategies per learner)
            num_strategies = np.random.randint(3, 7)
            preferred_strategies = np.random.choice(
                self.learning_strategies, size=num_strategies, replace=False
            ).tolist()
            
            profile = LearnerProfile(
                learner_id=f"learner_{i:04d}",
                age=age,
                education_level=education_level,
                domain=domain,
                baseline_ability=baseline_ability,
                learning_style=learning_style,
                motivation_level=motivation_level,
                metacognitive_maturity=metacognitive_maturity,
                personality_traits=personality_traits,
                learning_goals=learning_goals,
                preferred_strategies=preferred_strategies
            )
            
            profiles.append(profile)
        
        self.logger.info(f"Generated {len(profiles)} learner profiles")
        return profiles
    
    def generate_srl_state(
        self, 
        profile: LearnerProfile, 
        session_context: Dict[str, Any],
        previous_state: Optional[SRLState] = None
    ) -> SRLState:
        """Generate SRL state based on learner profile and context"""
        
        # Base state influenced by learner characteristics
        base_metacognitive = profile.metacognitive_maturity
        base_cognitive = profile.baseline_ability
        base_behavioral = profile.motivation_level
        base_emotional = 1.0 - profile.personality_traits["neuroticism"]
        base_knowledge = profile.baseline_ability
        
        # Context influences
        difficulty_factor = session_context.get("difficulty_level", 0.5)
        time_factor = session_context.get("time_of_day", 0.5)  # 0=morning, 1=night
        fatigue_factor = session_context.get("fatigue_level", 0.3)
        
        # Apply context modifications
        metacognitive_noise = np.random.normal(0, 0.1)
        cognitive_noise = np.random.normal(0, 0.1)
        behavioral_noise = np.random.normal(0, 0.15)
        emotional_noise = np.random.normal(0, 0.2)
        knowledge_noise = np.random.normal(0, 0.05)
        
        # Difficulty effects
        cognitive_adjustment = -0.3 * (difficulty_factor - 0.5)
        emotional_adjustment = -0.2 * (difficulty_factor - 0.5)
        
        # Time of day effects
        behavioral_adjustment = -0.2 * time_factor  # Lower energy at night
        cognitive_adjustment += -0.1 * time_factor
        
        # Fatigue effects
        metacognitive_adjustment = -0.3 * fatigue_factor
        cognitive_adjustment += -0.4 * fatigue_factor
        behavioral_adjustment += -0.5 * fatigue_factor
        
        # Apply temporal continuity if previous state exists
        if previous_state is not None:
            continuity_factor = 0.7  # How much previous state influences current
            prev_metacog = torch.mean(previous_state.metacognitive).item()
            prev_cognitive = torch.mean(previous_state.cognitive).item()
            prev_behavioral = torch.mean(previous_state.behavioral).item()
            prev_emotional = torch.mean(previous_state.emotional).item()
            prev_knowledge = torch.mean(previous_state.knowledge).item()
            
            base_metacognitive = continuity_factor * prev_metacog + (1 - continuity_factor) * base_metacognitive
            base_cognitive = continuity_factor * prev_cognitive + (1 - continuity_factor) * base_cognitive
            base_behavioral = continuity_factor * prev_behavioral + (1 - continuity_factor) * base_behavioral
            base_emotional = continuity_factor * prev_emotional + (1 - continuity_factor) * base_emotional
            base_knowledge = continuity_factor * prev_knowledge + (1 - continuity_factor) * base_knowledge
        
        # Compute final state values
        metacognitive_val = np.clip(
            base_metacognitive + metacognitive_adjustment + metacognitive_noise, 0, 1
        )
        cognitive_val = np.clip(
            base_cognitive + cognitive_adjustment + cognitive_noise, 0, 1
        )
        behavioral_val = np.clip(
            base_behavioral + behavioral_adjustment + behavioral_noise, 0, 1
        )
        emotional_val = np.clip(
            base_emotional + emotional_adjustment + emotional_noise, 0, 1
        )
        knowledge_val = np.clip(
            base_knowledge + knowledge_noise, 0, 1
        )
        
        # Create state tensors (each component has multiple dimensions)
        metacognitive = torch.tensor([
            metacognitive_val,  # awareness
            metacognitive_val * 0.9 + np.random.normal(0, 0.05),  # monitoring
            metacognitive_val * 0.8 + np.random.normal(0, 0.05),  # control
        ], dtype=torch.float32)
        
        cognitive = torch.tensor([
            cognitive_val,  # attention
            cognitive_val * 0.9 + np.random.normal(0, 0.05),  # memory
            cognitive_val * 0.85 + np.random.normal(0, 0.05),  # processing speed
            cognitive_val * 0.8 + np.random.normal(0, 0.05),  # working memory
        ], dtype=torch.float32)
        
        behavioral = torch.tensor([
            behavioral_val,  # engagement
            behavioral_val * 0.9 + np.random.normal(0, 0.05),  # persistence
            behavioral_val * 0.8 + np.random.normal(0, 0.05),  # strategy use
            behavioral_val * 0.85 + np.random.normal(0, 0.05),  # help seeking
        ], dtype=torch.float32)
        
        emotional = torch.tensor([
            emotional_val,  # motivation
            1.0 - emotional_val * 0.5 + np.random.normal(0, 0.1),  # anxiety (inverse)
            emotional_val * 0.9 + np.random.normal(0, 0.05),  # confidence
            emotional_val * 0.8 + np.random.normal(0, 0.05),  # interest
        ], dtype=torch.float32)
        
        knowledge = torch.tensor([
            knowledge_val,  # domain knowledge
            knowledge_val * 0.9 + np.random.normal(0, 0.03),  # skill level
            knowledge_val * 0.85 + np.random.normal(0, 0.03),  # conceptual understanding
        ], dtype=torch.float32)
        
        # Clip all values to [0, 1]
        metacognitive = torch.clamp(metacognitive, 0, 1)
        cognitive = torch.clamp(cognitive, 0, 1)
        behavioral = torch.clamp(behavioral, 0, 1)
        emotional = torch.clamp(emotional, 0, 1)
        knowledge = torch.clamp(knowledge, 0, 1)
        
        return SRLState(
            metacognitive=metacognitive,
            cognitive=cognitive,
            behavioral=behavioral,
            emotional=emotional,
            knowledge=knowledge,
            timestamp=session_context.get("timestamp", 0.0),
            learner_id=profile.learner_id
        )
    
    def generate_multimodal_input(
        self, 
        state: SRLState, 
        session_context: Dict[str, Any]
    ) -> MultiModalInput:
        """Generate multi-modal input data corresponding to SRL state"""
        
        # Text features (simulating NLP features from journals/reflections)
        # Higher metacognitive states lead to more complex text features
        metacog_level = torch.mean(state.metacognitive).item()
        text_complexity = 0.3 + 0.7 * metacog_level
        text_features = torch.randn(768) * text_complexity + metacog_level
        
        # Visual features (simulating computer vision features from engagement)
        # Higher behavioral engagement leads to different visual patterns
        engagement_level = torch.mean(state.behavioral).item()
        visual_features = torch.randn(512) * (0.5 + 0.5 * engagement_level)
        
        # Temporal features (simulating time series patterns)
        # Based on learning patterns and cognitive state
        cognitive_level = torch.mean(state.cognitive).item()
        temporal_base = np.sin(np.linspace(0, 4*np.pi, 256)) * cognitive_level
        temporal_noise = np.random.normal(0, 0.1, 256)
        temporal_features = torch.tensor(temporal_base + temporal_noise, dtype=torch.float32)
        
        # Graph features (simulating knowledge graph embeddings)
        # Based on knowledge state and domain
        knowledge_level = torch.mean(state.knowledge).item()
        graph_features = torch.randn(128) * knowledge_level + 0.5
        
        # Metadata
        metadata = {
            "timestamp": state.timestamp,
            "learner_id": state.learner_id,
            "session_id": session_context.get("session_id", "unknown"),
            "content_topic": session_context.get("content_topic", "general"),
            "difficulty_level": session_context.get("difficulty_level", 0.5)
        }
        
        return MultiModalInput(
            text_features=text_features,
            visual_features=visual_features,
            temporal_features=temporal_features,
            graph_features=graph_features,
            metadata=metadata
        )
    
    def generate_learning_session(
        self, 
        profile: LearnerProfile, 
        session_date: datetime,
        previous_session: Optional[LearningSession] = None
    ) -> LearningSession:
        """Generate a complete learning session"""
        
        session_id = f"{profile.learner_id}_{session_date.strftime('%Y%m%d_%H%M%S')}"
        
        # Session characteristics
        duration_minutes = np.random.randint(15, 180)  # 15 minutes to 3 hours
        content_topic = np.random.choice(self.domains[profile.domain])
        
        # Difficulty progression over time
        base_difficulty = 0.3 + 0.4 * profile.baseline_ability
        if previous_session:
            # Adaptive difficulty based on previous performance
            prev_performance = np.mean(list(previous_session.performance_metrics.values()))
            difficulty_adjustment = 0.1 * (prev_performance - 0.7)  # Target 70% performance
            base_difficulty = np.clip(base_difficulty + difficulty_adjustment, 0.1, 0.9)
        
        difficulty_level = base_difficulty + np.random.normal(0, 0.1)
        difficulty_level = np.clip(difficulty_level, 0.1, 0.9)
        
        # Session context
        time_of_day = session_date.hour / 24.0
        fatigue_level = np.random.beta(2, 5)  # Most sessions with low fatigue
        
        session_context = {
            "session_id": session_id,
            "timestamp": session_date.timestamp(),
            "content_topic": content_topic,
            "difficulty_level": difficulty_level,
            "time_of_day": time_of_day,
            "fatigue_level": fatigue_level,
            "duration_minutes": duration_minutes
        }
        
        # Generate initial state
        previous_state = previous_session.final_state if previous_session else None
        initial_state = self.generate_srl_state(profile, session_context, previous_state)
        
        # Simulate learning progression during session
        num_timesteps = max(3, duration_minutes // 15)  # One timestep per 15 minutes
        states = [initial_state]
        actions = []
        multimodal_data = []
        metacognitive_events = []
        
        for t in range(num_timesteps - 1):
            current_state = states[-1]
            
            # Generate multi-modal input
            multimodal_input = self.generate_multimodal_input(current_state, session_context)
            multimodal_data.append(multimodal_input)
            
            # Generate metacognitive event
            metacog_event = MetacognitiveComponents(
                awareness=torch.mean(current_state.metacognitive).item(),
                monitoring=current_state.metacognitive[1].item(),
                control=current_state.metacognitive[2].item(),
                mas_score=torch.mean(current_state.metacognitive).item(),
                confidence=np.random.beta(2, 2),
                timestamp=session_context["timestamp"] + t * 900  # 15-minute intervals
            )
            metacognitive_events.append(metacog_event)
            
            # Generate action (intervention)
            action = self.generate_srl_action(current_state, profile, session_context)
            actions.append(action)
            
            # Generate next state based on action and learning progression
            next_context = session_context.copy()
            next_context["timestamp"] += 900  # 15 minutes later
            next_state = self.generate_srl_state(profile, next_context, current_state)
            
            # Apply learning progression (knowledge should generally increase)
            learning_rate = 0.01 + 0.02 * torch.mean(current_state.metacognitive).item()
            knowledge_gain = learning_rate * (1 - torch.mean(current_state.knowledge).item())
            next_state.knowledge += knowledge_gain
            next_state.knowledge = torch.clamp(next_state.knowledge, 0, 1)
            
            states.append(next_state)
        
        final_state = states[-1]
        
        # Generate performance metrics
        performance_metrics = self.generate_performance_metrics(
            initial_state, final_state, difficulty_level, duration_minutes
        )
        
        # Calculate overall engagement
        engagement_level = np.mean([torch.mean(state.behavioral).item() for state in states])
        
        return LearningSession(
            session_id=session_id,
            learner_id=profile.learner_id,
            timestamp=session_date,
            duration_minutes=duration_minutes,
            content_topic=content_topic,
            difficulty_level=difficulty_level,
            initial_state=initial_state,
            final_state=final_state,
            actions_taken=actions,
            performance_metrics=performance_metrics,
            engagement_level=engagement_level,
            metacognitive_events=metacognitive_events,
            multimodal_data=multimodal_data
        )
    
    def generate_srl_action(
        self, 
        state: SRLState, 
        profile: LearnerProfile, 
        context: Dict[str, Any]
    ) -> SRLAction:
        """Generate SRL action based on state and profile"""
        
        # Action dimensions
        action_dim = 16  # 4 components × 4 dimensions each
        quarter_dim = action_dim // 4
        
        # Base action influenced by current state and needs
        metacog_level = torch.mean(state.metacognitive).item()
        cognitive_level = torch.mean(state.cognitive).item()
        behavioral_level = torch.mean(state.behavioral).item()
        emotional_level = torch.mean(state.emotional).item()
        
        # Content recommendation (based on knowledge gaps)
        knowledge_level = torch.mean(state.knowledge).item()
        content_rec = torch.zeros(quarter_dim)
        if knowledge_level < 0.5:
            content_rec[0] = 0.8  # Basic content
        elif knowledge_level < 0.8:
            content_rec[1] = 0.7  # Intermediate content
        else:
            content_rec[2] = 0.6  # Advanced content
        
        # Strategy suggestion (based on metacognitive state)
        strategy_sug = torch.zeros(quarter_dim)
        if metacog_level < 0.4:
            strategy_sug[0] = 0.9  # Basic strategies
        elif metacog_level < 0.7:
            strategy_sug[1] = 0.8  # Intermediate strategies
        else:
            strategy_sug[2] = 0.7  # Advanced strategies
        
        # Feedback delivery (based on emotional state)
        feedback_del = torch.zeros(quarter_dim)
        if emotional_level < 0.4:
            feedback_del[0] = 0.8  # Encouraging feedback
        elif emotional_level < 0.7:
            feedback_del[1] = 0.6  # Neutral feedback
        else:
            feedback_del[2] = 0.5  # Challenging feedback
        
        # Social facilitation (based on behavioral state)
        social_fac = torch.zeros(quarter_dim)
        if behavioral_level < 0.4:
            social_fac[0] = 0.7  # Peer support
        elif behavioral_level < 0.7:
            social_fac[1] = 0.5  # Group activities
        else:
            social_fac[2] = 0.3  # Independent work
        
        # Determine intervention type
        intervention_types = ["content", "strategy", "feedback", "social"]
        # Choose based on greatest need
        needs = [1-knowledge_level, 1-metacog_level, 1-emotional_level, 1-behavioral_level]
        intervention_type = intervention_types[np.argmax(needs)]
        
        # Calculate confidence based on state certainty
        state_variance = np.var([metacog_level, cognitive_level, behavioral_level, emotional_level])
        confidence = 1.0 - min(state_variance * 4, 0.5)  # Higher variance = lower confidence
        
        return SRLAction(
            content_recommendation=content_rec,
            strategy_suggestion=strategy_sug,
            feedback_delivery=feedback_del,
            social_facilitation=social_fac,
            intervention_type=intervention_type,
            confidence=confidence
        )
    
    def generate_performance_metrics(
        self, 
        initial_state: SRLState, 
        final_state: SRLState, 
        difficulty: float, 
        duration: int
    ) -> Dict[str, float]:
        """Generate performance metrics for a learning session"""
        
        # Knowledge gain
        initial_knowledge = torch.mean(initial_state.knowledge).item()
        final_knowledge = torch.mean(final_state.knowledge).item()
        knowledge_gain = final_knowledge - initial_knowledge
        
        # Engagement metrics
        initial_engagement = torch.mean(initial_state.behavioral).item()
        final_engagement = torch.mean(final_state.behavioral).item()
        avg_engagement = (initial_engagement + final_engagement) / 2
        
        # Metacognitive development
        initial_metacog = torch.mean(initial_state.metacognitive).item()
        final_metacog = torch.mean(final_state.metacognitive).item()
        metacog_development = final_metacog - initial_metacog
        
        # Performance score (adjusted for difficulty)
        base_performance = final_knowledge * avg_engagement
        difficulty_adjustment = 1.0 + 0.5 * (difficulty - 0.5)  # Harder tasks worth more
        performance_score = base_performance * difficulty_adjustment
        
        # Efficiency (knowledge per minute)
        efficiency = knowledge_gain / (duration / 60.0) if duration > 0 else 0
        
        # Persistence (based on engagement maintenance)
        persistence = 1.0 - abs(final_engagement - initial_engagement)
        
        return {
            "knowledge_gain": knowledge_gain,
            "performance_score": performance_score,
            "engagement_score": avg_engagement,
            "metacognitive_development": metacog_development,
            "efficiency": efficiency,
            "persistence": persistence,
            "completion_rate": 1.0  # Assume all sessions completed for simplicity
        }
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """Generate complete synthetic dataset"""
        
        self.logger.info("Starting complete dataset generation...")
        
        # Generate learner profiles
        profiles = self.generate_learner_profiles()
        
        # Assign learners to institutions
        institution_assignments = {}
        learners_per_institution = self.num_learners // self.num_institutions
        
        for i, profile in enumerate(profiles):
            institution_id = f"institution_{i // learners_per_institution:02d}"
            institution_assignments[profile.learner_id] = institution_id
        
        # Generate learning sessions over time
        all_sessions = []
        learner_session_history = {profile.learner_id: [] for profile in profiles}
        
        start_date = datetime.now() - timedelta(days=self.simulation_days)
        
        for day in range(self.simulation_days):
            current_date = start_date + timedelta(days=day)
            
            # Each learner has 0-3 sessions per day (weighted toward 1-2)
            for profile in profiles:
                num_sessions = np.random.choice([0, 1, 2, 3], p=[0.2, 0.5, 0.25, 0.05])
                
                for session_num in range(num_sessions):
                    # Random time during the day
                    hour = np.random.randint(8, 22)  # 8 AM to 10 PM
                    minute = np.random.randint(0, 60)
                    session_time = current_date.replace(hour=hour, minute=minute)
                    
                    # Get previous session for continuity
                    previous_session = (
                        learner_session_history[profile.learner_id][-1] 
                        if learner_session_history[profile.learner_id] 
                        else None
                    )
                    
                    # Generate session
                    session = self.generate_learning_session(
                        profile, session_time, previous_session
                    )
                    
                    all_sessions.append(session)
                    learner_session_history[profile.learner_id].append(session)
            
            if (day + 1) % 30 == 0:
                self.logger.info(f"Generated data for {day + 1} days...")
        
        # Organize data by institution for federated learning
        institutional_data = {}
        for session in all_sessions:
            institution_id = institution_assignments[session.learner_id]
            if institution_id not in institutional_data:
                institutional_data[institution_id] = []
            institutional_data[institution_id].append(session)
        
        dataset = {
            "profiles": profiles,
            "sessions": all_sessions,
            "institution_assignments": institution_assignments,
            "institutional_data": institutional_data,
            "metadata": {
                "num_learners": self.num_learners,
                "num_institutions": self.num_institutions,
                "simulation_days": self.simulation_days,
                "total_sessions": len(all_sessions),
                "generation_date": datetime.now().isoformat()
            }
        }
        
        self.logger.info(f"Generated complete dataset with {len(all_sessions)} sessions")
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], format: str = "pickle"):
        """Save dataset to files"""
        
        if format == "pickle":
            import pickle
            filepath = os.path.join(self.output_dir, "synthetic_srl_dataset.pkl")
            with open(filepath, "wb") as f:
                pickle.dump(dataset, f)
            self.logger.info(f"Dataset saved to {filepath}")
        
        elif format == "json":
            # Convert to JSON-serializable format
            json_dataset = self._convert_to_json_serializable(dataset)
            filepath = os.path.join(self.output_dir, "synthetic_srl_dataset.json")
            with open(filepath, "w") as f:
                json.dump(json_dataset, f, indent=2)
            self.logger.info(f"Dataset saved to {filepath}")
        
        # Save institutional data separately for federated learning
        for institution_id, sessions in dataset["institutional_data"].items():
            institution_dir = os.path.join(self.output_dir, "institutions", institution_id)
            os.makedirs(institution_dir, exist_ok=True)
            
            if format == "pickle":
                import pickle
                filepath = os.path.join(institution_dir, "sessions.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(sessions, f)
            
            # Also save summary statistics
            stats = self._compute_institutional_statistics(sessions)
            stats_filepath = os.path.join(institution_dir, "statistics.json")
            with open(stats_filepath, "w") as f:
                json.dump(stats, f, indent=2)
        
        # Save learner profiles
        profiles_data = [asdict(profile) for profile in dataset["profiles"]]
        profiles_filepath = os.path.join(self.output_dir, "learner_profiles.json")
        with open(profiles_filepath, "w") as f:
            json.dump(profiles_data, f, indent=2)
        
        self.logger.info("Dataset saved successfully")
    
    def _convert_to_json_serializable(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dataset to JSON-serializable format"""
        # This is a simplified version - in practice, you'd need to handle
        # torch tensors and other non-serializable objects
        json_dataset = {
            "metadata": dataset["metadata"],
            "institution_assignments": dataset["institution_assignments"],
            "num_profiles": len(dataset["profiles"]),
            "num_sessions": len(dataset["sessions"])
        }
        return json_dataset
    
    def _compute_institutional_statistics(self, sessions: List[LearningSession]) -> Dict[str, Any]:
        """Compute statistics for an institution's data"""
        if not sessions:
            return {}
        
        # Basic statistics
        total_sessions = len(sessions)
        unique_learners = len(set(session.learner_id for session in sessions))
        
        # Performance statistics
        performance_scores = [
            session.performance_metrics["performance_score"] 
            for session in sessions
        ]
        engagement_scores = [session.engagement_level for session in sessions]
        
        # Learning progression
        knowledge_gains = [
            session.performance_metrics["knowledge_gain"] 
            for session in sessions
        ]
        
        stats = {
            "total_sessions": total_sessions,
            "unique_learners": unique_learners,
            "avg_performance": np.mean(performance_scores),
            "std_performance": np.std(performance_scores),
            "avg_engagement": np.mean(engagement_scores),
            "std_engagement": np.std(engagement_scores),
            "avg_knowledge_gain": np.mean(knowledge_gains),
            "total_learning_hours": sum(session.duration_minutes for session in sessions) / 60.0
        }
        
        return stats


def main():
    """Main function for generating synthetic dataset"""
    generator = SyntheticDataGenerator(
        num_learners=1000,
        num_institutions=10,
        simulation_days=180,
        output_dir="data/synthetic"
    )
    
    dataset = generator.generate_complete_dataset()
    generator.save_dataset(dataset, format="pickle")
    generator.save_dataset(dataset, format="json")
    
    print("Synthetic dataset generation completed!")


if __name__ == "__main__":
    main()

