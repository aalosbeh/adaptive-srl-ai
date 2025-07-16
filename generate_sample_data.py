#!/usr/bin/env python3
"""
Simplified Sample Data Generator

This script generates sample data for demonstration purposes without requiring
heavy ML dependencies.
"""

import json
import random
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data():
    """Generate sample synthetic data for demonstration"""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create data directory
    os.makedirs("data/sample_datasets", exist_ok=True)
    
    # Generate sample learner profiles
    learner_profiles = []
    for i in range(100):
        profile = {
            "learner_id": f"learner_{i:03d}",
            "age": random.randint(16, 25),
            "education_level": random.choice(["high_school", "undergraduate", "graduate"]),
            "domain": random.choice(["mathematics", "science", "programming", "language"]),
            "baseline_ability": round(random.uniform(0.2, 0.9), 3),
            "learning_style": random.choice(["visual", "auditory", "kinesthetic", "mixed"]),
            "motivation_level": round(random.uniform(0.3, 1.0), 3),
            "metacognitive_maturity": round(random.uniform(0.1, 0.8), 3)
        }
        learner_profiles.append(profile)
    
    # Generate sample learning sessions
    learning_sessions = []
    for i in range(500):
        learner = random.choice(learner_profiles)
        session = {
            "session_id": f"session_{i:04d}",
            "learner_id": learner["learner_id"],
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 180))).isoformat(),
            "duration_minutes": random.randint(15, 180),
            "content_topic": random.choice(["algebra", "calculus", "physics", "chemistry", "python", "javascript"]),
            "difficulty_level": round(random.uniform(0.1, 0.9), 3),
            "performance_metrics": {
                "knowledge_gain": round(random.uniform(0.0, 0.3), 3),
                "engagement_score": round(random.uniform(0.2, 1.0), 3),
                "completion_rate": round(random.uniform(0.7, 1.0), 3)
            },
            "metacognitive_state": {
                "awareness": round(random.uniform(0.1, 0.9), 3),
                "monitoring": round(random.uniform(0.1, 0.9), 3),
                "control": round(random.uniform(0.1, 0.9), 3)
            }
        }
        learning_sessions.append(session)
    
    # Generate institutional assignments
    institution_assignments = {}
    for i, profile in enumerate(learner_profiles):
        institution_id = f"institution_{i // 10:02d}"
        institution_assignments[profile["learner_id"]] = institution_id
    
    # Create dataset
    dataset = {
        "metadata": {
            "num_learners": len(learner_profiles),
            "num_sessions": len(learning_sessions),
            "num_institutions": 10,
            "generation_date": datetime.now().isoformat(),
            "description": "Sample synthetic dataset for Adaptive SRL AI Framework"
        },
        "learner_profiles": learner_profiles,
        "learning_sessions": learning_sessions,
        "institution_assignments": institution_assignments
    }
    
    # Save dataset
    with open("data/sample_datasets/sample_srl_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    # Save institutional data separately
    institutional_data = {}
    for session in learning_sessions:
        institution_id = institution_assignments[session["learner_id"]]
        if institution_id not in institutional_data:
            institutional_data[institution_id] = []
        institutional_data[institution_id].append(session)
    
    for institution_id, sessions in institutional_data.items():
        institution_dir = f"data/sample_datasets/institutions/{institution_id}"
        os.makedirs(institution_dir, exist_ok=True)
        
        with open(f"{institution_dir}/sessions.json", "w") as f:
            json.dump(sessions, f, indent=2)
        
        # Generate statistics
        stats = {
            "total_sessions": len(sessions),
            "unique_learners": len(set(s["learner_id"] for s in sessions)),
            "avg_performance": round(np.mean([s["performance_metrics"]["engagement_score"] for s in sessions]), 3),
            "avg_duration": round(np.mean([s["duration_minutes"] for s in sessions]), 1)
        }
        
        with open(f"{institution_dir}/statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
    
    print(f"Generated sample dataset with {len(learner_profiles)} learners and {len(learning_sessions)} sessions")
    print("Data saved to data/sample_datasets/")

if __name__ == "__main__":
    generate_sample_data()

