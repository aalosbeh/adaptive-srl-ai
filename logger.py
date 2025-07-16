"""
Logging Utility for Adaptive SRL AI Framework

This module provides comprehensive logging functionality for the framework.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json


class Logger:
    """
    Enhanced Logger for the Adaptive SRL AI Framework
    
    Provides structured logging with different levels and output formats.
    """
    
    def __init__(
        self,
        name: str,
        log_level: str = "INFO",
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        Initialize the Logger
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            console_output: Whether to output to console
            file_output: Whether to output to file
        """
        self.name = name
        self.log_dir = log_dir
        
        # Create log directory
        if file_output:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            log_filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        if extra:
            message = f"{message} | Extra: {json.dumps(extra)}"
        self.logger.debug(message)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        if extra:
            message = f"{message} | Extra: {json.dumps(extra)}"
        self.logger.info(message)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        if extra:
            message = f"{message} | Extra: {json.dumps(extra)}"
        self.logger.warning(message)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message"""
        if extra:
            message = f"{message} | Extra: {json.dumps(extra)}"
        self.logger.error(message)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        if extra:
            message = f"{message} | Extra: {json.dumps(extra)}"
        self.logger.critical(message)
    
    def log_experiment(self, experiment_name: str, parameters: Dict[str, Any], results: Dict[str, Any]):
        """Log experiment details"""
        experiment_data = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters,
            "results": results
        }
        
        self.info(f"Experiment: {experiment_name}", extra=experiment_data)
        
        # Save to separate experiment log
        experiment_log_path = os.path.join(self.log_dir, "experiments.jsonl")
        with open(experiment_log_path, "a") as f:
            f.write(json.dumps(experiment_data) + "\n")
    
    def log_training_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics"""
        metrics_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        
        self.info(f"Training Epoch {epoch}", extra=metrics_data)
        
        # Save to separate metrics log
        metrics_log_path = os.path.join(self.log_dir, "training_metrics.jsonl")
        with open(metrics_log_path, "a") as f:
            f.write(json.dumps(metrics_data) + "\n")
    
    def log_federated_round(self, round_num: int, participants: list, aggregation_results: Dict[str, Any]):
        """Log federated learning round"""
        federated_data = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "participants": participants,
            "aggregation_results": aggregation_results
        }
        
        self.info(f"Federated Round {round_num}", extra=federated_data)
        
        # Save to separate federated log
        federated_log_path = os.path.join(self.log_dir, "federated_learning.jsonl")
        with open(federated_log_path, "a") as f:
            f.write(json.dumps(federated_data) + "\n")

