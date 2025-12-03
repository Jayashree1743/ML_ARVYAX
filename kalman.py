#!/usr/bin/env python3
"""
Kalman Filter Implementation
Simple 1D Kalman filter for smoothing hand position data.

Author: MiniMax Agent
Created: 2025-12-03
"""

import numpy as np


class KalmanFilter:
    """
    Simple 1D Kalman Filter for noise reduction and prediction.
    
    This implementation is optimized for hand tracking applications
    where we need to smooth noisy position data.
    """
    
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1, initial_value=0):
        """
        Initialize Kalman Filter.
        
        Args:
            process_variance: Process noise variance (how much the signal changes)
            measurement_variance: Measurement noise variance (sensor accuracy)
            initial_value: Initial value of the state
        """
        # Process variance
        self.process_variance = float(process_variance)
        
        # Measurement variance  
        self.measurement_variance = measurement_variance if measurement_variance > 0 else 1e-1
        
        # State variables
        self.posteri_estimate = float(initial_value)
        self.posteri_error_estimate = 1.0  # Initial error estimate
        
        # Priori variables
        self.priori_estimate = None
        self.priori_error_estimate = None
    
    def update(self, measurement):
        """
        Update the Kalman filter with a new measurement.
        
        Args:
            measurement: New measurement value
            
        Returns:
            Filtered/estimated value
        """
        # Priori estimate
        self.priori_estimate = self.posteri_estimate
        self.priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        # Blending factor
        blending_factor = self.priori_error_estimate / (self.priori_error_estimate + self.measurement_variance)
        
        # Posteriori estimate
        self.posteri_estimate = self.priori_estimate + blending_factor * (measurement - self.priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * self.priori_error_estimate
        
        return self.posteri_estimate
    
    def predict(self, steps=1):
        """
        Predict future values.
        
        Args:
            steps: Number of steps to predict ahead
            
        Returns:
            Predicted value
        """
        predicted_value = self.posteri_estimate
        predicted_error = self.posteri_error_estimate
        
        for _ in range(steps):
            predicted_value = predicted_value
            predicted_error = predicted_error + self.process_variance
        
        return predicted_value
    
    def reset(self, initial_value=0):
        """Reset the filter to initial state."""
        self.posteri_estimate = float(initial_value)
        self.posteri_error_estimate = 1.0
        self.priori_estimate = None
        self.priori_error_estimate = None