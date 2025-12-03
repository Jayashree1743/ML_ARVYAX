#!/usr/bin/env python3
"""
Demo Mode for Hand-of-Sauron AR
Simulated hand tracking for testing without camera access.

Author: MiniMax Agent
Created: 2025-12-03
"""

import cv2
import numpy as np
import time
import math
from kalman import KalmanFilter
from cube_renderer import CubeRenderer


class DemoHandDangerDetector:
    """
    Demo version that simulates hand tracking for testing purposes.
    Creates synthetic hand movements to demonstrate the system capabilities.
    """
    
    def __init__(self, width=640, height=480, target_fps=15):
        """Initialize demo mode."""
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.frame_count = 0
        
        # Create a synthetic background
        self.background = self.create_demo_background()
        
        # Initialize Kalman filters
        self.kalman_x = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)
        self.kalman_y = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)
        
        # Cube renderer
        self.cube_renderer = CubeRenderer()
        
        # State management
        self.current_state = "SAFE"
        self.state_history = []
        self.easter_egg_triggered = False
        self.still_frame_count = 0
        self.last_hand_position = None
        self.demo_mode = "circle"  # circle, line, still, zigzag
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Demo parameters
        self.center_x = width // 2
        self.center_y = height // 2
        self.radius = 100
        self.angle = 0
    
    def create_demo_background(self):
        """Create a synthetic background for demo."""
        background = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some gradient for visual appeal
        for y in range(480):
            for x in range(640):
                background[y, x] = [
                    int(30 + 20 * x / 640),  # Red gradient
                    int(30 + 20 * y / 480),  # Green gradient
                    50                       # Blue constant
                ]
        
        # Add some pattern
        cv2.rectangle(background, (0, 0), (640, 80), (20, 20, 40), -1)
        cv2.rectangle(background, (0, 400), (640, 480), (20, 20, 40), -1)
        
        return background
    
    def generate_demo_hand_position(self):
        """Generate synthetic hand position based on demo mode."""
        current_time = time.time()
        
        if self.demo_mode == "circle":
            # Circular motion
            self.angle += 0.05
            x = self.center_x + self.radius * math.cos(self.angle)
            y = self.center_y + self.radius * math.sin(self.angle) * 0.7
            
        elif self.demo_mode == "line":
            # Back and forth motion
            t = current_time * 0.8
            x = self.center_x + 150 * math.sin(t)
            y = self.center_y + 50 * math.cos(t * 1.5)
            
        elif self.demo_mode == "zigzag":
            # Zigzag pattern
            t = current_time * 1.2
            x = self.center_x + 200 * math.sin(t) * math.cos(t * 3)
            y = self.center_y + 150 * math.cos(t * 2)
            
        elif self.demo_mode == "still":
            # Almost still (for easter egg trigger)
            x = self.center_x + 5 * math.sin(current_time * 0.1)
            y = self.center_y + 5 * math.cos(current_time * 0.1)
            
        else:  # Default to circle
            self.angle += 0.03
            x = self.center_x + self.radius * math.cos(self.angle)
            y = self.center_y + self.radius * math.sin(self.angle) * 0.5
        
        # Add some noise
        noise_x = np.random.normal(0, 2)
        noise_y = np.random.normal(0, 2)
        
        x += noise_x
        y += noise_y
        
        # Keep within bounds
        x = max(50, min(self.width - 50, x))
        y = max(50, min(self.height - 50, y))
        
        return (int(x), int(y))
    
    def apply_kalman_filter(self, x, y):
        """Apply Kalman filtering to smooth hand position."""
        filtered_x = self.kalman_x.update(x)
        filtered_y = self.kalman_y.update(y)
        return int(filtered_x), int(filtered_y)
    
    def calculate_distance_to_cube(self, hand_pos, cube_vertices):
        """Calculate minimum distance from hand to cube vertices."""
        if hand_pos is None:
            return float('inf')
        
        min_distance = float('inf')
        hx, hy = hand_pos
        
        for vertex in cube_vertices:
            vx, vy, _ = vertex
            distance = math.sqrt((hx - vx) ** 2 + (hy - vy) ** 2)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def classify_danger_state(self, distance):
        """Classify danger state based on distance."""
        if distance > 120:
            return "SAFE"
        elif 60 < distance <= 120:
            return "WARNING"
        else:
            return "DANGER"
    
    def apply_hysteresis(self, new_state):
        """Apply 3-frame median filter to prevent state flicker."""
        self.state_history.append(new_state)
        
        if len(self.state_history) > 3:
            self.state_history.pop(0)
        
        if len(self.state_history) >= 3:
            # Return median of last 3 states
            return sorted(self.state_history)[1]
        
        return new_state
    
    def check_easter_egg(self, hand_pos):
        """Check if easter egg should be triggered."""
        if self.demo_mode != "still":
            self.still_frame_count = 0
            self.last_hand_position = None
            return False
        
        if hand_pos is None:
            self.still_frame_count = 0
            self.last_hand_position = None
            return False
        
        if self.last_hand_position is not None:
            distance = math.sqrt(
                (hand_pos[0] - self.last_hand_position[0]) ** 2 + 
                (hand_pos[1] - self.last_hand_position[1]) ** 2
            )
            
            if distance < 10:  # Hand is very still
                self.still_frame_count += 1
            else:
                self.still_frame_count = 0
        
        self.last_hand_position = hand_pos
        
        if self.still_frame_count > 5 * 15:  # 5 seconds at 15 FPS
            return True
        
        return False
    
    def update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def draw_demo_hud(self, frame, state, hand_pos, cube_vertices, distance, easter_egg_active):
        """Draw demo-specific HUD."""
        height, width = frame.shape[:2]
        
        # Create pulsing effect for danger state
        pulse_factor = 0.5 + 0.5 * math.sin(time.time() * 3)
        
        # State colors and banners
        if state == "SAFE":
            color = (0, 255, 0)  # Green
            banner_text = "SAFE"
        elif state == "WARNING":
            color = (0, 255, 255)  # Yellow
            banner_text = "WARNING"
        else:  # DANGER
            color = (0, 0, 255)  # Red
            banner_text = "DANGER DANGER"
        
        # Semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Banner text
        font_scale = 2.0 if state == "DANGER" else 1.5
        font_thickness = 3 if state == "DANGER" else 2
        
        if state == "DANGER":
            # Pulsing effect for danger text
            alpha = 0.7 + 0.3 * pulse_factor
            text_color = (0, 0, 255)
        else:
            alpha = 0.9
            text_color = color
        
        text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 50
        
        cv2.putText(frame, banner_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        
        # Demo mode indicator
        demo_text = f"DEMO MODE: {self.demo_mode.upper()}"
        cv2.putText(frame, demo_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS counter
        fps_text = f"FPS: {self.current_fps}"
        cv2.putText(frame, fps_text, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Distance info
        dist_text = f"Distance: {int(distance)} px"
        cv2.putText(frame, dist_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        if not easter_egg_active:
            instruction_text = "Demo: Watch the hand simulate movement"
            cv2.putText(frame, instruction_text, (10, height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Arvyax logo
        logo_text = "Arvyax"
        cv2.putText(frame, logo_text, (10, height - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return frame
    
    def run_demo(self):
        """Run the demo mode."""
        print("🎭 HAND-OF-SAURON AR - DEMO MODE")
        print("=" * 50)
        print()
        print("📹 This is a demonstration mode with simulated hand tracking")
        print("🎮 Watch the synthetic hand interact with the holographic cube")
        print()
        print("🎯 Demo Modes:")
        print("   1. Circle - Circular hand movement")
        print("   2. Line - Back and forth motion")
        print("   3. Zigzag - Complex movement pattern")
        print("   4. Still - For easter egg trigger")
        print()
        print("⚡ The system automatically switches modes every 10 seconds")
        print("🛑 Press 'q' to quit")
        print("=" * 50)
        print()
        
        mode_switch_time = time.time()
        
        while True:
            # Create frame
            frame = self.background.copy()
            
            # Switch demo mode every 10 seconds
            if time.time() - mode_switch_time > 10:
                self.switch_demo_mode()
                mode_switch_time = time.time()
            
            # Generate demo hand position
            hand_pos = self.generate_demo_hand_position()
            
            # Apply Kalman filtering
            filtered_x, filtered_y = self.apply_kalman_filter(hand_pos[0], hand_pos[1])
            hand_pos = (filtered_x, filtered_y)
            
            # Calculate fingertip (offset from palm)
            fingertip_pos = (hand_pos[0], hand_pos[1] - 30)
            
            # Check for easter egg
            easter_egg_active = False
            if self.check_easter_egg(hand_pos):
                easter_egg_active = True
                self.easter_egg_triggered = True
                print("🎉 Easter egg triggered! Arvyax monogram activated!")
            
            # Render cube or monogram
            if easter_egg_active and self.easter_egg_triggered:
                # Show Arvyax monogram
                frame = self.cube_renderer.draw_arvyax_monogram(frame)
            else:
                # Render normal cube
                cube_vertices = self.cube_renderer.render_cube(frame)
                
                # Calculate distance to cube
                distance = self.calculate_distance_to_cube(hand_pos, cube_vertices)
                
                # Classify state
                new_state = self.classify_danger_state(distance)
                current_state = self.apply_hysteresis(new_state)
                
                # Draw hand visualization
                cv2.circle(frame, hand_pos, 10, (255, 0, 0), -1)  # Palm
                cv2.circle(frame, fingertip_pos, 5, (0, 255, 255), -1)  # Fingertip
                cv2.line(frame, hand_pos, fingertip_pos, (255, 255, 0), 2)  # Connection
                
                # Draw HUD
                frame = self.draw_demo_hud(frame, current_state, hand_pos, cube_vertices, distance, easter_egg_active)
            
            # Update FPS
            self.update_fps()
            
            # Display frame
            cv2.imshow('Hand-of-Sauron AR - Demo Mode', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Control frame rate
            target_delay = 1.0 / self.target_fps
            time.sleep(target_delay)
        
        self.cleanup()
    
    def switch_demo_mode(self):
        """Switch to next demo mode."""
        modes = ["circle", "line", "zigzag", "still"]
        current_index = modes.index(self.demo_mode)
        next_index = (current_index + 1) % len(modes)
        self.demo_mode = modes[next_index]
        
        print(f"🔄 Switched to {self.demo_mode} mode")
        
        # Reset easter egg when switching modes
        if self.demo_mode != "still":
            self.easter_egg_triggered = False
    
    def cleanup(self):
        """Clean up resources."""
        cv2.destroyAllWindows()
        print("Demo mode stopped")


def main():
    """Run demo mode."""
    try:
        detector = DemoHandDangerDetector()
        detector.run_demo()
        return 0
    except Exception as e:
        print(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())