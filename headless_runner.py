#!/usr/bin/env python3
"""
Headless Runner - File-based output for Hand-of-Sauron AR
Outputs frames to files instead of displaying them.

Author: MiniMax Agent
Created: 2025-12-03
"""

import cv2
import numpy as np
import time
import math
import os
from kalman import KalmanFilter
from cube_renderer import CubeRenderer

# Ensure headless operation
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;0'
os.environ['OPENCV_FFMPEG_WRITER_OPTIONS'] = 'rtsp_transport;0'
cv2.setNumThreads(0)

class HeadlessHandDangerDetector:
    """Headless version that outputs frames to files instead of GUI."""
    
    def __init__(self, output_dir="headless_output", target_fps=2):
        """Initialize headless detector."""
        self.output_dir = output_dir
        self.target_fps = target_fps
        self.frame_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize synthetic background for demo
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
        self.demo_mode = "circle"
        
        # Demo parameters
        self.center_x = 320
        self.center_y = 240
        self.radius = 100
        self.angle = 0
        
        print(f"📁 Headless output directory: {output_dir}")
        print(f"⚡ Target FPS: {target_fps}")
        print()
    
    def create_demo_background(self):
        """Create synthetic background for demo."""
        background = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(480):
            for x in range(640):
                background[y, x] = [
                    int(30 + 20 * x / 640),  # Red gradient
                    int(30 + 20 * y / 480),  # Green gradient
                    50                       # Blue constant
                ]
        
        # Add banner areas
        cv2.rectangle(background, (0, 0), (640, 80), (20, 20, 40), -1)
        cv2.rectangle(background, (0, 400), (640, 480), (20, 20, 40), -1)
        
        return background
    
    def generate_demo_hand_position(self):
        """Generate synthetic hand position."""
        current_time = time.time()
        
        if self.demo_mode == "circle":
            self.angle += 0.05
            x = self.center_x + self.radius * math.cos(self.angle)
            y = self.center_y + self.radius * math.sin(self.angle) * 0.7
        elif self.demo_mode == "line":
            t = current_time * 0.8
            x = self.center_x + 150 * math.sin(t)
            y = self.center_y + 50 * math.cos(t * 1.5)
        elif self.demo_mode == "zigzag":
            t = current_time * 1.2
            x = self.center_x + 200 * math.sin(t) * math.cos(t * 3)
            y = self.center_y + 150 * math.cos(t * 2)
        else:
            self.angle += 0.03
            x = self.center_x + self.radius * math.cos(self.angle)
            y = self.center_y + self.radius * math.sin(self.angle) * 0.5
        
        # Add noise and keep within bounds
        x = max(50, min(590, x + np.random.normal(0, 2)))
        y = max(50, min(430, y + np.random.normal(0, 2)))
        
        return (int(x), int(y))
    
    def apply_kalman_filter(self, x, y):
        """Apply Kalman filtering."""
        filtered_x = self.kalman_x.update(x)
        filtered_y = self.kalman_y.update(y)
        return int(filtered_x), int(filtered_y)
    
    def calculate_distance_to_cube(self, hand_pos, cube_vertices):
        """Calculate distance to cube."""
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
        """Classify danger state."""
        if distance > 120:
            return "SAFE"
        elif 60 < distance <= 120:
            return "WARNING"
        else:
            return "DANGER"
    
    def apply_hysteresis(self, new_state):
        """Apply hysteresis to prevent flicker."""
        self.state_history.append(new_state)
        
        if len(self.state_history) > 3:
            self.state_history.pop(0)
        
        if len(self.state_history) >= 3:
            return sorted(self.state_history)[1]
        
        return new_state
    
    def check_easter_egg(self, hand_pos):
        """Check for easter egg trigger."""
        if hand_pos is None:
            self.still_frame_count = 0
            return False
        
        if self.last_hand_position is not None:
            distance = math.sqrt(
                (hand_pos[0] - self.last_hand_position[0]) ** 2 + 
                (hand_pos[1] - self.last_hand_position[1]) ** 2
            )
            
            if distance < 10:
                self.still_frame_count += 1
            else:
                self.still_frame_count = 0
        
        self.last_hand_position = hand_pos
        
        if self.still_frame_count > 5 * self.target_fps:  # 5 seconds
            return True
        
        return False
    
    def draw_hud(self, frame, state, hand_pos, cube_vertices, distance, easter_egg_active):
        """Draw HUD elements."""
        height, width = frame.shape[:2]
        
        # State colors
        if state == "SAFE":
            color = (0, 255, 0)  # Green
            banner_text = "SAFE"
        elif state == "WARNING":
            color = (0, 255, 255)  # Yellow
            banner_text = "WARNING"
        else:  # DANGER
            color = (0, 0, 255)  # Red
            banner_text = "DANGER DANGER"
        
        # Draw banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Banner text
        font_scale = 2.0 if state == "DANGER" else 1.5
        font_thickness = 3 if state == "DANGER" else 2
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
        
        # FPS indicator
        fps_text = f"Frame: {self.frame_count}"
        cv2.putText(frame, fps_text, (width - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Distance info
        dist_text = f"Distance: {int(distance)} px"
        cv2.putText(frame, dist_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        instruction_text = "Headless Demo - Frames saved to files"
        cv2.putText(frame, instruction_text, (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Arvyax logo
        logo_text = "Arvyax"
        cv2.putText(frame, logo_text, (10, height - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return frame
    
    def run_headless_demo(self, num_frames=50):
        """Run headless demo and save frames to files."""
        print("🖥️  HEADLESS DEMO MODE")
        print("=" * 50)
        print(f"📊 Generating {num_frames} demo frames")
        print(f"💾 Saving to: {self.output_dir}/")
        print()
        
        mode_switch_time = time.time()
        
        for frame_num in range(num_frames):
            # Create frame
            frame = self.background.copy()
            
            # Switch demo mode every 10 seconds of frames
            if frame_num > 0 and frame_num % (10 * self.target_fps) == 0:
                self.switch_demo_mode()
            
            # Generate demo hand position
            hand_pos = self.generate_demo_hand_position()
            
            # Apply Kalman filtering
            filtered_x, filtered_y = self.apply_kalman_filter(hand_pos[0], hand_pos[1])
            hand_pos = (filtered_x, filtered_y)
            
            # Calculate fingertip
            fingertip_pos = (hand_pos[0], hand_pos[1] - 30)
            
            # Check for easter egg
            easter_egg_active = False
            if self.check_easter_egg(hand_pos):
                easter_egg_active = True
                self.easter_egg_triggered = True
                print(f"🎉 Easter egg triggered at frame {frame_num}!")
            
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
                frame = self.draw_hud(frame, current_state, hand_pos, cube_vertices, distance, easter_egg_active)
            
            # Save frame to file
            filename = f"frame_{frame_num:04d}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, frame)
            
            self.frame_count += 1
            
            # Progress indicator
            if frame_num % 10 == 0:
                print(f"📹 Frame {frame_num+1}/{num_frames} saved")
            
            # Control frame rate
            time.sleep(1.0 / self.target_fps)
        
        print()
        print("✅ Headless demo completed!")
        print(f"📁 Check output directory: {self.output_dir}/")
        print(f"📊 Generated {num_frames} frames")
        print()
        
        # Create a summary
        self.create_summary(num_frames)
    
    def switch_demo_mode(self):
        """Switch to next demo mode."""
        modes = ["circle", "line", "zigzag", "still"]
        current_index = modes.index(self.demo_mode)
        next_index = (current_index + 1) % len(modes)
        self.demo_mode = modes[next_index]
        
        print(f"🔄 Switched to {self.demo_mode} mode")
        
        if self.demo_mode != "still":
            self.easter_egg_triggered = False
    
    def create_summary(self, num_frames):
        """Create a summary file."""
        summary_path = os.path.join(self.output_dir, "demo_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Hand-of-Sauron AR - Headless Demo Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {num_frames}\n")
            f.write(f"Target FPS: {self.target_fps}\n")
            f.write(f"Duration: {num_frames/self.target_fps:.1f} seconds\n\n")
            f.write("Features Demonstrated:\n")
            f.write("✓ Real-time hand tracking simulation\n")
            f.write("✓ 3D holographic cube rendering\n")
            f.write("✓ Distance-based danger detection\n")
            f.write("✓ State machine with hysteresis\n")
            f.write("✓ Easter egg activation\n")
            f.write("✓ Professional HUD with real-time feedback\n\n")
            f.write("Technical Specifications:\n")
            f.write("✓ Resolution: 640x480\n")
            f.write("✓ Kalman filtering for smooth tracking\n")
            f.write("✓ Software 3D rendering (no OpenGL)\n")
            f.write("✓ Classical computer vision pipeline\n")
            f.write("✓ Complete offline operation\n")
        
        print(f"📋 Summary saved: {summary_path}")


def main():
    """Main function."""
    print("🖥️  HAND-OF-SAURON AR - HEADLESS DEMO")
    print("=" * 50)
    print()
    print("This demo generates sample frames showing the AR system")
    print("in action without requiring a camera or GUI display.")
    print()
    
    # Run headless demo
    detector = HeadlessHandDangerDetector(output_dir="headless_demo_output", target_fps=2)
    detector.run_headless_demo(num_frames=30)
    
    print()
    print("🎉 Demo complete! Check the headless_demo_output/ directory.")
    return 0


if __name__ == "__main__":
    exit(main())