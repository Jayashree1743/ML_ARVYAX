#!/usr/bin/env python3
"""
Hand-of-Sauron AR - Hand Danger Detection System
A webcam-based AR experience that tracks hand movement and creates
interactive 3D holographic effects.

Author: MiniMax Agent
Created: 2025-12-03
"""

import sys
import time
import math
import os

# Guard imports to provide clearer, actionable errors when binary
# extension compatibility issues occur (e.g. NumPy 2 vs compiled C extensions).
def _import_or_exit(module_name, hint=None):
    try:
        return __import__(module_name)
    except Exception as exc:  # ImportError or binary incompatibility
        print(f"Error importing '{module_name}': {exc}")
        if hint:
            print(hint)
        print("\nSuggested fix:")
        print("1. Create and activate a virtual environment:")
        print("   python3 -m venv .venv && source .venv/bin/activate")
        print("2. Install pinned dependencies:")
        print("   pip install -r requirements.txt")
        print("3. If you still see binary incompatibility, try downgrading NumPy:")
        print("   pip install 'numpy<2'")
        sys.exit(1)

# Import numpy and cv2 with helpful guidance on failure
np = _import_or_exit('numpy', hint="This project requires a NumPy build compatible with binary extensions used by OpenCV.")
cv2 = _import_or_exit('cv2', hint="OpenCV (cv2) failed to import. Ensure you installed 'opencv-python' compatible with your NumPy.")

from kalman import KalmanFilter
from cube_renderer import CubeRenderer

# Set OpenCV to use headless backend for better compatibility
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;0'
os.environ['OPENCV_FFMPEG_WRITER_OPTIONS'] = 'rtsp_transport;0'

# Disable Qt GUI backend issues
cv2.setNumThreads(0)  # Prevent threading issues


class HandDangerDetector:
    """
    Core class for hand tracking and danger detection system.
    
    Features:
    - Background subtraction for hand detection
    - Skin tone segmentation in YCbCr color space
    - Kalman filtering for smooth tracking
    - 3D cube rendering and interaction
    - Real-time danger state classification
    """
    
    def __init__(self, camera_id=0, target_fps=15):
        """Initialize the hand detection system."""
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_count = 0
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Background subtractor (30-frame bootstrap)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=30, detectShadows=False
        )
        
        # Skin tone detection in YCbCr space
        self.skin_range = {
            'lower_cb': np.array([77]),
            'upper_cb': np.array([127]),
            'lower_cr': np.array([133]),
            'upper_cr': np.array([173])
        }
        
        # Kalman filter for smoothing hand position
        self.kalman_x = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)
        self.kalman_y = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)
        
        # Cube renderer
        self.cube_renderer = CubeRenderer()

        # Processing scale: operate on smaller frames for speed (map back to display)
        # 1.0 = full resolution, 0.5 = half resolution (recommended)
        self.proc_scale = 0.5
        
        # State management
        self.current_state = "SAFE"
        self.state_history = []
        self.easter_egg_triggered = False
        self.still_frame_count = 0
        self.last_hand_position = None

        # Automatic recorder for demo highlights when DANGER occurs
        self.record_on_danger = True
        self.record_duration = 15  # seconds; default 15 (can be changed)
        self.recording = False
        self.record_start_time = None
        self.video_writer = None
        self.recordings_dir = os.path.join(os.getcwd(), 'recordings')
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def detect_skin_mask(self, frame):
        """Create skin tone mask using YCbCr color space."""
        # Convert BGR to YCbCr
        ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        
        # Create skin mask
        skin_mask = cv2.inRange(cr, self.skin_range['lower_cr'], self.skin_range['upper_cr'])
        skin_mask2 = cv2.inRange(cb, self.skin_range['lower_cb'], self.skin_range['upper_cb'])
        
        # Combine masks (bitwise AND)
        skin_mask = cv2.bitwise_and(skin_mask, skin_mask2)
        
        return skin_mask
    
    def get_hand_contour(self, frame):
        """Extract largest hand contour from frame."""
        # Downscale frame for faster processing
        if self.proc_scale != 1.0:
            proc_h = int(frame.shape[0] * self.proc_scale)
            proc_w = int(frame.shape[1] * self.proc_scale)
            small = cv2.resize(frame, (proc_w, proc_h))
        else:
            small = frame.copy()

        # Get background mask (on smaller frame)
        bg_mask = self.bg_subtractor.apply(small)

        # Get skin mask (on smaller frame)
        skin_mask = self.detect_skin_mask(small)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(bg_mask, skin_mask)
        
        # Morphological operations for cleanup
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
        
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Ignore tiny contours
        if cv2.contourArea(largest_contour) < 300:
            return None, None, None
        
        # Get convex hull
        hull = cv2.convexHull(largest_contour)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            return None, None, None
        
        # Try to detect fingertip using convexity defects (more robust)
        fingertip = None
        try:
            hull_idx = cv2.convexHull(largest_contour, returnPoints=False)
            if hull_idx is not None and len(hull_idx) > 3:
                defects = cv2.convexityDefects(largest_contour, hull_idx)
                if defects is not None:
                    # Find defect with the largest depth and choose its start point as fingertip candidate
                    max_depth = 0
                    candidate = None
                    for i in range(defects.shape[0]):
                        s, e, f, depth = defects[i, 0]
                        depth = depth / 256.0  # OpenCV stores depth * 256
                        if depth > max_depth:
                            max_depth = depth
                            start = tuple(largest_contour[s][0])
                            end = tuple(largest_contour[e][0])
                            far = tuple(largest_contour[f][0])
                            candidate = start
                    # Only accept candidate if it's reasonably far from centroid
                    if candidate is not None:
                        cand_dist = math.hypot(candidate[0] - cx, candidate[1] - cy)
                        if cand_dist > 20:
                            fingertip = candidate
        except Exception:
            fingertip = None

        # Fallback: farthest hull point from centroid
        if fingertip is None:
            max_dist = 0
            farthest_point = (cx, cy)
            for point in hull:
                px, py = point[0]
                dist = math.hypot(px - cx, py - cy)
                if dist > max_dist:
                    max_dist = dist
                    farthest_point = (px, py)
            fingertip = farthest_point
        
        # Map coordinates back to original frame size if processed on small frame
        if self.proc_scale != 1.0:
            scale_inv = 1.0 / self.proc_scale
            cx = int(cx * scale_inv)
            cy = int(cy * scale_inv)
            fingertip = (int(fingertip[0] * scale_inv), int(fingertip[1] * scale_inv))
            # Also scale mask up for debugging/visualization
            combined_mask = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0]))

        return (cx, cy), fingertip, combined_mask
    
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
        """Check if easter egg should be triggered (hand still for 5 seconds)."""
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
            # Keep a small history to smooth FPS display
            self.current_fps = self.fps_counter
            if not hasattr(self, 'fps_history'):
                self.fps_history = []
            self.fps_history.append(self.fps_counter)
            if len(self.fps_history) > 5:
                self.fps_history.pop(0)
            # moving average
            self.current_fps = int(sum(self.fps_history) / len(self.fps_history))
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def draw_hud(self, frame, state, hand_pos, cube_vertices, danger_distance, easter_egg_active):
        """Draw heads-up display with state indicators."""
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
            # Strong pulsing, large centered DANGER DANGER overlay
            pulse = 0.6 + 0.4 * math.sin(time.time() * 4)
            big_scale = 2.2 * pulse
            big_thickness = 6
            big_text = "DANGER DANGER"
            big_size = cv2.getTextSize(big_text, cv2.FONT_HERSHEY_DUPLEX, big_scale, big_thickness)[0]
            big_x = (width - big_size[0]) // 2
            big_y = (height // 2) + big_size[1] // 2
            # Shadow/glow
            cv2.putText(frame, big_text, (big_x+4, big_y+4), cv2.FONT_HERSHEY_DUPLEX, big_scale, (0,0,0), big_thickness+2)
            cv2.putText(frame, big_text, (big_x, big_y), cv2.FONT_HERSHEY_DUPLEX, big_scale, (0,0,255), big_thickness)
            # Also draw a small banner at top to keep legacy HUD
            text_color = (0, 0, 255)
            alpha = 0.9
        else:
            alpha = 0.9
            text_color = color

        # Draw smaller banner text too
        text_size = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 50
        cv2.putText(frame, banner_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        
        # Draw FPS counter
        fps_text = f"FPS: {self.current_fps}"
        cv2.putText(frame, fps_text, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw distance info with proper handling of infinity
        if math.isinf(danger_distance):
            dist_text = "Distance: No hand detected"
        else:
            dist_text = f"Distance: {int(danger_distance)} px"
        cv2.putText(frame, dist_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw Arvyax logo (placeholder)
        logo_text = "Arvyax"
        cv2.putText(frame, logo_text, (10, height - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return frame
    
    def run(self):
        """Main application loop."""
        print("Hand-of-Sauron AR Starting...")
        print("Controls:")
        print("- Move your hand to interact with the cube")
        print("- Hold hand still for 5 seconds to trigger easter egg")
        print("- Press 'q' to quit")
        print()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect hand
                hand_pos, fingertip_pos, hand_mask = self.get_hand_contour(frame)
                
                # Apply Kalman filtering
                if hand_pos is not None:
                    filtered_x, filtered_y = self.apply_kalman_filter(hand_pos[0], hand_pos[1])
                    hand_pos = (filtered_x, filtered_y)
                
                # Check for easter egg
                easter_egg_active = False
                if self.check_easter_egg(hand_pos):
                    easter_egg_active = True
                    self.easter_egg_triggered = True
                    print("🎉 Easter egg triggered! Arvyax monogram activated!")
                
                # Render cube
                if easter_egg_active and self.easter_egg_triggered:
                    # Show Arvyax monogram instead of cube
                    frame = self.cube_renderer.draw_arvyax_monogram(frame)
                else:
                    # Render normal cube
                    cube_vertices = self.cube_renderer.render_cube(frame)
                    
                    # Calculate distance to cube
                    distance = self.calculate_distance_to_cube(hand_pos, cube_vertices)
                    
                    # Classify state
                    new_state = self.classify_danger_state(distance)
                    current_state = self.apply_hysteresis(new_state)

                    # Update recorder state: start recording on entry to DANGER
                    try:
                        prev = self.current_state
                    except Exception:
                        prev = None
                    # update stored state
                    self.current_state = current_state

                    if self.record_on_danger and current_state == 'DANGER' and not self.recording:
                        # start recording
                        try:
                            self.start_recording(frame)
                        except Exception as e:
                            print(f"Failed to start recording: {e}")

                    # If recording, write frames and check duration
                    if self.recording and self.video_writer is not None:
                        try:
                            self.video_writer.write(frame)
                            if time.time() - self.record_start_time >= self.record_duration:
                                self.stop_recording()
                        except Exception as e:
                            print(f"Recording write error: {e}")
                    
                    # Draw hand visualization
                    if (hand_pos is not None and fingertip_pos is not None and
                        not (math.isnan(hand_pos[0]) or math.isnan(hand_pos[1]) or
                             math.isnan(fingertip_pos[0]) or math.isnan(fingertip_pos[1]))):
                        # Validate positions are not infinite
                        try:
                            if (not (math.isinf(hand_pos[0]) or math.isinf(hand_pos[1]) or
                                    math.isinf(fingertip_pos[0]) or math.isinf(fingertip_pos[1]))):
                                # Draw hand centroid
                                cv2.circle(frame, hand_pos, 10, (255, 0, 0), -1)
                                
                                # Draw fingertip
                                cv2.circle(frame, fingertip_pos, 5, (0, 255, 255), -1)
                                
                                # Draw connection line
                                cv2.line(frame, hand_pos, fingertip_pos, (255, 255, 0), 2)
                        except (ValueError, TypeError):
                            # Skip drawing if positions are invalid
                            pass
                    
                    # Draw HUD
                    frame = self.draw_hud(frame, current_state, hand_pos, cube_vertices, distance, easter_egg_active)
                
                # Update FPS
                self.update_fps()
                
                # Display frame with error handling
                try:
                    cv2.imshow('Hand-of-Sauron AR', frame)
                    key = cv2.waitKey(1) & 0xFF
                except Exception as e:
                    print(f"Display error (continuing without GUI): {e}")
                    key = 27  # ESC key to exit in headless mode
                if key == ord('q'):
                    break
                
                # Control frame rate
                target_delay = 1.0 / self.target_fps
                time.sleep(target_delay)
        
        except KeyboardInterrupt:
            print("\nShutdown requested by user")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        # Ensure any active recording is stopped
        try:
            if self.recording:
                self.stop_recording()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Hand-of-Sauron AR stopped")

    def start_recording(self, frame):
        """Start recording a highlight video. Frame must be current display frame."""
        height, width = frame.shape[:2]
        # filename with timestamp
        ts = int(time.time())
        filename = f"danger_highlight_{ts}.mp4"
        path = os.path.join(self.recordings_dir, filename)

        # Try to pick a reasonable FPS for the writer
        write_fps = max(10, int(self.target_fps))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            self.video_writer = cv2.VideoWriter(path, fourcc, write_fps, (width, height))
        except Exception as e:
            self.video_writer = None
            raise

        if not self.video_writer or not self.video_writer.isOpened():
            self.video_writer = None
            raise RuntimeError('Could not open VideoWriter')

        self.recording = True
        self.record_start_time = time.time()
        print(f"Recording started: {path} for {self.record_duration}s")

    def stop_recording(self):
        """Stop the active recording and finalize the file."""
        if self.video_writer is not None:
            try:
                self.video_writer.release()
                print("Recording saved")
            except Exception as e:
                print(f"Error releasing writer: {e}")
        self.video_writer = None
        self.recording = False
        self.record_start_time = None


def main():
    """Main function to run the application with CLI options."""
    import argparse

    parser = argparse.ArgumentParser(description='Hand-of-Sauron AR demo')
    parser.add_argument('--headless', action='store_true', help='Run demo mode without camera')
    parser.add_argument('--proc-scale', type=float, default=None, help='Processing scale (0.25-1.0)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device id')
    args = parser.parse_args()

    if args.headless:
        print("🖥️  Running in headless mode - demo only")
        try:
            from demo_mode import DemoHandDangerDetector
            detector = DemoHandDangerDetector()
            detector.run_demo()
        except Exception as e:
            print(f"Demo mode failed: {e}")
        return 0

    try:
        detector = HandDangerDetector(camera_id=args.camera)
        # Apply processing scale override if provided
        if args.proc_scale is not None:
            if 0.25 <= args.proc_scale <= 1.0:
                detector.proc_scale = args.proc_scale
            else:
                print("--proc-scale must be between 0.25 and 1.0; using default")
        detector.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("\n💡 Troubleshooting:")
        print("1. If you have display issues, try: python handDanger.py --headless")
        print("2. Check that camera is not being used by other applications")
        print("3. For testing without camera, use: python demo_mode.py")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())