#!/usr/bin/env python3
"""
Hand-of-Sauron AR Launcher
Simple launcher script with system checks and error handling.

Author: MiniMax Agent
Created: 2025-12-03
"""

import sys
import os
import subprocess

def check_requirements():
    """Check if required packages are installed."""
    required_packages = ['cv2', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nTo install, run:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_camera():
    """Check if camera is available."""
    import cv2
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible!")
        print("Please check:")
        print("   - Camera is not being used by another application")
        print("   - Camera permissions are granted")
        print("   - Camera is properly connected")
        return False
    
    cap.release()
    return True

def show_welcome():
    """Display welcome message and instructions."""
    print("=" * 60)
    print("🚀 HAND-OF-SAURON AR LAUNCHER")
    print("=" * 60)
    print()
    print("📹 Webcam-based Hand Tracking AR System")
    print("🎮 Interactive 3D Holographic Cube")
    print("⚡ Real-time Danger Detection")
    print()
    print("🎯 Instructions:")
    print("   1. Move your hand in front of the camera")
    print("   2. Watch the cube respond to your hand position")
    print("   3. Approach the cube to trigger danger states")
    print("   4. Hold still for 5s to activate easter egg")
    print("   5. Press 'q' to quit")
    print()
    print("🎨 Danger States:")
    print("   🟢 SAFE (green)    - Distance > 120px")
    print("   🟡 WARNING (yellow) - Distance 60-120px") 
    print("   🔴 DANGER (red)     - Distance < 60px")
    print()
    print("=" * 60)
    print()

def show_system_info():
    """Display system information."""
    print("🔍 System Check:")
    
    # Python version
    python_version = sys.version.split()[0]
    print(f"   Python: {python_version}")
    
    # Package versions
    import cv2
    print(f"   OpenCV: {cv2.__version__}")
    
    import numpy as np
    print(f"   NumPy: {np.__version__}")
    
    # Camera availability
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"   Camera: {width}x{height} @ {fps}fps ✅")
        cap.release()
    else:
        print("   Camera: Not available ❌")
    
    print()

def main():
    """Main launcher function."""
    try:
        # Show welcome
        show_welcome()
        
        # System checks
        print("🔍 Performing system checks...")
        
        if not check_requirements():
            return 1
        
        if not check_camera():
            return 1
        
        show_system_info()
        
        print("✅ All checks passed!")
        print()
        print("🚀 Starting Hand-of-Sauron AR...")
        print("=" * 60)
        print()
        
        # Launch main application
        from handDanger import HandDangerDetector
        
        detector = HandDangerDetector()
        detector.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Launch cancelled by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that your camera is not being used by other applications")
        print("3. Verify camera permissions are granted")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())