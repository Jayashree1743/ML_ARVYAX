#!/usr/bin/env python3
"""
Quick Fix and Run Script
Handles display issues and provides multiple run options.

Author: MiniMax Agent
Created: 2025-12-03
"""

import os
import sys
import subprocess

def check_display():
    """Check if display is available."""
    display_env = os.environ.get('DISPLAY')
    wayland_display = os.environ.get('WAYLAND_DISPLAY')
    
    if display_env:
        return True, "X11"
    elif wayland_display:
        return True, "Wayland"
    else:
        return False, "Headless"

def run_fix():
    """Apply fixes and run the application."""
    print("🔧 HAND-OF-SAURON AR - QUICK FIX")
    print("=" * 50)
    
    # Check display
    has_display, display_type = check_display()
    print(f"📱 Display detected: {display_type}")
    
    if not has_display:
        print("⚠️  No display detected - running in headless mode")
        print()
        print("🖥️  Starting Headless Demo (file output)...")
        try:
            from headless_runner import HeadlessHandDangerDetector
            detector = HeadlessHandDangerDetector()
            detector.run_headless_demo(num_frames=30)
            return True
        except Exception as e:
            print(f"❌ Headless mode failed: {e}")
            print("🔄 Trying demo mode as fallback...")
            try:
                from demo_mode import DemoHandDangerDetector
                detector = DemoHandDangerDetector()
                detector.run_demo()
                return True
            except Exception as e2:
                print(f"❌ Demo mode also failed: {e2}")
                return False
    
    print("✅ Display available - attempting to run full application")
    print()
    
    # Set environment variables to fix display issues
    os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force X11 backend
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;0'
    os.environ['OPENCV_FFMPEG_WRITER_OPTIONS'] = 'rtsp_transport;0'
    
    try:
        from handDanger import HandDangerDetector
        detector = HandDangerDetector()
        detector.run()
        return True
    except Exception as e:
        print(f"❌ Full application failed: {e}")
        print()
        print("🔄 Trying demo mode as fallback...")
        try:
            from demo_mode import DemoHandDangerDetector
            detector = DemoHandDangerDetector()
            detector.run_demo()
            return True
        except Exception as e2:
            print(f"❌ Demo mode also failed: {e2}")
            return False

def show_options():
    """Show available run options."""
    print("🎯 RUN OPTIONS:")
    print("1. python fix_and_run.py       # Auto-detect and run")
    print("2. python handDanger.py        # Full AR mode")
    print("3. python handDanger.py --headless  # Headless demo mode")
    print("4. python demo_mode.py         # Demo mode only")
    print("5. python run.py               # Launcher with checks")
    print()
    print("💡 If you get display errors, try:")
    print("   export DISPLAY=:0           # For X11")
    print("   export QT_QPA_PLATFORM=xcb  # Force X11 backend")

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        show_options()
        return 0
    
    print("🚀 HAND-OF-SAURON AR - READY TO RUN")
    print()
    
    # Try auto-detection and run
    success = run_fix()
    
    if not success:
        print()
        print("💡 Try one of these alternatives:")
        show_options()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())