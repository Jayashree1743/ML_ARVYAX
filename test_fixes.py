#!/usr/bin/env python3
"""
Test script to verify the fixes for infinity and display errors.
"""

import sys
import os

def test_infinity_handling():
    """Test that infinity values are properly handled."""
    print("🧪 Testing infinity value handling...")
    
    try:
        # Test the cube renderer with infinity values
        from cube_renderer import CubeRenderer
        
        # Create a dummy frame
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        renderer = CubeRenderer()
        
        # Test drawing monogram with normal coordinates
        result_frame = renderer.draw_arvyax_monogram(frame.copy())
        print("✅ Normal monogram drawing: PASSED")
        
        # Test distance calculation with None hand position
        from handDanger import HandDangerDetector
        detector = HandDangerDetector()
        
        # This should not crash
        distance = detector.calculate_distance_to_cube(None, [(100, 100, 0)])
        print(f"✅ Distance with None hand position: {distance} (should be inf)")
        
        # Test distance display function with infinity
        test_frame = frame.copy()
        result_frame = detector.draw_hud(test_frame, "SAFE", None, [(100, 100, 0)], distance, False)
        print("✅ HUD with infinity distance: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Infinity handling test failed: {e}")
        return False

def test_headless_mode():
    """Test headless mode works."""
    print("🖥️  Testing headless mode...")
    
    try:
        from headless_runner import HeadlessHandDangerDetector
        detector = HeadlessHandDangerDetector()
        
        # Run a short headless demo
        detector.run_headless_demo(num_frames=5)
        print("✅ Headless mode: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Headless mode test failed: {e}")
        return False

def test_auto_detection():
    """Test auto-detection logic."""
    print("🔍 Testing auto-detection...")
    
    try:
        # Test the check_display function
        sys.path.insert(0, '/workspace')
        from fix_and_run import check_display
        
        has_display, display_type = check_display()
        print(f"✅ Display detection: {has_display}, {display_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-detection test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 HAND-OF-SAURON AR - FIX VERIFICATION")
    print("=" * 50)
    
    tests = [
        test_infinity_handling,
        test_headless_mode,
        test_auto_detection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes verified! System is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())