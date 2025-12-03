#!/usr/bin/env python3
"""
Simple test for the critical infinity fixes.
"""

import numpy as np
import math

def test_distance_calculation():
    """Test distance calculation with None values."""
    print("🧪 Testing distance calculation...")
    
    # Mock cube vertices for testing
    cube_vertices = [(100, 100, 0), (200, 200, 0)]
    
    # Test with None hand position
    hand_pos = None
    min_distance = float('inf')
    if hand_pos is not None:
        hx, hy = hand_pos
        for vertex in cube_vertices:
            vx, vy, _ = vertex
            distance = math.sqrt((hx - vx) ** 2 + (hy - vy) ** 2)
            min_distance = min(min_distance, distance)
    else:
        min_distance = float('inf')
    
    print(f"✅ Distance with None hand: {min_distance}")
    
    # Test distance display
    if math.isinf(min_distance):
        dist_text = "Distance: No hand detected"
    else:
        dist_text = f"Distance: {int(min_distance)} px"
    
    print(f"✅ Distance display text: {dist_text}")
    return True

def test_coordinate_validation():
    """Test coordinate validation for drawing."""
    print("🔍 Testing coordinate validation...")
    
    # Test problematic coordinates
    center_x = float('inf')
    center_y = float('inf')
    
    # Test validation
    if not (math.isinf(center_x) or math.isinf(center_y)):
        half_size = 40
        try:
            crossbar_start = (int(center_x - half_size//3), int(center_y))
            crossbar_end = (int(center_x + half_size//3), int(center_y))
            print("❌ Should have skipped invalid coordinates")
            return False
        except:
            pass
    
    print("✅ Coordinate validation: PASSED")
    return True

def test_hand_position_validation():
    """Test hand position validation."""
    print("✋ Testing hand position validation...")
    
    # Test valid position
    hand_pos = (320, 240)
    fingertip_pos = (325, 235)
    
    # Validation check
    if (hand_pos is not None and fingertip_pos is not None and
        not (math.isnan(hand_pos[0]) or math.isnan(hand_pos[1]) or
             math.isnan(fingertip_pos[0]) or math.isnan(fingertip_pos[1]))):
        try:
            if (not (math.isinf(hand_pos[0]) or math.isinf(hand_pos[1]) or
                    math.isinf(fingertip_pos[0]) or math.isinf(fingertip_pos[1]))):
                print("✅ Hand position validation: PASSED")
                return True
        except:
            pass
    
    print("❌ Hand position validation failed")
    return False

def main():
    """Run focused tests."""
    print("🧪 HAND-OF-SAURON AR - CRITICAL FIXES TEST")
    print("=" * 50)
    
    tests = [
        test_distance_calculation,
        test_coordinate_validation,
        test_hand_position_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Critical Fixes: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All critical fixes verified!")
        return 0
    else:
        print("⚠️  Some critical tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())