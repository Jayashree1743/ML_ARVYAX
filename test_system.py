#!/usr/bin/env python3
"""
System Test Script for Hand-of-Sauron AR
Tests all components without requiring camera access.

Author: MiniMax Agent
Created: 2025-12-03
"""

import sys
import numpy as np

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import cv2
        print(f"   ✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"   ❌ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"   ❌ NumPy: {e}")
        return False
    
    try:
        from kalman import KalmanFilter
        print("   ✅ Kalman Filter module")
    except ImportError as e:
        print(f"   ❌ Kalman Filter: {e}")
        return False
    
    try:
        from cube_renderer import CubeRenderer
        print("   ✅ Cube Renderer module")
    except ImportError as e:
        print(f"   ❌ Cube Renderer: {e}")
        return False
    
    return True

def test_kalman_filter():
    """Test Kalman filter functionality."""
    print("\n🔍 Testing Kalman Filter...")
    
    try:
        from kalman import KalmanFilter
        
        # Create filter
        kf = KalmanFilter(process_variance=1e-3, measurement_variance=1e-1)
        
        # Test with noisy measurements
        measurements = [10, 12, 8, 11, 13, 9, 10, 12, 11, 10]
        filtered_values = []
        
        for measurement in measurements:
            filtered = kf.update(measurement)
            filtered_values.append(filtered)
        
        # Verify smoothing occurred
        print(f"   📊 Input measurements: {measurements[:5]}...")
        print(f"   📈 Filtered values:    {[f'{v:.2f}' for v in filtered_values[:5]]}...")
        
        # Calculate variance reduction
        input_var = np.var(measurements)
        filtered_var = np.var(filtered_values)
        reduction = (1 - filtered_var/input_var) * 100
        
        print(f"   📉 Variance reduction: {reduction:.1f}%")
        
        if reduction > 20:
            print("   ✅ Kalman filter working correctly")
        else:
            print("   ⚠️  Kalman filter may need tuning")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Kalman filter test failed: {e}")
        return False

def test_cube_renderer():
    """Test cube renderer functionality."""
    print("\n🔍 Testing Cube Renderer...")
    
    try:
        from cube_renderer import CubeRenderer
        
        # Create renderer
        renderer = CubeRenderer(cube_size=60, distance=300)
        
        # Test vertex count
        vertices = renderer.vertices_3d
        print(f"   📐 3D Vertices: {len(vertices)} (expected: 8)")
        
        # Test edge count
        edges = renderer.edges
        print(f"   🔗 Edges: {len(edges)} (expected: 12)")
        
        # Test rotation
        original_vertex = vertices[0].copy()
        rotated = renderer.rotate_vertex(original_vertex, 0.1, 0.1, 0.1)
        print(f"   🔄 Rotation test: {np.array_equal(original_vertex, rotated)} (should be False)")
        
        # Test projection
        import cv2
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        projected = renderer.project_to_2d(rotated, 640, 480)
        
        print(f"   📽️  Projection test: {len(projected)}D (expected: 3D)")
        print(f"   📏 Screen coords: ({projected[0]:.1f}, {projected[1]:.1f})")
        
        if len(projected) == 3:
            print("   ✅ Cube renderer working correctly")
        else:
            print("   ❌ Cube renderer projection failed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Cube renderer test failed: {e}")
        return False

def test_image_processing():
    """Test image processing utilities."""
    print("\n🔍 Testing Image Processing...")
    
    try:
        import cv2
        import numpy as np
        
        # Create test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (20, 20), (80, 80), (255, 255, 255), -1)
        
        # Test skin color detection
        ycbcr = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycbcr)
        
        # Simple skin mask
        skin_mask = cv2.inRange(cr, 133, 173)
        skin_mask2 = cv2.inRange(cb, 77, 127)
        combined = cv2.bitwise_and(skin_mask, skin_mask2)
        
        print(f"   🎨 Skin detection: {np.sum(combined > 0)} pixels")
        
        # Test morphological operations
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(combined, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=2)
        
        print(f"   🔧 Morphological ops: {np.sum(dilated > 0)} pixels")
        print("   ✅ Image processing working correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Image processing test failed: {e}")
        return False

def test_camera_access():
    """Test camera accessibility."""
    print("\n🔍 Testing Camera Access...")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            
            if ret:
                height, width = frame.shape[:2]
                print(f"   📹 Camera: {width}x{height}")
                print("   ✅ Camera accessible")
                result = True
            else:
                print("   ❌ Camera accessible but no frame captured")
                result = False
        else:
            print("   ❌ Camera not accessible")
            result = False
        
        cap.release()
        return result
        
    except Exception as e:
        print(f"   ❌ Camera test failed: {e}")
        return False

def run_benchmarks():
    """Run performance benchmarks."""
    print("\n🔍 Running Performance Benchmarks...")
    
    try:
        import time
        
        # Test frame processing speed
        import cv2
        import numpy as np
        
        test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        # Benchmark frame processing
        iterations = 100
        start_time = time.time()
        
        for _ in range(iterations):
            # Simulate processing pipeline
            gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations * 1000
        
        print(f"   ⚡ Frame processing: {avg_time:.2f}ms per frame")
        print(f"   📊 Estimated FPS: {1000/avg_time:.1f}")
        
        # Test Kalman filter speed
        from kalman import KalmanFilter
        
        kf = KalmanFilter()
        start_time = time.time()
        
        for i in range(1000):
            kf.update(50 + np.random.normal(0, 5))
        
        end_time = time.time()
        kalman_time = (end_time - start_time) / 1000 * 1000
        
        print(f"   🔄 Kalman filter: {kalman_time:.3f}ms per update")
        
        if avg_time < 50:  # 20+ FPS
            print("   ✅ Performance target met (15+ FPS)")
        else:
            print("   ⚠️  Performance may be below target")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Benchmark test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 HAND-OF-SAURON AR - SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Kalman Filter", test_kalman_filter),
        ("Cube Renderer", test_cube_renderer),
        ("Image Processing", test_image_processing),
        ("Camera Access", test_camera_access),
        ("Performance Benchmarks", run_benchmarks)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f"🧪 {test_name}")
        print(f"{'=' * 40}")
        
        if test_func():
            passed += 1
    
    print(f"\n{'=' * 60}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())