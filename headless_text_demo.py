#!/usr/bin/env python3
"""
Headless text-only demo for Hand-of-Sauron AR

This script avoids importing NumPy and OpenCV and instead prints
simulated frame output to stdout. Useful when binary dependencies
like OpenCV cannot be imported due to NumPy binary incompatibility.
"""

import os
import math
import time

OUTPUT_DIR = "headless_demo_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def classify_danger_state(distance):
    if distance > 120:
        return "SAFE"
    elif 60 < distance <= 120:
        return "WARNING"
    else:
        return "DANGER"

def run_demo(frames=60, target_fps=15):
    print("🎭 HAND-OF-SAURON AR - HEADLESS TEXT DEMO")
    print(f"Simulating {frames} frames at ~{target_fps} FPS")
    print("Press Ctrl+C to stop")
    print("=")

    start = time.time()
    log_lines = []

    for i in range(frames):
        # Simulate a hand distance that oscillates: closer -> danger, farther -> safe
        # Use a sinusoid between 30 and 220 px
        distance = 125 + 95 * math.sin(2 * math.pi * (i / 20.0))
        state = classify_danger_state(distance)

        # Simulate an easter-egg trigger at frame ~50
        easter = "" 
        if i == int(frames * 0.8):
            easter = "🎉 EASTER EGG TRIGGERED"

        line = f"Frame {i+1:03d}/{frames:03d} -- State: {state:7} -- Distance: {int(distance):3d} px {easter}"
        print(line)
        log_lines.append(line)

        # Maintain approx target FPS (but don't sleep too long in CI)
        time.sleep(1.0 / target_fps)

    elapsed = time.time() - start
    summary = ["\nDemo Summary:", f"Frames: {frames}", f"Elapsed: {elapsed:.2f}s", f"Avg FPS: {frames/elapsed:.2f}"]
    print("\n".join(summary))

    # Save summary and frame log
    summary_path = os.path.join(OUTPUT_DIR, "demo_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(log_lines) + "\n\n")
        f.write("\n".join(summary) + "\n")

    print(f"Saved demo summary to: {summary_path}")

if __name__ == '__main__':
    try:
        run_demo(frames=60, target_fps=15)
        exit(0)
    except KeyboardInterrupt:
        print("Demo interrupted by user")
        exit(0)