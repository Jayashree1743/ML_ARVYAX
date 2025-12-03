#!/usr/bin/env python3
"""
3D Cube Renderer
Software-based 3D rendering for holographic cube effects without OpenGL dependencies.

Author: MiniMax Agent
Created: 2025-12-03
"""

import numpy as np
import time
import math
import cv2
import math
import time


class CubeRenderer:
    """
    Software 3D cube renderer with rotation and projection.
    
    Features:
    - 3D cube with 8 vertices and 12 edges
    - Rotation in all three axes
    - Perspective projection to 2D screen
    - Wireframe visualization
    - Arvyax monogram rendering for easter egg
    """
    
    def __init__(self, cube_size=60, distance=300):
        """
        Initialize cube renderer.
        
        Args:
            cube_size: Size of the cube in pixels
            distance: Distance from camera in pixels
        """
        self.cube_size = cube_size
        self.distance = distance
        
        # Define 8 vertices of a cube centered at origin
        half_size = cube_size // 2
        self.vertices_3d = np.array([
            [-half_size, -half_size, -half_size],  # 0
            [ half_size, -half_size, -half_size],  # 1
            [ half_size,  half_size, -half_size],  # 2
            [-half_size,  half_size, -half_size],  # 3
            [-half_size, -half_size,  half_size],  # 4
            [ half_size, -half_size,  half_size],  # 5
            [ half_size,  half_size,  half_size],  # 6
            [-half_size,  half_size,  half_size]   # 7
        ], dtype=np.float32)
        
        # Define edges (pairs of vertex indices)
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        # Rotation angles
        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0
        
        # Rotation speed
        self.rotation_speed = 0.02
    
    def rotate_vertex(self, vertex, angle_x, angle_y, angle_z):
        """
        Apply rotation matrix to a 3D vertex.
        
        Args:
            vertex: 3D vertex coordinates [x, y, z]
            angle_x: Rotation angle around X-axis (radians)
            angle_y: Rotation angle around Y-axis (radians)  
            angle_z: Rotation angle around Z-axis (radians)
            
        Returns:
            Rotated 3D vertex
        """
        x, y, z = vertex
        
        # Rotation around X-axis
        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
        y = y * cos_x - z * sin_x
        z = y * sin_x + z * cos_x
        
        # Rotation around Y-axis
        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        x = x * cos_y + z * sin_y
        z = -x * sin_y + z * cos_y
        
        # Rotation around Z-axis
        cos_z, sin_z = math.cos(angle_z), math.sin(angle_z)
        x = x * cos_z - y * sin_z
        y = x * sin_z + y * cos_z
        
        return np.array([x, y, z])
    
    def project_to_2d(self, vertex_3d, width, height, focal_length=500):
        """
        Project 3D vertex to 2D screen coordinates using perspective projection.
        
        Args:
            vertex_3d: 3D vertex coordinates
            width: Screen width
            height: Screen height
            focal_length: Focal length for perspective projection
            
        Returns:
            2D screen coordinates [x, y]
        """
        x, y, z = vertex_3d
        
        # Add distance to move cube away from camera
        z += self.distance
        
        # Avoid division by zero
        if z <= 0:
            z = 1
        
        # Perspective projection
        scale = focal_length / z
        
        # Center the cube on screen
        screen_x = int(x * scale + width // 2)
        screen_y = int(-y * scale + height // 2)  # Flip Y coordinate
        
        return np.array([screen_x, screen_y, z])
    
    def render_cube(self, frame):
        """
        Render the 3D cube on the given frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of 2D projected vertices
        """
        height, width = frame.shape[:2]
        
        # Update rotation angles for animation
        self.angle_x += self.rotation_speed
        self.angle_y += self.rotation_speed * 0.7
        self.angle_z += self.rotation_speed * 0.3
        
        # Rotate and project all vertices
        projected_vertices = []
        for vertex in self.vertices_3d:
            rotated = self.rotate_vertex(vertex, self.angle_x, self.angle_y, self.angle_z)
            projected = self.project_to_2d(rotated, width, height)
            projected_vertices.append(projected)
        
        # Draw wireframe edges
        for start_idx, end_idx in self.edges:
            start_point = projected_vertices[start_idx][:2].astype(int)
            end_point = projected_vertices[end_idx][:2].astype(int)
            
            # Calculate depth-based brightness
            avg_depth = (projected_vertices[start_idx][2] + projected_vertices[end_idx][2]) / 2
            brightness = max(0.3, min(1.0, 500 / avg_depth))
            
            # Draw line with holographic effect
            color = (int(0 * brightness), int(255 * brightness), int(100 * brightness))
            thickness = 2
            
            cv2.line(frame, tuple(start_point), tuple(end_point), color, thickness)
            
            # Add glow effect for closer edges
            if brightness > 0.7:
                cv2.line(frame, tuple(start_point), tuple(end_point), 
                        (0, 150, 200), thickness + 1)
        
        # Draw vertices
        for vertex in projected_vertices:
            point = vertex[:2].astype(int)
            depth = vertex[2]
            brightness = max(0.3, min(1.0, 500 / depth))
            
            color = (int(100 * brightness), int(200 * brightness), int(255 * brightness))
            cv2.circle(frame, tuple(point), 3, color, -1)
        
        return projected_vertices
    
    def draw_arvyax_monogram(self, frame):
        """
        Draw the Arvyax "A" monogram for the easter egg.
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with monogram drawn
        """
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Animate the monogram
        try:
            scale = 1.0 + 0.2 * math.sin(time.time() * 2)
        except:
            scale = 1.0  # Fallback if time module issues
        
        # Define monogram points (stylized 'A')
        monogram_size = 80 * scale
        half_size = monogram_size // 2
        
        # Create the 'A' shape points
        points = np.array([
            [center_x - half_size, center_y + half_size],    # Bottom left
            [center_x, center_y - half_size],                # Top center
            [center_x + half_size, center_y + half_size],    # Bottom right
            [center_x - half_size//3, center_y],             # Left inner
            [center_x + half_size//3, center_y],             # Right inner
        ], np.int32)
        
        # Draw outer 'A' shape
        cv2.polylines(frame, [points], True, (0, 255, 100), 3)
        
        # Draw crossbar with proper type conversion and validation
        try:
            if not (math.isinf(center_x) or math.isinf(center_y)):
                crossbar_start = (int(center_x - half_size//3), int(center_y))
                crossbar_end = (int(center_x + half_size//3), int(center_y))
                cv2.line(frame, crossbar_start, crossbar_end, (0, 255, 100), 3)
        except (ValueError, TypeError):
            # Skip drawing if coordinates are invalid
            pass
        cv2.line(frame, crossbar_start, crossbar_end, (0, 255, 100), 3)
        
        # Add glow effect
        glow_color = (100, 255, 200)
        cv2.polylines(frame, [points], True, glow_color, 1)
        cv2.line(frame, crossbar_start, crossbar_end, glow_color, 1)
        
        # Add pulsing text below
        text = "ARVYGX"
        font_scale = 1.5
        font_thickness = 2
        
        # Pulsing effect
        pulse = 0.7 + 0.3 * math.sin(time.time() * 3)
        font_scale_pulsed = font_scale * pulse
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_pulsed, font_thickness)[0]
        text_x = int(center_x - text_size[0] // 2)
        text_y = int(center_y + half_size + 30)
        
        cv2.putText(frame, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale_pulsed, (0, 255, 150), font_thickness)
        
        return frame
    
    def set_cube_size(self, size):
        """Update cube size."""
        self.cube_size = size
        half_size = size // 2
        self.vertices_3d = np.array([
            [-half_size, -half_size, -half_size],
            [ half_size, -half_size, -half_size],
            [ half_size,  half_size, -half_size],
            [-half_size,  half_size, -half_size],
            [-half_size, -half_size,  half_size],
            [ half_size, -half_size,  half_size],
            [ half_size,  half_size,  half_size],
            [-half_size,  half_size,  half_size]
        ], dtype=np.float32)
    
    def set_distance(self, distance):
        """Update cube distance from camera."""
        self.distance = distance
    
    def set_rotation_speed(self, speed):
        """Update rotation speed."""
        self.rotation_speed = speed