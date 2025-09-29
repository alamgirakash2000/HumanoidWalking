"""
MuJoCo rendering utilities for the combined walking + obstacle avoidance system.

This module provides functions to add visual elements (spheres, lines) to the MuJoCo simulation
so that obstacles, goals, and collision volumes are visible during runtime.
"""
import numpy as np
import mujoco


class MujocoVisualRenderer:
    """Handles adding visual elements to MuJoCo simulation."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.visual_geom_ids = []  # Track created visual geoms for cleanup
        
    def clear_visual_geoms(self):
        """Clear all previously created visual geoms."""
        # Note: MuJoCo doesn't support runtime geom deletion, so we'll hide them
        for geom_id in self.visual_geom_ids:
            if geom_id < self.model.ngeom:
                self.model.geom_rgba[geom_id] = [0, 0, 0, 0]  # Make transparent
        self.visual_geom_ids.clear()
        
    def add_visual_sphere(self, position, radius, color, name_prefix="visual_sphere"):
        """
        Add a visual sphere to the simulation.
        
        Args:
            position: 3D position [x, y, z]
            radius: sphere radius
            color: RGBA color [r, g, b, a] 
            name_prefix: name prefix for the geom
        """
        # Since we can't add geoms at runtime, we'll use sites or mocap bodies
        # For now, let's use the data.mocap_pos to position visual elements
        
        # Store the visual info for rendering in the step function
        if not hasattr(self, 'visual_spheres'):
            self.visual_spheres = []
            
        self.visual_spheres.append({
            'position': np.array(position),
            'radius': radius,
            'color': np.array(color)
        })
        
    def render_spheres_as_mocap(self):
        """Render stored spheres using mocap bodies if available."""
        if not hasattr(self, 'visual_spheres'):
            return
            
        # Use mocap bodies if available in the model
        mocap_count = min(len(self.visual_spheres), self.model.nmocap)
        for i in range(mocap_count):
            sphere = self.visual_spheres[i]
            self.data.mocap_pos[i] = sphere['position']
            # Note: mocap bodies don't have color control, this is a limitation
            
    def update_visual_elements(self, obstacles, goals, collision_volumes):
        """
        Update all visual elements for the current timestep.
        
        Args:
            obstacles: List of obstacle info dicts
            goals: List of goal info dicts  
            collision_volumes: List of collision volume info dicts
        """
        # Clear previous visuals
        if hasattr(self, 'visual_spheres'):
            self.visual_spheres.clear()
        else:
            self.visual_spheres = []
            
        # Add obstacle spheres (red)
        for obs in obstacles:
            self.add_visual_sphere(
                obs['position'], 
                obs['radius'], 
                [1.0, 0.0, 0.0, 0.8],  # Red
                "obstacle"
            )
            
        # Add goal spheres (green)  
        for goal in goals:
            self.add_visual_sphere(
                goal['position'],
                goal['radius'],
                [0.0, 1.0, 0.0, 0.8],  # Green
                "goal"
            )
            
        # Add collision volume spheres (blue, semi-transparent)
        for vol in collision_volumes:
            self.add_visual_sphere(
                vol['position'],
                vol['radius'],
                [0.0, 0.0, 1.0, 0.3],  # Blue, transparent
                "collision"
            )
            
        # Update mocap positions
        self.render_spheres_as_mocap()


def create_visual_model_with_spheres(base_model_path, num_visual_spheres=20):
    """
    Create a modified version of the model that includes visual sphere geoms.
    
    This function modifies the XML to add visual-only sphere geoms that can be
    repositioned at runtime for obstacles, goals, and collision volumes.
    """
    import xml.etree.ElementTree as ET
    import os
    
    # Load the base model XML
    tree = ET.parse(base_model_path)
    root = tree.getroot()
    
    # Find worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')
        
    # Add visual sphere geoms
    for i in range(num_visual_spheres):
        sphere_body = ET.SubElement(worldbody, 'body', {
            'name': f'visual_sphere_{i}',
            'pos': '0 0 -10'  # Start below ground (hidden)
        })
        
        # Add visual geom (no collision)
        ET.SubElement(sphere_body, 'geom', {
            'name': f'visual_sphere_geom_{i}',
            'type': 'sphere',
            'size': '0.05',
            'rgba': '1 0 0 0.8',  # Red by default
            'contype': '0',  # No collision
            'conaffinity': '0',  # No collision
            'group': '1'  # Visual group
        })
        
        # Add freejoint for positioning
        ET.SubElement(sphere_body, 'freejoint', {
            'name': f'visual_sphere_joint_{i}'
        })
    
    # Save modified model
    modified_path = base_model_path.replace('.xml', '_with_visuals.xml')
    tree.write(modified_path)
    
    return modified_path

