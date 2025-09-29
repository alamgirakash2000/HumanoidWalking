"""
Unified robot configuration for G1 robot with both walking and upper body control.
Combines the walking leg control with upper body obstacle avoidance.
"""
import os
import xml.etree.ElementTree as ET
import shutil
import models

# Import required joint definitions
from envs.g1.gen_xml import LEG_JOINTS, WAIST_JOINTS, ARM_JOINTS

# Import upper body components from simplified system
from simplified.g1_benchmark_pid.robot_impl import G1BasicConfig


class G1CombinedConfig:
    """
    Combined configuration for G1 robot supporting both walking and upper body control.
    
    This configuration:
    - Uses all leg joints for walking (controlled by trained policy)
    - Uses upper body joints for obstacle avoidance (controlled by PID+safety)
    - Maintains proper joint indexing for both subsystems
    """
    
    def __init__(self):
        # Joint definitions - all joints that will be actuated
        self.leg_joints = LEG_JOINTS  # 12 joints for walking policy
        self.waist_joints = WAIST_JOINTS  # 3 joints for upper body control
        self.arm_joints = ARM_JOINTS  # 14 joints for upper body control
        
        self.upper_body_joints = self.waist_joints + self.arm_joints  # 17 total
        self.all_joints = self.leg_joints + self.upper_body_joints  # 29 total
        
        # For compatibility with walking system
        self.walking_joint_names = self.leg_joints
        self.walking_joint_indices = list(range(len(self.leg_joints)))
        
        # For compatibility with upper body system (reuse existing config structure)
        self.upper_body_config = G1BasicConfig()
        
        # MuJoCo model paths
        self.mjcf_path = 'g1_combined/g1_combined.xml'
        self.export_dir = '/tmp/mjcf-export/g1_combined'
        
    def get_leg_action_indices(self):
        """Get indices for leg joints in the combined action space."""
        return list(range(len(self.leg_joints)))
    
    def get_upper_body_action_indices(self):
        """Get indices for upper body joints in the combined action space."""
        start_idx = len(self.leg_joints)
        return list(range(start_idx, start_idx + len(self.upper_body_joints)))
    
    def get_walking_joint_mapping(self):
        """Map walking policy outputs to combined action space."""
        return {i: i for i in range(len(self.leg_joints))}
    
    def get_upper_body_joint_mapping(self):
        """Map upper body policy outputs to combined action space."""
        start_idx = len(self.leg_joints)
        return {i: start_idx + i for i in range(len(self.upper_body_joints))}


def create_combined_g1_model(export_path):
    """
    Create a unified G1 MuJoCo model that supports both walking and upper body control.
    
    This function:
    1. Takes the full G1 model
    2. Adds actuators for all required joints (legs + upper body)
    3. Sets up proper collision volumes and constraints
    4. Ensures compatibility with both policy systems
    """
    
    print("Creating combined G1 model for walking + obstacle avoidance...")
    
    # Ensure export directory exists
    os.makedirs(export_path, exist_ok=True)
    
    # Copy G1 source assets
    G1_SOURCE_DIR = os.path.join(os.path.dirname(models.__file__), "mujoco_menagerie/unitree_g1")
    shutil.copytree(G1_SOURCE_DIR, export_path, dirs_exist_ok=True)
    
    # Load and modify the G1 XML
    g1_xml_path = os.path.join(export_path, 'g1.xml')
    if not os.path.exists(g1_xml_path):
        raise FileNotFoundError(f"Expected g1.xml at {g1_xml_path}")

    tree = ET.parse(g1_xml_path)
    root = tree.getroot()
    
    # Fix mesh directory path to be absolute
    compiler = root.find('compiler')
    if compiler is not None:
        # Set absolute path to meshes
        meshdir = os.path.join(export_path, 'assets')
        compiler.set('meshdir', meshdir)
        print(f"Set mesh directory to: {meshdir}")
    
    # Rename freejoint to 'root' for compatibility
    pelvis = root.find(".//body[@name='pelvis']")
    if pelvis is not None:
        fj = pelvis.find('freejoint')
        if fj is not None:
            fj.set('name', 'root')
    
    # Add foot sites for walking system compatibility
    for body_name, site_name in [
        ('left_ankle_roll_link', 'lf_force'),
        ('right_ankle_roll_link', 'rf_force')
    ]:
        b = root.find(f".//body[@name='{body_name}']")
        if b is not None and b.find(f"site[@name='{site_name}']") is None:
            site = ET.SubElement(b, 'site')
            site.set('name', site_name)
            site.set('pos', '0.03 0 -0.03')
            site.set('size', '0.001')
            site.set('rgba', '0.5 0.5 0.5 0.3')
            site.set('group', '4')
    
    # Remove existing actuators and add our combined set
    actuator = root.find('actuator')
    if actuator is None:
        actuator = ET.SubElement(root, 'actuator')
    else:
        # Clear existing actuators
        for child in list(actuator):
            actuator.remove(child)
    
    # Add actuators for all controlled joints (legs + upper body)
    config = G1CombinedConfig()
    all_controlled_joints = config.all_joints
    
    for jn in all_controlled_joints:
        ET.SubElement(actuator, 'motor', {
            'name': f"{jn}_motor",
            'joint': jn,
            'ctrllimited': 'true',
            'ctrlrange': '-300 300',
            'gear': '1'
        })
    
    # Ensure proper floor
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')
    
    floor = worldbody.find("geom[@name='floor']")
    if floor is None:
        floor = ET.SubElement(worldbody, 'geom', {'name': 'floor'})
    
    floor.set('type', 'plane')
    floor.set('size', '0 0 0.05')
    floor.set('material', 'groundplane')
    
    # Add assets for rendering
    _ensure_rendering_assets(root)
    
    # Remove keyframes to avoid control size mismatches
    for kf in root.findall('keyframe'):
        root.remove(kf)
    
    # Write the combined model
    combined_xml_path = os.path.join(export_path, 'g1_combined.xml')
    tree.write(combined_xml_path)
    
    print(f"✅ Combined G1 model created at: {combined_xml_path}")
    print(f"   - Total actuators: {len(all_controlled_joints)}")
    print(f"   - Leg joints: {len(config.leg_joints)} (walking)")
    print(f"   - Upper body joints: {len(config.upper_body_joints)} (avoidance)")
    
    return combined_xml_path


def _ensure_rendering_assets(root):
    """Add necessary rendering assets to the XML."""
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    
    # Sky texture
    if asset.find("texture[@type='skybox']") is None:
        ET.SubElement(asset, 'texture', {
            'type': 'skybox', 'builtin': 'gradient',
            'rgb1': '0.3 0.5 0.7', 'rgb2': '0 0 0', 
            'width': '512', 'height': '3072'
        })
    
    # Ground texture
    if asset.find("texture[@name='groundplane']") is None:
        ET.SubElement(asset, 'texture', {
            'type': '2d', 'name': 'groundplane', 'builtin': 'checker', 
            'mark': 'edge', 'rgb1': '0.2 0.3 0.4', 'rgb2': '0.1 0.2 0.3',
            'markrgb': '0.8 0.8 0.8', 'width': '300', 'height': '300'
        })
    
    # Ground material  
    if asset.find("material[@name='groundplane']") is None:
        ET.SubElement(asset, 'material', {
            'name': 'groundplane', 'texture': 'groundplane', 
            'texuniform': 'true', 'texrepeat': '5 5', 'reflectance': '0.2'
        })


if __name__ == "__main__":
    # Test model creation
    config = G1CombinedConfig()
    model_path = create_combined_g1_model(config.export_dir)
    print(f"Model created successfully: {model_path}")
