import os
import shutil
import xml.etree.ElementTree as ET
import models

G1_SOURCE_DIR = os.path.join(os.path.dirname(models.__file__), "mujoco_menagerie/unitree_g1")

# Order: 6 joints per leg: yaw, roll, pitch, knee, ankle_pitch, ankle_roll
LEG_JOINTS = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _add_leg_motors_et(root):
    actuator = root.find('actuator')
    if actuator is None:
        actuator = ET.SubElement(root, 'actuator')
    else:
        # remove all existing actuators
        for child in list(actuator):
            actuator.remove(child)
    for jn in LEG_JOINTS:
        ET.SubElement(actuator, 'motor', {
            'name': f"{jn}_motor",
            'joint': jn,
            'ctrllimited': 'true',
            'ctrlrange': '-300 300',
            'gear': '1'
        })
    return root

def _remove_unused_joints(root):
    # Remove waist and arm joints to keep only freejoint + legs
    joint_names_to_remove = set(WAIST_JOINTS + ARM_JOINTS)
    for jn in joint_names_to_remove:
        for joint in root.findall(f".//joint[@name='{jn}']"):
            parent = joint.getparent() if hasattr(joint, 'getparent') else None
            # xml.etree.ElementTree doesn't have getparent; find by iteration
            if parent is None:
                for elem in root.iter():
                    for child in list(elem):
                        if child is joint:
                            elem.remove(child)
                            break
            else:
                parent.remove(joint)
    return root

def builder(export_path, config):
    print("Preparing G1 model assets...")
    _ensure_dir(export_path)
    # Copy source directory assets to export path
    shutil.copytree(G1_SOURCE_DIR, export_path, dirs_exist_ok=True)

    # Modify g1.xml in-place under export_path
    g1_xml_path = os.path.join(export_path, 'g1.xml')
    if not os.path.exists(g1_xml_path):
        raise FileNotFoundError(f"Expected g1.xml at {g1_xml_path}")

    tree = ET.parse(g1_xml_path)
    root = tree.getroot()

    # Add/update solver settings for better contact handling
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('integrator', 'implicitfast')
    option.set('timestep', '0.001')
    option.set('iterations', '100')  # Increased from 50 for better convergence
    option.set('solver', 'Newton')
    option.set('tolerance', '1e-10')
    option.set('cone', 'pyramidal')  # Better friction model
    option.set('jacobian', 'auto')  # Let MuJoCo choose best Jacobian
    
    # Add contact solver parameters to default
    default_elem = root.find('default')
    if default_elem is None:
        default_elem = ET.SubElement(root, 'default')
    
    # Find or create g1 default class
    g1_default = default_elem.find(".//default[@class='g1']")
    if g1_default is None:
        g1_default = ET.SubElement(default_elem, 'default', {'class': 'g1'})
    
    # Add geom defaults with contact parameters
    geom_default = g1_default.find('geom')
    if geom_default is None:
        geom_default = ET.SubElement(g1_default, 'geom')
    geom_default.set('solref', '0.01 1')  # Soft contacts with good damping
    geom_default.set('solimp', '0.99 0.99 0.001')  # High impedance, low penetration
    geom_default.set('friction', '1.0 0.005 0.0001')  # Good friction for stairs
    
    # Update foot collision sphere parameters specifically
    collision_default = g1_default.find(".//default[@class='collision']")
    if collision_default is not None:
        foot_default = collision_default.find(".//default[@class='foot']")
        if foot_default is not None:
            foot_geom = foot_default.find('geom')
            if foot_geom is None:
                foot_geom = ET.SubElement(foot_default, 'geom')
            foot_geom.set('type', 'sphere')
            foot_geom.set('size', '0.012')  # Slightly larger: 12mm instead of 5mm
            foot_geom.set('solref', '0.001 1')  # Very stiff for feet
            foot_geom.set('solimp', '0.995 0.999 0.0001')  # Extremely high impedance
            foot_geom.set('friction', '1.5 0.005 0.0001')  # Very high friction
            foot_geom.set('condim', '3')
            foot_geom.set('priority', '1')
    
    # Rename freejoint to 'root'
    pelvis = root.find(".//body[@name='pelvis']")
    if pelvis is not None:
        fj = pelvis.find('freejoint')
        if fj is not None:
            fj.set('name', 'root')

    # Add ankle sites and additional foot collision geometry
    for body_name, site_name in [
        ('left_ankle_roll_link', 'lf_force'),
        ('right_ankle_roll_link', 'rf_force')
    ]:
        b = root.find(f".//body[@name='{body_name}']")
        if b is not None:
            # Add force site if not present
            if b.find(f"site[@name='{site_name}']") is None:
                site = ET.SubElement(b, 'site')
                site.set('name', site_name)
                site.set('pos', '0.03 0 -0.03')
                site.set('size', '0.001')
                site.set('rgba', '0.5 0.5 0.5 0.3')
                site.set('group', '4')
            
            # Add more collision spheres to prevent toe penetration
            # Original has 4 spheres - add 6 more for better coverage
            additional_contact_points = [
                # Mid-foot area
                ('0.0 0.02 -0.03', '0.015'),   # mid left
                ('0.0 -0.02 -0.03', '0.015'),  # mid right
                ('0.035 0.02 -0.03', '0.015'), # mid-front left
                ('0.035 -0.02 -0.03', '0.015'),# mid-front right
                # Extra toe coverage (most important!)
                ('0.08 0 -0.03', '0.015'),     # center toe
                ('0.105 0 -0.03', '0.015'),    # forward toe
            ]
            
            for pos, size in additional_contact_points:
                # Check if geom already exists at this position
                existing = [g for g in b.findall('geom') 
                           if g.get('pos') == pos]
                if not existing:
                    geom = ET.SubElement(b, 'geom')
                    geom.set('class', 'foot')
                    geom.set('pos', pos)
                    geom.set('size', size)

    # Replace actuators with torque motors per LEG_JOINTS
    root = _add_leg_motors_et(root)

    # Remove unused joints so nq matches base(7)+legs(12)=19
    root = _remove_unused_joints(root)

    # Wrap floor geom in a body (needed for stepping task)
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')
    
    # Remove any existing floor geom
    for floor_geom in worldbody.findall("geom[@name='floor']"):
        worldbody.remove(floor_geom)
    
    # Create floor body with geom inside it
    floor_body = ET.SubElement(worldbody, 'body', {'name': 'floor'})
    floor_geom = ET.SubElement(floor_body, 'geom', {
        'name': 'floor',
        'type': 'plane',
        'size': '0 0 0.25',
        'material': 'groundplane',
        'solref': '0.002 1',  # Very stiff contact (was 0.01 1)
        'solimp': '0.99 0.995 0.001',  # High impedance, minimal penetration
        'friction': '1.2 0.005 0.0001',  # High friction for grip
        'condim': '3'  # 3D friction cone
    })

    # Remove keyframes or adjust to match actuator count (avoid ctrl size mismatch)
    keyframes = root.findall('keyframe')
    if keyframes:
        for kf in keyframes:
            root.remove(kf)
    else:
        # Fallback: ensure ctrl size matches number of actuators (12)
        kf = root.find('keyframe')
        if kf is not None:
            key = kf.find('key')
            if key is not None:
                key.set('ctrl', ' '.join(['0']*len(LEG_JOINTS)))

    # Optionally add stepping boxes (stairs/blocks) for stepping task
    if 'boxes' in config and config['boxes']:
        for idx in range(20):
            x = 0.4 * idx
            body = ET.SubElement(worldbody, 'body', {
                'name': f"box{str(idx+1).zfill(2)}",
                'pos': f"{x:.3f} 0 -0.2"
            })
            ET.SubElement(body, 'geom', {
                'name': f"box{str(idx+1).zfill(2)}",
                'type': 'box',
                'size': '0.15 0.1 0.1',
                'rgba': '0.8 0.8 0.8 1',
                'group': '0',
                'solref': '0.002 1',  # Very stiff contact
                'solimp': '0.99 0.995 0.001',  # High impedance
                'friction': '1.2 0.005 0.0001',  # High friction
                'condim': '3'  # 3D friction cone
            })

    # Ensure groundplane texture/material assets exist
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    has_sky = asset.find("texture[@type='skybox']") is not None
    if not has_sky:
        ET.SubElement(asset, 'texture', {
            'type': 'skybox', 'builtin': 'gradient',
            'rgb1': '0.3 0.5 0.7', 'rgb2': '0 0 0', 'width': '512', 'height': '3072'
        })
    has_ground_tex = asset.find("texture[@name='groundplane']") is not None
    if not has_ground_tex:
        ET.SubElement(asset, 'texture', {
            'type': '2d', 'name': 'groundplane', 'builtin': 'checker', 'mark': 'edge',
            'rgb1': '0.2 0.3 0.4', 'rgb2': '0.1 0.2 0.3', 'markrgb': '0.8 0.8 0.8',
            'width': '300', 'height': '300'
        })
    has_ground_mat = asset.find("material[@name='groundplane']") is not None
    if not has_ground_mat:
        ET.SubElement(asset, 'material', {
            'name': 'groundplane', 'texture': 'groundplane', 'texuniform': 'true',
            'texrepeat': '5 5', 'reflectance': '0.2'
        })

    # Write back
    tree.write(g1_xml_path)
    print("Exported modified G1 XML to", g1_xml_path)
    return


