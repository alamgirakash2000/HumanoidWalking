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

    # Rename freejoint to 'root'
    pelvis = root.find(".//body[@name='pelvis']")
    if pelvis is not None:
        fj = pelvis.find('freejoint')
        if fj is not None:
            fj.set('name', 'root')

    # Add ankle sites
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

    # Replace actuators with torque motors per LEG_JOINTS
    root = _add_leg_motors_et(root)

    # Remove unused joints so nq matches base(7)+legs(12)=19
    root = _remove_unused_joints(root)

    # Ensure a simple floor plane exists in worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        worldbody = ET.SubElement(root, 'worldbody')
    floor = worldbody.find("geom[@name='floor']")
    if floor is None:
        floor = ET.SubElement(worldbody, 'geom', {'name': 'floor'})
    # Make it an infinite plane with the standard checkerboard material
    floor.set('type', 'plane')
    floor.set('size', '0 0 0.05')
    floor.set('material', 'groundplane')

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


