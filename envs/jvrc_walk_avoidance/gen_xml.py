import sys
import os
import models
from dm_control import mjcf

# Define joints directly (copied from jvrc gen_xml)
JVRC_DESCRIPTION_PATH = os.path.join(os.path.dirname(models.__file__), "jvrc_mj_description/xml/scene.xml")

WAIST_JOINTS = ['WAIST_Y', 'WAIST_P', 'WAIST_R']
HEAD_JOINTS = ['NECK_Y', 'NECK_R', 'NECK_P']
HAND_JOINTS = ['R_UTHUMB', 'R_LTHUMB', 'R_UINDEX', 'R_LINDEX', 'R_ULITTLE', 'R_LLITTLE',
               'L_UTHUMB', 'L_LTHUMB', 'L_UINDEX', 'L_LINDEX', 'L_ULITTLE', 'L_LLITTLE']
ARM_JOINTS = ['R_SHOULDER_P', 'R_SHOULDER_R', 'R_SHOULDER_Y', 'R_ELBOW_P', 'R_ELBOW_Y', 'R_WRIST_R', 'R_WRIST_Y',
              'L_SHOULDER_P', 'L_SHOULDER_R', 'L_SHOULDER_Y', 'L_ELBOW_P', 'L_ELBOW_Y', 'L_WRIST_R', 'L_WRIST_Y']
LEG_JOINTS = ['R_HIP_P', 'R_HIP_R', 'R_HIP_Y', 'R_KNEE', 'R_ANKLE_R', 'R_ANKLE_P',
              'L_HIP_P', 'L_HIP_R', 'L_HIP_Y', 'L_KNEE', 'L_ANKLE_R', 'L_ANKLE_P']

# Upper body joints (for obstacle avoidance)
UPPER_BODY_JOINTS = WAIST_JOINTS + HEAD_JOINTS + ARM_JOINTS

def builder(export_path, config):
    print("Building JVRC walk + upper body obstacle avoidance environment...")
    mjcf_model = mjcf.from_path(JVRC_DESCRIPTION_PATH)

    mjcf_model.model = 'jvrc_walk_avoidance'

    # set njmax and nconmax
    mjcf_model.size.njmax = -1
    mjcf_model.size.nconmax = -1
    mjcf_model.statistic.meansize = 0.1
    mjcf_model.statistic.meanmass = 2

    # modify skybox
    for tx in mjcf_model.asset.texture:
        if tx.type == "skybox":
            tx.rgb1 = '0.8 0.9 1.0'
            tx.rgb2 = '0.5 0.7 1.0'

    # remove all collisions
    mjcf_model.contact.remove()

    # remove actuators except for LEG_JOINTS + UPPER_BODY_JOINTS
    actuated_joints = LEG_JOINTS + UPPER_BODY_JOINTS
    for mot in mjcf_model.actuator.motor:
        if mot.joint.name not in actuated_joints:
            mot.remove()

    # remove unused joints (hands only)
    for joint in HAND_JOINTS:
        mjcf_model.find('joint', joint).remove()

    # remove existing equality
    mjcf_model.equality.remove()

    # set arm joints to neutral standing position
    arm_bodies = {
        "R_SHOULDER_P_S": [0, -0.052, 0], "R_SHOULDER_R_S": [-0.17, 0, 0], "R_ELBOW_P_S": [0, -0.524, 0],
        "L_SHOULDER_P_S": [0, -0.052, 0], "L_SHOULDER_R_S": [0.17, 0, 0], "L_ELBOW_P_S": [0, -0.524, 0],
    }
    for bname, euler in arm_bodies.items():
        mjcf_model.find('body', bname).euler = euler

    # collision geoms (legs + upper body)
    collision_geoms = [
        'R_HIP_R_S', 'R_HIP_Y_S', 'R_KNEE_S',  # Right leg
        'L_HIP_R_S', 'L_HIP_Y_S', 'L_KNEE_S',  # Left leg
        'WAIST_Y_S', 'WAIST_R_S', 'NECK_Y_S', 'NECK_P_S',  # Torso/head
        'R_SHOULDER_P_S', 'R_SHOULDER_Y_S', 'R_ELBOW_P_S', 'R_ELBOW_Y_S', 'R_WRIST_Y_S',  # Right arm
        'L_SHOULDER_P_S', 'L_SHOULDER_Y_S', 'L_ELBOW_P_S', 'L_ELBOW_Y_S', 'L_WRIST_Y_S',  # Left arm
    ]

    # remove unused collision geoms
    for body in mjcf_model.worldbody.find_all('body'):
        for idx, geom in enumerate(body.geom):
            geom.name = body.name + '-geom-' + repr(idx)
            if (geom.dclass.dclass == "collision"):
                if body.name not in collision_geoms:
                    geom.remove()

    # move collision geoms to different group
    mjcf_model.default.default['collision'].geom.group = 3

    # manually create collision geom for feet (essential for walking)
    mjcf_model.worldbody.find('body', 'R_ANKLE_P_S').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='0.029 0 -0.09778', type='box')
    mjcf_model.worldbody.find('body', 'L_ANKLE_P_S').add('geom', dclass='collision', size='0.1 0.05 0.01', pos='0.029 0 -0.09778', type='box')

    # ignore collision between adjacent body parts
    mjcf_model.contact.add('exclude', body1='R_KNEE_S', body2='R_ANKLE_P_S')
    mjcf_model.contact.add('exclude', body1='L_KNEE_S', body2='L_ANKLE_P_S')

    # Add EXACT G1 benchmark moving obstacles  
    num_obstacles = config.get('num_obstacles', 3)
    
    for idx in range(num_obstacles):
        obstacle_name = f'moving_obstacle_{idx}'
        
        # Create obstacle body
        obstacle_body = mjcf_model.worldbody.add('body', name=obstacle_name)
        
        # Initial position (will be set by G1 TaskObject3D at runtime)
        obstacle_body.pos = [0.1, 0.1, 0.9]  # Within G1 bounds [-0.3,0.5] x [-0.3,0.5] x [0.8,1.0]
        
        # EXACT G1 parameters: radius=0.05, gray color with 70% opacity
        obstacle_body.add('geom',
                         name=f'{obstacle_name}_geom',
                         type='sphere',
                         size='0.05',
                         rgba='0.5 0.5 0.5 0.7',
                         group='0')
        
        # Add site for tracking
        obstacle_body.add('site',
                         name=f'{obstacle_name}_site',
                         size='0.01',
                         pos='0 0 0')

    # remove unused meshes
    meshes = [g.mesh.name for g in mjcf_model.find_all('geom') if g.type=='mesh' or g.type==None]
    for mesh in mjcf_model.find_all('mesh'):
        if mesh.name not in meshes:
            mesh.remove()

    # fix site pos (for force sensors)
    mjcf_model.worldbody.find('site', 'rf_force').pos = '0.03 0.0 -0.1'
    mjcf_model.worldbody.find('site', 'lf_force').pos = '0.03 0.0 -0.1'

    # wrap floor geom in a body
    mjcf_model.find('geom', 'floor').remove()
    mjcf_model.worldbody.add('body', name='floor')
    mjcf_model.find('body', 'floor').add('geom', name='floor', type="plane", size="0 0 0.25", material="groundplane")

    # export model
    mjcf.export_with_assets(mjcf_model, out_dir=export_path, precision=5)
    path_to_xml = os.path.join(export_path, mjcf_model.model + '.xml')
    print("Exporting XML model to ", path_to_xml)
    return path_to_xml

if __name__=='__main__':
    builder(sys.argv[1], config={'num_obstacles': 3})
