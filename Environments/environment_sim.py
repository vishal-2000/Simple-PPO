import time
import pybullet as p
import pybullet_data
import numpy as np
import Environments.cameras as cameras
from Config.constants import PIXEL_SIZE, WORKSPACE_LIMITS
import os

import pybullet_utils.bullet_client as bc

class Environment:
    def __init__(self, gui=True, time_step=1 / 480):
        """Creates environment with PyBullet.

        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """

        self.time_step = time_step
        self.gui = gui
        self.pixel_size = PIXEL_SIZE
        self.obj_ids = {"fixed": [], "rigid": []}
        self.agent_cams = cameras.RealSenseD435.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.bounds = WORKSPACE_LIMITS
        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.drop_joints0 = np.array([0.5, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.drop_joints1 = np.array([1, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi

        # Start PyBullet.
        self.client_id = bc.BulletClient(p.GUI if gui else p.DIRECT) # p.connect(p.GUI if gui else p.DIRECT) # To spawn multiple instances
        self.client_id.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client_id.setTimeStep(time_step)
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        if gui:
            target = self.client_id.getDebugVisualizerCamera()[11]
            self.client_id.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target,
            )

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(self.client_id.getBaseVelocity(i)[0]) for i in self.obj_ids["rigid"]]
        return all(np.array(v) < 5e-3)

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = self.client_id.getBasePositionAndOrientation(obj_id)
                dim = self.client_id.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    def add_object_push_from_file(self, file_name):
        body_ids = []
        success = True
        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            num_obj = len(file_content)
            obj_files = []
            obj_mesh_colors = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx].split()
                obj_file = os.path.join("Assets", "blocks", file_content_curr_object[0])
                obj_files.append(obj_file)
                obj_positions.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )
                obj_orientations.append(
                    [
                        float(file_content_curr_object[7]),
                        float(file_content_curr_object[8]),
                        float(file_content_curr_object[9]),
                    ]
                )
                obj_mesh_colors.append(
                    [
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ]
                )

        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            object_color = [
                obj_mesh_colors[object_idx][0],
                obj_mesh_colors[object_idx][1],
                obj_mesh_colors[object_idx][2],
                1,
            ]
            body_id = self.client_id.loadURDF(
                curr_mesh_file, object_position, p.getQuaternionFromEuler(object_orientation)
            )
            if object_color[0]!=-1: # Vishal change
                self.client_id.changeVisualShape(body_id, -1, rgbaColor=object_color)
            body_ids.append(body_id)
            self.add_object_id(body_id)
            success &= self.wait_static()
            success &= self.wait_static()

        print("Success ---------------------------------------- : {}".format(success))
        print(body_ids)
        # give time to stop
        for _ in range(5):
            self.client_id.stepSimulation()

        return body_ids, success

    def add_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].append(obj_id)

    def remove_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].remove(obj_id)

    def wait_static(self, timeout=5):
        """Step simulator asynchronously until objects settle."""
        self.client_id.stepSimulation()
        self.client_id.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            self.client_id.stepSimulation()
            self.client_id.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def reset(self, use_gripper=True):
        self.obj_ids = {"fixed": [], "rigid": []}
        self.use_gripper = use_gripper
        self.client_id.resetSimulation()
        self.client_id.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # Load workspace
        self.plane = self.client_id.loadURDF("plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True)
        self.workspace = self.client_id.loadURDF(
            "Assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True
        )
        self.client_id.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        self.client_id.changeDynamics(
            self.workspace,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        # Load UR5e
        self.ur5e = self.client_id.loadURDF("Assets/ur5e/ur5e.urdf", basePosition=(0, 0, 0), useFixedBase=True)
        self.ur5e_joints = []
        for i in range(self.client_id.getNumJoints(self.ur5e)):
            info = self.client_id.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id
            if joint_type == self.client_id.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)
        self.client_id.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)

        if use_gripper:
            self.setup_gripper()
        else:
            self.setup_spatula()

        # Move robot to home joint configuration.
        success = self.go_home()
        if self.use_gripper:
            self.close_gripper()
            self.open_gripper()

        if not success:
            print("Simulation is wrong!")
            exit()

        # Re-enable rendering.
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def setup_gripper(self):
        """Load end-effector: gripper"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.ee = self.client_id.loadURDF(
            "Assets/ur5e/gripper/robotiq_2f_85.urdf",
            ee_position,
            self.client_id.getQuaternionFromEuler((-np.pi / 2, 0, 0)),
        )
        self.ee_tip_offset = 0.15
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.7
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(self.client_id.getNumJoints(self.ee)):
            info = self.client_id.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif (
                joint_name == "left_inner_finger_pad_joint"
                or joint_name == "right_inner_finger_pad_joint"
            ):
                self.client_id.changeDynamics(self.ee, joint_id, lateralFriction=1)
            elif joint_type == p.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Keep the joints static
                self.client_id.setJointMotorControl2(
                    self.ee, joint_id, self.client_id.VELOCITY_CONTROL, targetVelocity=0, force=0
                )
        self.ee_constraint = self.client_id.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=p.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self.client_id.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=1, erp=0.5, maxForce=800)
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=-1, erp=0.5, maxForce=800)
        # Set up mimic joints in robotiq gripper: right
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=1, erp=0.5, maxForce=800)
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=-1, erp=0.5, maxForce=800)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = self.client_id.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        self.client_id.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=100)

    def setup_spatula(self):
        """Load end-effector: spatula"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.ee = self.client_id.loadURDF(
            "Assets/ur5e/spatula/spatula-base.urdf",
            ee_position,
            p.getQuaternionFromEuler((-np.pi / 2, 0, 0)),
        )
        self.ee_tip_offset = (
            0.12  # tip_distance: the add-on distance to the tip from ur5e ee joint.
        )
        self.ee_tip_id = 0  # id of tip_link
        self.ee_constraint = self.client_id.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0.0, 0.01),
            childFrameOrientation=p.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )

    def replace_to_gripper(self):
        self.client_id.removeConstraint(self.ee_constraint)
        self.client_id.removeBody(self.ee)
        self.setup_gripper()
        self.use_gripper = True
        success = self.go_home()

        if not success:
            print("Simulation is wrong!")
            exit()

        self.close_gripper()
        self.open_gripper()

    def replace_to_spatula(self):
        self.client_id.removeConstraint(self.ee_constraint)
        self.client_id.removeBody(self.ee)
        self.setup_spatula()
        self.use_gripper = False
        success = self.go_home()

        if not success:
            print("Simulation is wrong!")
            exit()

    def step(self, pose0=None, pose1=None):
        """Execute action with specified primitive.

        Args:
            action: action to execute.

        Returns:
            obs, done
        """
        if pose0 is not None and pose1 is not None:
            success = self.push(pose0, pose1)
            # Exit early if action times out.
            if not success:
                return {}, False

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            self.client_id.stepSimulation()

        # Get RGB-D camera image observations.
        obs = {"color": [], "depth": []}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            obs["color"].append(color)
            obs["depth"].append(depth)

        return obs, True

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = p.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = p.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = self.client_id.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=1,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, config["image_size"]))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def __del__(self):
        self.client_id.disconnect()

    def get_link_pose(self, body, link):
        result = self.client_id.getLinkState(body, link)
        return result[4], result[5]

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self):
        return self.move_joints(self.home_joints)

    def move_joints(self, target_joints, speed=0.01, timeout=3):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array([self.client_id.getJointState(self.ur5e, i)[0] for i in self.ur5e_joints])
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 1e-2):
                # give time to stop
                for _ in range(10):
                    self.client_id.stepSimulation()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            self.client_id.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            self.client_id.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = self.client_id.calculateInverseKinematics(
            bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ur5e_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
            restPoses=np.float32(self.ik_rest_joints).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def straight_move(self, pose0, pose1, rot, speed=0.003, max_force=300, detect_force=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(np.linalg.norm(pose1 - pose0) / step_distance))  # every 1 cm
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if detect_force:
                force = np.sum(np.abs(np.array(self.client_id.getJointState(self.ur5e, self.ur5e_ee_id)[2])))
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False
        success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def push(self, pose0, pose1, speed=0.0002):
        """Execute pushing primitive.

        Args:
            pose0: SE(3) starting pose.
            pose1: SE(3) ending pose.
            speed: the speed of the planar push.

        Returns:
            success: robot movement success if True.
        """

        # Adjust push start and end positions.
        pos0 = np.array(pose0, dtype=np.float32)
        pos1 = np.array(pose1, dtype=np.float32)
        pos0[2] += self.ee_tip_offset
        pos1[2] += self.ee_tip_offset
        vec = pos1 - pos0
        length = np.linalg.norm(vec)
        if length <= 0.005:
            length = 0.005
        vec = vec / length

        # print("I am here")

        # Align against push direction.
        theta = np.arctan2(vec[1], vec[0])
        rot = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, theta])

        over0 = (pos0[0], pos0[1], pos0[2] + 0.2)
        over1 = (pos1[0], pos1[1], pos1[2] + 0.2)

        # Execute push.
        success = self.move_joints(self.ik_rest_joints)

        # print("I am here too!")
        if success:
            success = self.move_ee_pose((over0, rot))
            # print("I am here too yo man!")
            # print(success)
        if success:
            success = self.straight_move(over0, pos0, rot, detect_force=True)
            # print("I am here too yo yo man!")
            # print(success)
        if success:
            success = self.straight_move(pos0, pos1, rot, speed, detect_force=True)
            # print("I am here too yo yo yo man!")
            # print(success)
        if success:
            success = self.straight_move(pos1, over1, rot)
            # print("I am here too yo yo yo yo man!")
            # print(success)
        success &= self.go_home()

        # print(f"Push from {pose0} to {pose1}, {success}")

        return success

    def grasp(self, pose, angle):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.
            angle: rotation angle

        Returns:
            success: robot movement success if True.
        """

        # Adjust grasp positions.
        pos = np.array(pose, dtype=np.float32)
        pos[2] = max(pos[2] - 0.04, self.bounds[2][0])
        pos[2] += self.ee_tip_offset

        # Align against grasp direction.
        angle = ((angle) % np.pi) - np.pi / 2
        rot = p.getQuaternionFromEuler([np.pi / 2, np.pi / 2, -angle])

        over = (pos[0], pos[1], pos[2] + 0.2)

        # Execute push.
        self.open_gripper()
        success = self.move_joints(self.ik_rest_joints)
        grasp_sucess = False
        if success:
            success = self.move_ee_pose((over, rot))
        if success:
            success = self.straight_move(over, pos, rot, detect_force=True)
        if success:
            self.close_gripper()
            success = self.straight_move(pos, over, rot)
            grasp_sucess = self.is_gripper_closed
        if success and grasp_sucess:
            success &= self.move_joints(self.drop_joints0, speed=0.005)
            success &= self.move_joints(self.drop_joints1, speed=0.005)
            grasp_sucess = self.is_gripper_closed
            self.open_gripper()
            grasp_sucess &= success
            success &= self.move_joints(self.drop_joints0)
        else:
            grasp_sucess &= success
        self.open_gripper()
        success &= self.go_home()

        print(f"Grasp at {pose}, {success}, the grasp {grasp_sucess}")

        return success, grasp_sucess

    def open_gripper(self):
        self._move_gripper(self.gripper_angle_open, speed=0.01)

    def close_gripper(self):
        self._move_gripper(self.gripper_angle_close, speed=0.01, is_slow=True)

    @property
    def is_gripper_closed(self):
        gripper_angle = self.client_id.getJointState(self.ee, self.gripper_main_joint)[0]
        return gripper_angle < self.gripper_angle_close_threshold

    def _move_gripper(self, target_angle, speed=0.01, timeout=3, max_force=5, is_slow=False):
        t0 = time.time()
        count = 0
        max_count = 3
        current_angle = self.client_id.getJointState(self.ee, self.gripper_main_joint)[0]

        # while (time.time() - t0) < timeout:
        #     # Move with constant velocity
        #     diff_angle = target_angle - current_angle
        #     norm = np.linalg.norm(diff_angle)
        #     v = diff_angle / norm if norm > 0 else 0
        #     step_angle = current_angle + v * speed
        #     p.setJointMotorControl2(
        #         self.ee,
        #         self.gripper_main_joint,
        #         p.POSITION_CONTROL,
        #         targetPosition=step_angle,
        #         force=5,
        #     )
        #     p.setJointMotorControl2(
        #         self.ee,
        #         self.gripper_mimic_joints["right_outer_knuckle_joint"],
        #         p.POSITION_CONTROL,
        #         targetPosition=step_angle,
        #         force=5,
        #     )
        #     p.stepSimulation()

        #     info = p.getJointState(self.ee, self.gripper_main_joint)
        #     current_angle = info[0]
        #     force = abs(info[2][1])
        #     # complete if the gripper reached its target angle
        #     if abs(target_angle - current_angle) < 1e-2:
        #         break
        #     # count if the gripper touched objects
        #     count = count + 1 if force > max_force else 0
        #     # complete if the count is enough
        #     if count > max_count:
        #         break
        # maintain the angles
        if is_slow:
            self.client_id.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                p.POSITION_CONTROL,
                targetPosition=target_angle,
                maxVelocity=0.5,
                force=1,
            )
            self.client_id.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                p.POSITION_CONTROL,
                targetPosition=target_angle,
                maxVelocity=0.5,
                force=1,
            )
            for _ in range(500):
                self.client_id.stepSimulation()
        self.client_id.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            p.POSITION_CONTROL,
            targetPosition=target_angle,
            maxVelocity=10,
            force=5,
        )
        self.client_id.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            p.POSITION_CONTROL,
            targetPosition=target_angle,
            maxVelocity=10,
            force=5,
        )
        for _ in range(10):
            self.client_id.stepSimulation()
        # p.setJointMotorControl2(
        #     self.ee,
        #     self.gripper_main_joint,
        #     p.POSITION_CONTROL,
        #     targetPosition=target_angle,
        #     force=5,
        # )
        # p.setJointMotorControl2(
        #     self.ee,
        #     self.gripper_mimic_joints["right_outer_knuckle_joint"],
        #     p.POSITION_CONTROL,
        #     targetPosition=target_angle,
        #     force=5,
        # )
        # for _ in range(10):
        #     p.stepSimulation()
        # while (time.time() - t0) < timeout:
        #     current_angle = p.getJointState(self.ee, self.gripper_main_joint)[0]
        #     diff_angle = abs(current_angle - prev_angle)
        #     prev_angle = current_angle
        #     if diff_angle < 1e-3:
        #         break
        #     for _ in range(10):
        #         p.stepSimulation()
        # for _ in range(10):
        #     p.stepSimulation()