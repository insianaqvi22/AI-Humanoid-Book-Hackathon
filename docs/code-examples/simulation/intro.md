---
title: Simulation Examples for AI Humanoid Robotics
sidebar_position: 3
description: Simulation code examples for AI humanoid robotics implementation
---

# Simulation Examples for AI Humanoid Robotics

Simulation is crucial for developing and testing humanoid robots before deployment on physical hardware. This section provides practical examples of simulation environments and techniques used in AI humanoid robotics development.

## Introduction to Robotics Simulation

Robotics simulation allows developers to:
- Test algorithms in a safe environment
- Develop and debug control systems
- Train AI models without physical hardware
- Evaluate human-robot interaction scenarios
- Validate safety systems

## PyBullet Simulation Example

PyBullet is a popular physics engine for robotics simulation. Here's an example of creating a humanoid robot simulation:

```python
import pybullet as p
import pybullet_data
import numpy as np
import time
import math

class HumanoidSimulation:
    def __init__(self, urdf_path=None, gui=True):
        """
        Initialize the humanoid simulation environment

        Args:
            urdf_path: Path to URDF file (if None, uses default)
            gui: Whether to run with GUI visualization
        """
        self.urdf_path = urdf_path
        self.gui = gui

        # Connect to physics engine
        if gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # Set gravity
        p.setGravity(0, 0, -9.81)

        # Load plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

        # Load humanoid robot
        if urdf_path:
            self.robot_id = p.loadURDF(urdf_path, [0, 0, 1])
        else:
            # Use a simple humanoid model or load from default
            self.robot_id = self.load_simple_humanoid()

        # Get joint information
        self.joint_info = self.get_joint_info()

        # Initialize control parameters
        self.control_mode = p.POSITION_CONTROL
        self.max_force = 100  # Maximum joint force in N*m

        # Initialize simulation parameters
        self.time_step = 1./240.  # 240 Hz simulation
        p.setTimeStep(self.time_step)

        print(f"Simulation initialized with robot ID: {self.robot_id}")

    def load_simple_humanoid(self):
        """
        Load a simple humanoid robot model if no URDF is provided
        """
        # For this example, we'll create a simple stick figure humanoid
        # In practice, you would load a proper humanoid URDF model
        # This is a placeholder - in real implementation you'd load a proper model
        print("Loading simple humanoid model...")

        # For demonstration, let's load a simple model
        # In real usage, you'd have a proper humanoid URDF
        try:
            # Try to load a simple humanoid model
            robot_id = p.loadURDF("r2d2.urdf", [0, 0, 1])  # Using R2D2 as placeholder
        except:
            # If no specific humanoid model, create a simple one
            # This would typically be a custom URDF file
            print("Could not load specific humanoid, using R2D2 as placeholder")
            robot_id = p.loadURDF("r2d2.urdf", [0, 0, 1])

        return robot_id

    def get_joint_info(self):
        """
        Get information about all joints in the robot
        """
        num_joints = p.getNumJoints(self.robot_id)
        joint_info = {}

        for i in range(num_joints):
            joint_data = p.getJointInfo(self.robot_id, i)
            joint_name = joint_data[1].decode('utf-8')
            joint_type = joint_data[2]
            joint_lower_limit = joint_data[8]
            joint_upper_limit = joint_data[9]

            joint_info[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower_limit': joint_lower_limit,
                'upper_limit': joint_upper_limit,
                'max_force': self.max_force
            }

            print(f"Joint {i}: {joint_name}, Type: {joint_type}, Limits: [{joint_lower_limit:.3f}, {joint_upper_limit:.3f}]")

        return joint_info

    def set_joint_positions(self, joint_positions):
        """
        Set target positions for joints

        Args:
            joint_positions: Dictionary mapping joint indices to target positions
        """
        for joint_index, target_position in joint_positions.items():
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=self.joint_info[joint_index]['max_force']
            )

    def get_joint_positions(self):
        """
        Get current joint positions

        Returns:
            Dictionary mapping joint indices to current positions
        """
        joint_positions = {}
        for joint_index in self.joint_info:
            joint_state = p.getJointState(self.robot_id, joint_index)
            joint_positions[joint_index] = joint_state[0]  # Position is first element

        return joint_positions

    def get_link_state(self, link_index):
        """
        Get state of a specific link (position, orientation, etc.)

        Args:
            link_index: Index of the link

        Returns:
            Link state information
        """
        return p.getLinkState(self.robot_id, link_index)

    def get_base_position_and_orientation(self):
        """
        Get base position and orientation of the robot
        """
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        return pos, orn

    def apply_external_force(self, link_index, force, position, force_type="world"):
        """
        Apply external force to a link

        Args:
            link_index: Index of the link to apply force to
            force: Force vector [fx, fy, fz]
            position: Position where force is applied [x, y, z]
            force_type: "world" or "link" coordinate system
        """
        if force_type == "world":
            p.applyExternalForce(
                objectUniqueId=self.robot_id,
                linkIndex=link_index,
                forceObj=force,
                posObj=position,
                flags=p.WORLD_FRAME
            )
        else:
            p.applyExternalForce(
                objectUniqueId=self.robot_id,
                linkIndex=link_index,
                forceObj=force,
                posObj=position,
                flags=p.LINK_FRAME
            )

    def reset_robot_position(self, position, orientation=[0, 0, 0, 1]):
        """
        Reset robot to a specific position and orientation

        Args:
            position: [x, y, z] position
            orientation: [x, y, z, w] quaternion orientation
        """
        p.resetBasePositionAndOrientation(self.robot_id, position, orientation)

        # Reset joint positions to zero
        for joint_index in self.joint_info:
            p.resetJointState(self.robot_id, joint_index, targetValue=0.0)

    def step_simulation(self, steps=1):
        """
        Step the simulation forward

        Args:
            steps: Number of simulation steps to take
        """
        for _ in range(steps):
            p.stepSimulation()
            if self.gui:
                time.sleep(self.time_step)

    def get_robot_state(self):
        """
        Get comprehensive robot state including joint positions, velocities, etc.
        """
        state = {
            'base_position': self.get_base_position_and_orientation()[0],
            'base_orientation': self.get_base_position_and_orientation()[1],
            'joint_positions': self.get_joint_positions(),
            'joint_velocities': {},
            'joint_reaction_forces': {}
        }

        # Get joint velocities and forces
        for joint_index in self.joint_info:
            joint_state = p.getJointState(self.robot_id, joint_index)
            state['joint_velocities'][joint_index] = joint_state[1]  # Velocity
            state['joint_reaction_forces'][joint_index] = joint_state[2]  # Reaction forces

        return state

    def enable_joint_motors(self, joint_indices=None, control_mode=None):
        """
        Enable joint motors for control

        Args:
            joint_indices: List of joint indices to enable (None for all)
            control_mode: Control mode (POSITION_CONTROL, VELOCITY_CONTROL, etc.)
        """
        if joint_indices is None:
            joint_indices = list(self.joint_info.keys())

        if control_mode is None:
            control_mode = self.control_mode

        for joint_index in joint_indices:
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=control_mode,
                force=self.joint_info[joint_index]['max_force']
            )

    def disable_joint_motors(self, joint_indices=None):
        """
        Disable joint motors (for free movement)

        Args:
            joint_indices: List of joint indices to disable (None for all)
        """
        if joint_indices is None:
            joint_indices = list(self.joint_info.keys())

        for joint_index in joint_indices:
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0
            )

    def add_obstacle(self, position, half_extents=[0.5, 0.5, 0.5]):
        """
        Add an obstacle to the simulation environment

        Args:
            position: [x, y, z] position of the obstacle
            half_extents: [x, y, z] half sizes of the box obstacle
        """
        obstacle_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )
        obstacle_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )
        obstacle_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=obstacle_collision,
            baseVisualShapeIndex=obstacle_visual,
            basePosition=position
        )
        return obstacle_id

    def detect_collisions(self):
        """
        Detect collisions between robot and environment
        """
        # Get all contact points involving the robot
        contacts = p.getContactPoints(bodyA=self.robot_id)

        collisions = []
        for contact in contacts:
            contact_info = {
                'body_a_id': contact[1],  # Robot ID
                'body_b_id': contact[2],  # Other body ID
                'link_index_a': contact[3],  # Robot link involved
                'link_index_b': contact[4],  # Other link involved
                'contact_position': contact[5],  # World position of contact
                'contact_normal': contact[7],  # Normal vector at contact
                'contact_distance': contact[8],  # Distance (negative if penetrating)
                'normal_force': contact[9]  # Normal force magnitude
            }

            # Only include contacts with significant force (indicating actual collision)
            if contact_info['normal_force'] > 0.1:  # Threshold for actual collision
                collisions.append(contact_info)

        return collisions

    def cleanup(self):
        """
        Clean up the simulation environment
        """
        p.disconnect(self.physics_client)

    def run_demo(self, duration=10.0):
        """
        Run a demonstration of the simulation capabilities

        Args:
            duration: Duration of the demo in seconds
        """
        print(f"Starting simulation demo for {duration} seconds...")

        steps = int(duration / self.time_step)

        # Enable joint motors
        self.enable_joint_motors()

        # Add some obstacles
        self.add_obstacle([2, 0, 0.5])
        self.add_obstacle([-2, 1, 0.5])

        start_time = time.time()

        for step in range(steps):
            # Simple oscillating motion for demonstration
            t = step * self.time_step

            # Create a simple walking-like motion
            target_positions = {
                0: 0.1 * math.sin(t * 2),  # Hip joint oscillation
                1: 0.1 * math.cos(t * 2),  # Knee joint oscillation
                2: 0.05 * math.sin(t * 3)  # Ankle joint oscillation
            }

            # Apply joint positions (only for available joints)
            active_positions = {}
            for joint_idx, pos in target_positions.items():
                if joint_idx in self.joint_info:
                    active_positions[joint_idx] = pos

            self.set_joint_positions(active_positions)

            # Step simulation
            self.step_simulation(1)

            # Detect and report collisions periodically
            if step % 60 == 0:  # Every 60 steps (about 4 times per second at 240Hz)
                collisions = self.detect_collisions()
                if collisions:
                    print(f"Detected {len(collisions)} collisions at step {step}")

        end_time = time.time()
        print(f"Demo completed in {end_time - start_time:.2f} seconds")


# Example usage
def simulation_example():
    print("=== Humanoid Robot Simulation Example ===")

    # Create simulation environment
    sim = HumanoidSimulation(gui=True)  # Use GUI for visualization

    # Run a demonstration
    sim.run_demo(duration=10.0)

    # Get robot state
    state = sim.get_robot_state()
    print(f"\nFinal robot state:")
    print(f"  Base position: {state['base_position']}")
    print(f"  Joint positions: {list(state['joint_positions'].values())[:5]}...")  # Show first 5 joints

    # Apply external force to demonstrate interaction
    print(f"\nApplying external force to robot...")
    sim.apply_external_force(link_index=-1, force=[50, 0, 0], position=[0, 0, 0])

    # Step simulation to see effect
    sim.step_simulation(60)  # 60 steps = 0.25 seconds at 240Hz

    # Clean up
    sim.cleanup()
    print("Simulation cleaned up successfully")


if __name__ == "__main__":
    simulation_example()
```

## Gazebo Simulation Example

Gazebo is another popular simulation environment. Here's an example using ROS/Gazebo:

```python
#!/usr/bin/env python3
"""
Gazebo simulation example for humanoid robotics
This example assumes ROS Noetic/Melodic with Gazebo and necessary packages installed
"""

import rospy
import numpy as np
import tf.transformations as tf_trans
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose, Point
from gazebo_msgs.srv import SetModelState, GetModelState, SpawnModel
from gazebo_msgs.msg import ModelState, ContactsState
from std_msgs.msg import Float64
import math

class GazeboHumanoidSimulation:
    def __init__(self):
        """
        Initialize Gazebo simulation interface
        """
        rospy.init_node('humanoid_simulation', anonymous=True)

        # Robot joint publishers (for position control)
        self.joint_publishers = {}
        self.setup_joint_controllers()

        # Robot state subscriber
        self.joint_state_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        self.current_joint_positions = {}
        self.current_joint_velocities = {}

        # Robot command publishers
        self.cmd_publishers = {}

        # Gazebo services
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)

        # Collision detection
        self.contact_subscriber = rospy.Subscriber('/gazebo/collisions', ContactsState, self.collision_callback)
        self.collision_data = []

        # Simulation parameters
        self.rate = rospy.Rate(100)  # 100 Hz control loop

        print("Gazebo humanoid simulation interface initialized")

    def setup_joint_controllers(self):
        """
        Set up publishers for joint position control
        This assumes the robot has been loaded with appropriate joint controllers
        """
        # Define joint names for a typical humanoid robot
        joint_names = [
            'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll',
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_elbow',
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_elbow'
        ]

        # Set up publishers for each joint
        for joint_name in joint_names:
            # Position controller topic (adjust based on your robot's controller configuration)
            pub = rospy.Publisher(f'/humanoid/{joint_name}_position_controller/command', Float64, queue_size=10)
            self.joint_publishers[joint_name] = pub

    def joint_state_callback(self, msg):
        """
        Callback function to update current joint states
        """
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.current_joint_velocities[name] = msg.velocity[i]

    def collision_callback(self, msg):
        """
        Callback function to handle collision detection
        """
        self.collision_data = msg.states

    def move_to_position(self, joint_positions, duration=2.0):
        """
        Move robot joints to specified positions using trajectory interpolation

        Args:
            joint_positions: Dictionary mapping joint names to target positions
            duration: Time to reach target positions (seconds)
        """
        # Get current positions
        current_positions = self.current_joint_positions.copy()

        # Create trajectory
        steps = int(duration * 100)  # 100Hz control rate
        trajectory = {}

        for joint_name, target_pos in joint_positions.items():
            if joint_name in current_positions:
                current_pos = current_positions[joint_name]

                # Create smooth trajectory for this joint
                joint_trajectory = []
                for i in range(steps + 1):
                    t = i / steps  # Normalized time (0 to 1)

                    # Use cubic interpolation for smooth motion
                    smooth_t = 3 * t**2 - 2 * t**3

                    pos = current_pos + smooth_t * (target_pos - current_pos)
                    joint_trajectory.append(pos)

                trajectory[joint_name] = joint_trajectory

        # Execute trajectory
        for step in range(steps + 1):
            commands = {}
            for joint_name, joint_traj in trajectory.items():
                if step < len(joint_traj):
                    commands[joint_name] = joint_traj[step]

            # Publish commands
            for joint_name, position in commands.items():
                if joint_name in self.joint_publishers:
                    self.joint_publishers[joint_name].publish(Float64(position))

            self.rate.sleep()

    def get_robot_state(self):
        """
        Get current robot state
        """
        state = {
            'joint_positions': self.current_joint_positions.copy(),
            'joint_velocities': self.current_joint_velocities.copy(),
            'collision_data': self.collision_data.copy()
        }

        # Get model state from Gazebo
        try:
            model_state = self.get_model_state('humanoid', 'world')
            state['model_position'] = model_state.pose.position
            state['model_orientation'] = model_state.pose.orientation
            state['model_linear_velocity'] = model_state.twist.linear
            state['model_angular_velocity'] = model_state.twist.angular
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")

        return state

    def apply_force_to_link(self, link_name, force, position, duration=0.1):
        """
        Apply external force to a specific link

        Args:
            link_name: Name of the link to apply force to
            force: Force vector [x, y, z]
            position: Position vector [x, y, z] where force is applied
            duration: Duration of force application (seconds)
        """
        # This would typically involve sending a wrench command to Gazebo
        # Implementation depends on Gazebo version and setup
        print(f"Applying force {force} to link {link_name} at position {position}")

    def spawn_obstacle(self, name, sdf_xml, position):
        """
        Spawn an obstacle in the simulation

        Args:
            name: Name of the obstacle
            sdf_xml: SDF description of the obstacle
            position: Position [x, y, z] where to spawn
        """
        try:
            # Create pose
            pose = Pose()
            pose.position = Point(position[0], position[1], position[2])

            # Spawn the model
            resp = self.spawn_model(name, sdf_xml, "", pose, "world")
            print(f"Spawned obstacle '{name}': {resp.success}")
            return resp.success
        except rospy.ServiceException as e:
            print(f"Spawn service call failed: {e}")
            return False

    def reset_simulation(self):
        """
        Reset the simulation to initial state
        """
        try:
            # Reset model state
            model_state = ModelState()
            model_state.model_name = 'humanoid'
            model_state.pose.position.x = 0.0
            model_state.pose.position.y = 0.0
            model_state.pose.position.z = 0.8  # Standing height
            model_state.pose.orientation.w = 1.0  # No rotation

            resp = self.set_model_state(model_state)
            print(f"Reset model state: {resp.success}")

            # Reset joint positions (send commands to zero position)
            zero_positions = {name: 0.0 for name in self.joint_publishers.keys()}
            self.move_to_position(zero_positions, duration=1.0)

        except rospy.ServiceException as e:
            print(f"Reset service call failed: {e}")

    def run_control_loop(self, duration=10.0):
        """
        Run a basic control loop demonstration

        Args:
            duration: Duration of the control loop (seconds)
        """
        print(f"Starting control loop for {duration} seconds...")

        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            elapsed = (current_time - start_time).to_sec()

            if elapsed > duration:
                break

            # Simple control logic - make robot do a simple movement
            t = elapsed

            # Create oscillating joint movements
            target_positions = {
                'left_hip_pitch': 0.1 * math.sin(t * 2),
                'right_hip_pitch': 0.1 * math.sin(t * 2 + math.pi),  # Opposite phase
                'left_knee': -0.1 * math.sin(t * 2),
                'right_knee': -0.1 * math.sin(t * 2 + math.pi),
                'left_shoulder_pitch': 0.05 * math.sin(t),
                'right_shoulder_pitch': 0.05 * math.sin(t + math.pi)
            }

            # Send commands (in a real implementation, you'd do this more efficiently)
            for joint_name, position in target_positions.items():
                if joint_name in self.joint_publishers:
                    self.joint_publishers[joint_name].publish(Float64(position))

            # Check for collisions
            if len(self.collision_data) > 0:
                print(f"Collision detected! {len(self.collision_data)} contacts")

            # Print robot state periodically
            if int(elapsed) % 2 == 0 and elapsed > 0:
                state = self.get_robot_state()
                print(f"Time: {elapsed:.1f}s, Position: ({state['model_position'].x:.2f}, {state['model_position'].y:.2f})")

            self.rate.sleep()

    def cleanup(self):
        """
        Cleanup simulation resources
        """
        # Stop all joint movements
        zero_positions = {name: 0.0 for name in self.joint_publishers.keys()}
        self.move_to_position(zero_positions, duration=0.5)
        print("Simulation cleanup completed")

def gazebo_simulation_example():
    """
    Example usage of Gazebo simulation
    """
    print("=== Gazebo Humanoid Simulation Example ===")

    try:
        # Create simulation interface
        sim = GazeboHumanoidSimulation()

        # Wait for initial state
        rospy.sleep(1.0)

        # Get initial state
        initial_state = sim.get_robot_state()
        print(f"Initial robot position: ({initial_state['model_position'].x:.2f}, {initial_state['model_position'].y:.2f})")

        # Run control loop
        sim.run_control_loop(duration=10.0)

        # Clean up
        sim.cleanup()

    except rospy.ROSInterruptException:
        print("Simulation interrupted by user")
    except Exception as e:
        print(f"Error in simulation: {e}")

if __name__ == '__main__':
    gazebo_simulation_example()
```

## MuJoCo Simulation Example

MuJoCo is a physics engine known for its speed and accuracy. Here's an example:

```python
try:
    import mujoco
    import mujoco.viewer
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    import time

    class MuJoCoHumanoidSimulation:
        def __init__(self, model_path=None):
            """
            Initialize MuJoCo simulation

            Args:
                model_path: Path to MuJoCo XML model file
            """
            if model_path:
                self.model = mujoco.MjModel.from_xml_path(model_path)
            else:
                # Use a default humanoid model if no path provided
                # This would typically be a specific humanoid XML file
                print("Loading default humanoid model...")
                # For this example, we'll create a simple model programmatically
                # In practice, you'd load a proper humanoid XML model
                self.model = self.create_simple_humanoid_model()

            # Create data instance
            self.data = mujoco.MjData(self.model)

            # Simulation parameters
            self.dt = 0.002  # 500 Hz simulation
            self.model.opt.timestep = self.dt

            print(f"MuJoCo simulation initialized with {self.model.njnt} joints and {self.model.nbody} bodies")

        def create_simple_humanoid_model(self):
            """
            Create a simple humanoid model in XML format
            This is a simplified example - in practice you'd have a proper XML file
            """
            xml_content = """
            <mujoco model="simple_humanoid">
                <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>

                <default>
                    <joint limited="true" damping="1" armature="0.01"/>
                    <geom contype="1" conaffinity="1" condim="1" friction="0.8 0.1 0.1"/>
                </default>

                <option timestep="0.002" gravity="0 0 -9.81"/>

                <worldbody>
                    <!-- Torso -->
                    <body name="torso" pos="0 0 1.0">
                        <geom name="torso_geom" type="capsule" size="0.1 0.3" pos="0 0 0.3"/>
                        <joint name="root" type="free" limited="false"/>

                        <!-- Head -->
                        <body name="head" pos="0 0 0.6">
                            <geom name="head_geom" type="sphere" size="0.1"/>
                        </body>

                        <!-- Left Arm -->
                        <body name="left_upper_arm" pos="0.2 0 0.3">
                            <geom name="left_upper_arm_geom" type="capsule" size="0.05 0.2" quat="0.707 0 0.707 0"/>
                            <joint name="left_shoulder" type="hinge" pos="0 0 0" axis="0 1 0" range="-90 90"/>
                            <body name="left_lower_arm" pos="0 0 -0.4">
                                <geom name="left_lower_arm_geom" type="capsule" size="0.04 0.15" quat="0.707 0 0.707 0"/>
                                <joint name="left_elbow" type="hinge" pos="0 0 0" axis="0 1 0" range="-90 0"/>
                            </body>
                        </body>

                        <!-- Right Arm -->
                        <body name="right_upper_arm" pos="-0.2 0 0.3">
                            <geom name="right_upper_arm_geom" type="capsule" size="0.05 0.2" quat="0.707 0 0.707 0"/>
                            <joint name="right_shoulder" type="hinge" pos="0 0 0" axis="0 1 0" range="-90 90"/>
                            <body name="right_lower_arm" pos="0 0 -0.4">
                                <geom name="right_lower_arm_geom" type="capsule" size="0.04 0.15" quat="0.707 0 0.707 0"/>
                                <joint name="right_elbow" type="hinge" pos="0 0 0" axis="0 1 0" range="-90 0"/>
                            </body>
                        </body>

                        <!-- Left Leg -->
                        <body name="left_thigh" pos="0.1 0 0">
                            <geom name="left_thigh_geom" type="capsule" size="0.06 0.25" quat="0.707 0 0.707 0"/>
                            <joint name="left_hip" type="hinge" pos="0 0 0" axis="0 1 0" range="-45 45"/>
                            <body name="left_shin" pos="0 0 -0.5">
                                <geom name="left_shin_geom" type="capsule" size="0.05 0.25" quat="0.707 0 0.707 0"/>
                                <joint name="left_knee" type="hinge" pos="0 0 0" axis="0 1 0" range="0 120"/>
                                <body name="left_foot" pos="0 0 -0.5">
                                    <geom name="left_foot_geom" type="box" size="0.1 0.05 0.02" pos="0 0 0.01"/>
                                    <joint name="left_ankle" type="hinge" pos="0 0 0" axis="0 1 0" range="-20 20"/>
                                </body>
                            </body>
                        </body>

                        <!-- Right Leg -->
                        <body name="right_thigh" pos="-0.1 0 0">
                            <geom name="right_thigh_geom" type="capsule" size="0.06 0.25" quat="0.707 0 0.707 0"/>
                            <joint name="right_hip" type="hinge" pos="0 0 0" axis="0 1 0" range="-45 45"/>
                            <body name="right_shin" pos="0 0 -0.5">
                                <geom name="right_shin_geom" type="capsule" size="0.05 0.25" quat="0.707 0 0.707 0"/>
                                <joint name="right_knee" type="hinge" pos="0 0 0" axis="0 1 0" range="0 120"/>
                                <body name="right_foot" pos="0 0 -0.5">
                                    <geom name="right_foot_geom" type="box" size="0.1 0.05 0.02" pos="0 0 0.01"/>
                                    <joint name="right_ankle" type="hinge" pos="0 0 0" axis="0 1 0" range="-20 20"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </worldbody>

                <actuator>
                    <motor name="left_shoulder_actuator" joint="left_shoulder" gear="20"/>
                    <motor name="left_elbow_actuator" joint="left_elbow" gear="20"/>
                    <motor name="right_shoulder_actuator" joint="right_shoulder" gear="20"/>
                    <motor name="right_elbow_actuator" joint="right_elbow" gear="20"/>
                    <motor name="left_hip_actuator" joint="left_hip" gear="30"/>
                    <motor name="left_knee_actuator" joint="left_knee" gear="30"/>
                    <motor name="left_ankle_actuator" joint="left_ankle" gear="20"/>
                    <motor name="right_hip_actuator" joint="right_hip" gear="30"/>
                    <motor name="right_knee_actuator" joint="right_knee" gear="30"/>
                    <motor name="right_ankle_actuator" joint="right_ankle" gear="20"/>
                </actuator>
            </mujoco>
            """

            # Write to temporary file and load
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(xml_content)
                temp_path = f.name

            try:
                model = mujoco.MjModel.from_xml_path(temp_path)
            finally:
                os.unlink(temp_path)

            return model

        def get_joint_positions(self):
            """
            Get current joint positions
            """
            return self.data.qpos.copy()

        def get_joint_velocities(self):
            """
            Get current joint velocities
            """
            return self.data.qvel.copy()

        def get_body_position(self, body_name):
            """
            Get position of a specific body
            """
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            return self.data.geom_xpos[body_id].copy()

        def set_joint_commands(self, joint_commands):
            """
            Set joint position/velocity/torque commands

            Args:
                joint_commands: Array of joint commands (depends on control mode)
            """
            self.data.ctrl[:] = joint_commands

        def step_simulation(self, steps=1):
            """
            Step the simulation forward

            Args:
                steps: Number of simulation steps to take
            """
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)

        def reset_simulation(self):
            """
            Reset simulation to initial state
            """
            self.data.qpos[:] = 0
            self.data.qvel[:] = 0
            self.data.ctrl[:] = 0
            mujoco.mj_forward(self.model, self.data)

        def get_sensor_data(self):
            """
            Get simulated sensor data
            """
            sensor_data = {}

            # Joint positions and velocities
            sensor_data['joint_positions'] = self.get_joint_positions()
            sensor_data['joint_velocities'] = self.get_joint_velocities()

            # IMU simulation (approximate from body velocities)
            torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_XBODY, "torso")
            sensor_data['imu_linear_velocity'] = self.data.qvel[0:3]  # Global velocity
            sensor_data['imu_angular_velocity'] = self.data.qvel[3:6]  # Angular velocity

            # Accelerometer simulation (derive from velocity)
            if hasattr(self, 'prev_lin_vel'):
                dt = self.dt
                lin_acc = (sensor_data['imu_linear_velocity'] - self.prev_lin_vel) / dt
                sensor_data['imu_linear_acceleration'] = lin_acc
            else:
                sensor_data['imu_linear_acceleration'] = np.zeros(3)
            self.prev_lin_vel = sensor_data['imu_linear_velocity'].copy()

            return sensor_data

        def apply_force_to_body(self, body_name, force, point=None):
            """
            Apply external force to a body

            Args:
                body_name: Name of the body to apply force to
                force: Force vector [fx, fy, fz]
                point: Point [x, y, z] where force is applied (body center if None)
            """
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

            # Apply force
            if point is None:
                # Apply at body center
                mujoco.mj_applyFT(
                    self.model, self.data,
                    force, [0, 0, 0],  # Force and torque
                    [0, 0, 0],  # Point (will be applied at body origin)
                    body_id
                )
            else:
                # Apply at specific point
                mujoco.mj_applyFT(
                    self.model, self.data,
                    force, [0, 0, 0],  # Force and torque
                    point,  # Point
                    body_id
                )

        def detect_collisions(self):
            """
            Detect collisions in the simulation
            """
            collisions = []

            # Check contact information
            for i in range(self.data.ncon):
                contact = self.data.contact[i]

                # Get names of colliding geoms
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

                # Get contact details
                contact_info = {
                    'geom1': geom1_name,
                    'geom2': geom2_name,
                    'position': contact.pos.copy(),
                    'normal': contact.frame[0:3].copy(),  # Normal vector
                    'distance': contact.dist,
                    'force_magnitude': contact.efc_force[0] if len(contact.efc_force) > 0 else 0
                }

                collisions.append(contact_info)

            return collisions

        def run_simulation_with_viewer(self, duration=10.0):
            """
            Run simulation with viewer

            Args:
                duration: Duration of simulation in seconds
            """
            print(f"Starting MuJoCo simulation with viewer for {duration} seconds...")

            # Reset simulation
            self.reset_simulation()

            # Start viewer
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start_time = time.time()

                while time.time() - start_time < duration:
                    # Simple control: make joints oscillate
                    t = time.time() - start_time

                    # Create simple movement pattern
                    joint_commands = np.zeros(self.model.nu)

                    # Apply oscillating commands to some joints
                    for i in range(min(6, len(joint_commands))):  # First 6 joints
                        joint_commands[i] = 0.5 * np.sin(t * 2 + i * 0.5)

                    # Set commands
                    self.data.ctrl[:] = joint_commands

                    # Step simulation
                    self.step_simulation(1)

                    # Sync viewer
                    viewer.sync()

                    # Slow down to real-time
                    time.sleep(self.dt)

        def run_simulation_headless(self, duration=10.0):
            """
            Run simulation without viewer

            Args:
                duration: Duration of simulation in seconds
            """
            print(f"Starting headless MuJoCo simulation for {duration} seconds...")

            # Reset simulation
            self.reset_simulation()

            steps = int(duration / self.dt)
            start_time = time.time()

            for step in range(steps):
                # Simple control: make joints oscillate
                t = step * self.dt

                # Create simple movement pattern
                joint_commands = np.zeros(self.model.nu)

                # Apply oscillating commands to some joints
                for i in range(min(6, len(joint_commands))):  # First 6 joints
                    joint_commands[i] = 0.3 * np.sin(t * 2 + i * 0.5)

                # Set commands
                self.data.ctrl[:] = joint_commands

                # Step simulation
                self.step_simulation(1)

                # Detect collisions periodically
                if step % 100 == 0:  # Every 100 steps (0.2 seconds)
                    collisions = self.detect_collisions()
                    if collisions:
                        print(f"Step {step}: Detected {len(collisions)} collisions")

                # Print status periodically
                if step % 500 == 0:  # Every 500 steps (1 second)
                    sensor_data = self.get_sensor_data()
                    print(f"Step {step}, Time: {step*self.dt:.1f}s, Joint 0 pos: {sensor_data['joint_positions'][0]:.3f}")

            end_time = time.time()
            print(f"Headless simulation completed in {end_time - start_time:.2f} seconds")

    def mujoco_simulation_example():
        """
        Example usage of MuJoCo simulation
        """
        print("=== MuJoCo Humanoid Simulation Example ===")

        try:
            # Create simulation
            sim = MuJoCoHumanoidSimulation()

            # Run headless simulation
            sim.run_simulation_headless(duration=5.0)

            # Get final state
            final_positions = sim.get_joint_positions()
            print(f"\nFinal joint positions: {final_positions[:6]}...")  # First 6 joints

            # Apply external force
            print(f"\nApplying external force to torso...")
            sim.apply_force_to_body("torso", [50, 0, 0])  # 50N force in X direction

            # Run for a bit more to see the effect
            for _ in range(100):  # 100 steps = 0.2 seconds
                sim.step_simulation(1)

            # Get state after force application
            sensor_data = sim.get_sensor_data()
            print(f"Post-force linear velocity: {sensor_data['imu_linear_velocity']}")

        except ImportError:
            print("MuJoCo not installed. Skipping MuJoCo simulation example.")
            print("To run this example, install MuJoCo: pip install mujoco")

    if __name__ == "__main__":
        mujoco_simulation_example()

except ImportError:
    print("MuJoCo not available, skipping MuJoCo example")
```

## Integration with AI Training

Here's an example of how to integrate simulation with AI training:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

class SimulationEnvironment:
    """
    Wrapper for simulation environment that can interface with AI training
    """
    def __init__(self, simulation_instance):
        self.sim = simulation_instance
        self.action_space = 10  # Example: 10 joint control commands
        self.observation_space = 20  # Example: joint positions, velocities, IMU data
        self.max_episode_steps = 1000
        self.current_step = 0

    def reset(self):
        """
        Reset the environment and return initial observation
        """
        self.sim.reset_simulation()
        self.current_step = 0
        return self.get_observation()

    def step(self, action):
        """
        Take a step in the environment

        Args:
            action: Action to take (array of joint commands)

        Returns:
            observation, reward, done, info
        """
        # Apply action to simulation
        self.sim.set_joint_commands(action)

        # Step simulation
        self.sim.step_simulation(1)

        # Get new observation
        observation = self.get_observation()

        # Calculate reward
        reward = self.calculate_reward(action, observation)

        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= self.max_episode_steps

        # Get additional info
        info = {
            'step': self.current_step,
            'collisions': len(self.sim.detect_collisions()),
            'torso_height': self.sim.get_body_position('torso')[2] if hasattr(self.sim, 'get_body_position') else 0
        }

        return observation, reward, done, info

    def get_observation(self):
        """
        Get current observation from simulation
        """
        # This would combine various sensor readings
        joint_positions = self.sim.get_joint_positions()
        joint_velocities = self.sim.get_joint_velocities()
        sensor_data = self.sim.get_sensor_data()

        # Combine into observation vector
        observation = np.concatenate([
            joint_positions[:10],      # First 10 joint positions
            joint_velocities[:10],     # First 10 joint velocities
            sensor_data.get('imu_linear_velocity', np.zeros(3)),  # IMU linear velocity
            sensor_data.get('imu_angular_velocity', np.zeros(3))  # IMU angular velocity
        ])

        # Ensure observation is the right size
        if len(observation) < self.observation_space:
            observation = np.pad(observation, (0, self.observation_space - len(observation)), 'constant')
        elif len(observation) > self.observation_space:
            observation = observation[:self.observation_space]

        return observation

    def calculate_reward(self, action, observation):
        """
        Calculate reward based on action and resulting state
        """
        reward = 0.0

        # Positive reward for staying upright
        torso_height = observation[-1] if len(observation) > 0 else 0.8  # Simplified
        if torso_height > 0.5:  # Above half a meter
            reward += 0.1

        # Negative reward for falling
        if torso_height < 0.3:  # Below 30cm (probably fallen)
            reward -= 1.0

        # Small penalty for large control actions (energy efficiency)
        action_penalty = 0.001 * np.sum(np.square(action))
        reward -= action_penalty

        # Small penalty for each step (to encourage efficiency)
        reward -= 0.01

        return reward

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for humanoid robot control
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        return self.network(state)

class DQNAgent:
    """
    DQN Agent for humanoid robot control
    """
    def __init__(self, state_size, action_size, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Copy weights to target network
        self.update_target_network()

    def update_target_network(self):
        """
        Copy weights from main network to target network
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose action using epsilon-greedy policy
        """
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """
        Train the model on a batch of experiences
        """
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_ai_with_simulation():
    """
    Example of training AI with simulation environment
    """
    print("=== Training AI with Simulation Environment ===")

    # Create simulation (using PyBullet as example)
    try:
        sim = HumanoidSimulation(gui=False)  # Headless mode for training
        env = SimulationEnvironment(sim)

        # Create AI agent
        state_size = env.observation_space
        action_size = env.action_space
        agent = DQNAgent(state_size, action_size)

        # Training parameters
        episodes = 100
        max_steps_per_episode = 500

        print(f"Starting training for {episodes} episodes...")

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps_per_episode):
                # Choose action
                action = agent.act(state)

                # Take action in environment
                next_state, reward, done, info = env.step([action/10.0] * action_size)  # Normalize action

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Train agent
                if len(agent.memory) > 32:
                    agent.replay(32)

                state = next_state
                total_reward += reward

                if done:
                    break

            # Update target network periodically
            if episode % 10 == 0:
                agent.update_target_network()

            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        print("Training completed!")

        # Clean up
        sim.cleanup()

    except Exception as e:
        print(f"Error in AI training example: {e}")
        print("This example requires the PyBullet simulation to be available.")

if __name__ == "__main__":
    # Run the simulation examples
    simulation_example()

    print("\n" + "="*50 + "\n")

    try:
        mujoco_simulation_example()
    except:
        print("Skipping MuJoCo example (not installed)")

    print("\n" + "="*50 + "\n")

    # Run AI training example
    train_ai_with_simulation()
```

## Visualization and Analysis Tools

Here's an example of tools for analyzing simulation data:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class SimulationAnalyzer:
    """
    Tools for analyzing simulation data and performance
    """
    def __init__(self):
        self.episode_data = []
        self.step_data = []

    def record_episode_data(self, episode_num, total_reward, steps_taken, success):
        """
        Record data about an episode
        """
        episode_record = {
            'episode': episode_num,
            'total_reward': total_reward,
            'steps_taken': steps_taken,
            'success': success,
            'timestamp': time.time()
        }
        self.episode_data.append(episode_record)

    def record_step_data(self, step_num, observation, action, reward, done):
        """
        Record data about a simulation step
        """
        step_record = {
            'step': step_num,
            'observation': observation.copy(),
            'action': action,
            'reward': reward,
            'done': done,
            'timestamp': time.time()
        }
        self.step_data.append(step_record)

    def plot_training_progress(self):
        """
        Plot training progress over episodes
        """
        if not self.episode_data:
            print("No episode data to plot")
            return

        df = pd.DataFrame(self.episode_data)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Total reward over time
        axes[0, 0].plot(df['episode'], df['total_reward'])
        axes[0, 0].set_title('Total Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')

        # Steps taken per episode
        axes[0, 1].plot(df['episode'], df['steps_taken'])
        axes[0, 1].set_title('Steps Taken per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')

        # Success rate (rolling average)
        df['success_rolling'] = df['success'].rolling(window=10, min_periods=1).mean()
        axes[1, 0].plot(df['episode'], df['success_rolling'])
        axes[1, 0].set_title('Success Rate (Rolling Average)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success Rate')

        # Reward histogram
        axes[1, 1].hist(df['total_reward'], bins=20)
        axes[1, 1].set_title('Distribution of Episode Rewards')
        axes[1, 1].set_xlabel('Total Reward')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def analyze_joint_trajectories(self, joint_indices=[0, 1, 2]):
        """
        Analyze joint trajectories over time
        """
        if not self.step_data:
            print("No step data to analyze")
            return

        # Extract joint positions for specified joints over time
        times = []
        joint_positions = {idx: [] for idx in joint_indices}

        for step_record in self.step_data:
            times.append(step_record['timestamp'])
            obs = step_record['observation']
            for idx in joint_indices:
                if idx < len(obs):
                    joint_positions[idx].append(obs[idx])

        # Plot joint trajectories
        plt.figure(figsize=(12, 6))
        for idx in joint_indices:
            if joint_positions[idx]:
                plt.plot(times, joint_positions[idx], label=f'Joint {idx}', linewidth=2)

        plt.title('Joint Position Trajectories')
        plt.xlabel('Time')
        plt.ylabel('Joint Position (rad)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def generate_performance_report(self):
        """
        Generate a performance report
        """
        if not self.episode_data:
            return "No data available for report."

        df = pd.DataFrame(self.episode_data)

        report = f"""
        SIMULATION PERFORMANCE REPORT
        ============================

        Total Episodes: {len(df)}
        Average Reward: {df['total_reward'].mean():.2f} (+/- {df['total_reward'].std():.2f})
        Average Steps: {df['steps_taken'].mean():.1f}
        Success Rate: {(df['success'].sum() / len(df) * 100):.1f}%

        Reward Statistics:
        - Best Episode: #{df['total_reward'].idxmax()} (Reward: {df['total_reward'].max():.2f})
        - Worst Episode: #{df['total_reward'].idxmin()} (Reward: {df['total_reward'].min():.2f})
        - Median Reward: {df['total_reward'].median():.2f}

        Performance Trend:
        - First 10% Avg: {df.head(len(df)//10)['total_reward'].mean():.2f}
        - Last 10% Avg: {df.tail(len(df)//10)['total_reward'].mean():.2f}
        - Improvement: {(df.tail(len(df)//10)['total_reward'].mean() - df.head(len(df)//10)['total_reward'].mean()):.2f}
        """

        return report

def simulation_analysis_example():
    """
    Example of using analysis tools
    """
    print("=== Simulation Analysis Tools Example ===")

    # Create analyzer
    analyzer = SimulationAnalyzer()

    # Simulate some training data
    for episode in range(50):
        total_reward = np.random.normal(100, 30) + episode * 0.5  # Improving over time
        steps_taken = np.random.randint(200, 500)
        success = episode > 10  # Success after first 10 episodes

        analyzer.record_episode_data(episode, total_reward, steps_taken, success)

        # Record some step data
        for step in range(steps_taken):
            obs = np.random.randn(20)  # Simulated observation
            action = np.random.randint(0, 10)  # Simulated action
            reward = np.random.randn()  # Simulated reward
            done = step == steps_taken - 1

            analyzer.record_step_data(step, obs, action, reward, done)

    # Generate report
    report = analyzer.generate_performance_report()
    print(report)

    # Plot results
    analyzer.plot_training_progress()
    analyzer.analyze_joint_trajectories([0, 1, 2])

if __name__ == "__main__":
    simulation_analysis_example()
```

## Summary

Simulation is an essential component of humanoid robotics development, providing:

1. **Safe Testing Environment**: Test algorithms without risking physical hardware
2. **Rapid Prototyping**: Quickly iterate on control algorithms and behaviors
3. **AI Training**: Train neural networks and reinforcement learning agents
4. **Performance Analysis**: Analyze robot behavior and optimize performance
5. **Scenario Testing**: Test various environmental conditions and edge cases

The examples provided demonstrate different simulation approaches using PyBullet, Gazebo, and MuJoCo, each with their own strengths and use cases. These simulation environments can be integrated with AI training pipelines to develop sophisticated humanoid robot behaviors in a safe, reproducible environment before deployment on physical hardware.