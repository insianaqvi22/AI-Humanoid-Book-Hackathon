---
title: Basic Movements Implementation
sidebar_position: 1
description: Implementing basic movement patterns for AI humanoid robotics
---

# Basic Movements Implementation

This chapter covers the practical implementation of basic movement patterns in humanoid robots. We'll explore how to translate the theoretical concepts from previous chapters into actual code that controls robot actuators and produces human-like movements.

## Introduction to Movement Implementation

Implementing basic movements in humanoid robots requires:
- Understanding of robot kinematics
- Proper control of multiple actuators simultaneously
- Real-time execution capabilities
- Safety and stability considerations
- Smooth transitions between movements

## Kinematics Implementation

### Forward Kinematics

Forward kinematics calculates the position of end effectors (hands, feet) given joint angles.

```python
import numpy as np

class ForwardKinematics:
    def __init__(self, dh_parameters):
        """
        Initialize with Denavit-Hartenberg parameters
        dh_parameters: List of [a, alpha, d, theta_offset] for each joint
        """
        self.dh_params = dh_parameters

    def dh_transform(self, a, alpha, d, theta):
        """
        Calculate Denavit-Hartenberg transformation matrix
        """
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def calculate_position(self, joint_angles):
        """
        Calculate end effector position given joint angles
        """
        T = np.eye(4)  # Identity transformation

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i

        # Extract position (x, y, z)
        position = T[:3, 3]
        return position

    def calculate_full_chain(self, joint_angles):
        """
        Calculate positions of all joints in the chain
        """
        T = np.eye(4)
        joint_positions = [np.array([0, 0, 0])]  # Base position

        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T = T @ T_i
            joint_positions.append(T[:3, 3])

        return np.array(joint_positions)
```

### Inverse Kinematics

Inverse kinematics calculates the joint angles needed to achieve a desired end effector position.

```python
class InverseKinematics:
    def __init__(self, dh_parameters, max_iterations=100, tolerance=1e-6):
        self.dh_params = dh_parameters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fk = ForwardKinematics(dh_parameters)

    def jacobian(self, joint_angles):
        """
        Calculate the Jacobian matrix for the robot arm
        """
        n = len(joint_angles)
        J = np.zeros((6, n))  # 6 DoF: 3 for position, 3 for orientation

        # Calculate all joint positions and orientations
        joint_positions = self.fk.calculate_full_chain(joint_angles)
        end_effector_pos = joint_positions[-1]

        for i in range(n):
            # Calculate the axis of rotation for joint i
            # This is a simplified version - in practice, you'd use the full DH approach
            joint_pos = joint_positions[i]
            z_axis = np.array([0, 0, 1])  # Assumption for revolute joints

            # Position contribution
            r = end_effector_pos - joint_pos
            J[:3, i] = np.cross(z_axis, r)
            # Orientation contribution
            J[3:, i] = z_axis

        return J

    def solve(self, target_position, initial_angles, max_attempts=5):
        """
        Solve inverse kinematics using Jacobian transpose method
        """
        current_angles = np.array(initial_angles)

        for attempt in range(max_attempts):
            for iteration in range(self.max_iterations):
                # Calculate current end effector position
                current_pos = self.fk.calculate_position(current_angles)

                # Calculate error
                error = target_position - current_pos

                # Check if we're close enough
                if np.linalg.norm(error) < self.tolerance:
                    return current_angles

                # Calculate Jacobian
                J = self.jacobian(current_angles)

                # Calculate angle adjustments using Jacobian transpose
                angle_delta = J.T @ error * 0.01  # Small step size
                current_angles += angle_delta

                # Check joint limits (not shown here for simplicity)

            # If this attempt didn't converge, try a different starting point
            if np.linalg.norm(error) >= self.tolerance:
                # Add some random perturbation to escape local minima
                current_angles = np.array(initial_angles) + np.random.normal(0, 0.1, len(initial_angles))

        # Return best solution found
        return current_angles

    def solve_reach(self, target_pos, current_angles, arm_length=0.7):
        """
        Simple reach movement for arm
        """
        # Calculate distance to target
        current_pos = self.fk.calculate_position(current_angles)
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance > arm_length * 0.9:  # Cannot reach, move as close as possible
            direction = direction / distance  # Normalize
            target_pos = current_pos + direction * arm_length * 0.9

        # Use full inverse kinematics solver
        return self.solve(target_pos, current_angles)
```

## Basic Movement Patterns

### Standing Position

The neutral standing position is the foundation for all other movements.

```python
class StandingController:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.standing_angles = self.calculate_standing_angles()
        self.balance_controller = BalanceController(robot_model.mass, robot_model.com_height)

    def calculate_standing_angles(self):
        """
        Calculate joint angles for stable standing position
        """
        # Default standing position - neutral joint angles
        standing_angles = {
            'left_hip': [0, 0, 0],      # [roll, pitch, yaw]
            'left_knee': [0],            # [flexion]
            'left_ankle': [0, 0],        # [pitch, roll]
            'right_hip': [0, 0, 0],
            'right_knee': [0],
            'right_ankle': [0, 0],
            'left_shoulder': [0, 0, 0],  # [abduction, flexion, rotation]
            'left_elbow': [0],           # [flexion]
            'left_wrist': [0, 0, 0],     # [pitch, yaw, roll]
            'right_shoulder': [0, 0, 0],
            'right_elbow': [0],
            'right_wrist': [0, 0, 0],
            'torso': [0, 0, 0],          # [bend, twist, side-bend]
            'head': [0, 0]               # [pan, tilt]
        }
        return standing_angles

    def maintain_balance(self, sensor_data):
        """
        Maintain balance while standing
        """
        # Read IMU data
        imu_data = sensor_data['imu']
        com_position = sensor_data['com_position']
        foot_positions = sensor_data['foot_positions']

        # Calculate ZMP
        zmp = self.balance_controller.calculate_zmp(com_position, sensor_data['com_acceleration'], 0.8)

        # Check if ZMP is within support polygon
        support_polygon = self.calculate_support_polygon(foot_positions)
        if not self.balance_controller.is_stable(zmp, support_polygon):
            # Apply balance correction
            correction_angles = self.balance_controller.calculate_balance_correction(
                zmp, com_position
            )
            return self.apply_balance_correction(correction_angles)
        else:
            return self.standing_angles  # Return to neutral position gradually

    def calculate_support_polygon(self, foot_positions):
        """
        Calculate support polygon based on foot positions
        """
        # Simplified rectangular support polygon
        x_coords = [pos[0] for pos in foot_positions]
        y_coords = [pos[1] for pos in foot_positions]

        return (min(x_coords), max(x_coords)), (min(y_coords), max(y_coords))
```

### Simple Arm Movements

Basic arm movements form the foundation for more complex manipulation.

```python
class ArmController:
    def __init__(self, robot_model, arm_side='left'):
        self.robot = robot_model
        self.arm_side = arm_side
        self.ik_solver = InverseKinematics(self.get_arm_dh_params())
        self.current_position = self.get_current_arm_position()

    def get_arm_dh_params(self):
        """
        Define DH parameters for the arm
        This is a simplified example - real robots have more complex kinematics
        """
        # [a, alpha, d, theta_offset] for each joint
        # This is an example for a simple 3-DOF arm
        return [
            [0.1, np.pi/2, 0.3, 0],      # Shoulder joint
            [0.3, 0, 0, 0],              # Elbow joint
            [0.25, 0, 0, 0]              # Wrist joint
        ]

    def move_to_position(self, target_position, duration=2.0):
        """
        Move arm to a specific position using inverse kinematics
        """
        # Solve for joint angles
        initial_angles = self.get_current_joint_angles()
        target_angles = self.ik_solver.solve(target_position, initial_angles)

        # Generate trajectory
        trajectory = self.generate_trajectory(initial_angles, target_angles, duration)

        return trajectory

    def generate_trajectory(self, start_angles, end_angles, duration, dt=0.01):
        """
        Generate smooth trajectory between start and end angles
        """
        steps = int(duration / dt)
        trajectory = []

        for i in range(steps + 1):
            t = i / steps  # Normalized time (0 to 1)

            # Use cubic interpolation for smooth motion
            # s(t) = 3t² - 2t³ (smoother than linear)
            smooth_t = 3 * t**2 - 2 * t**3

            angles = start_angles + smooth_t * (end_angles - start_angles)
            trajectory.append({
                'time': i * dt,
                'joint_angles': angles,
                'position': self.calculate_arm_position(angles)
            })

        return trajectory

    def wave_hand(self, amplitude=0.1, frequency=1.0, duration=3.0):
        """
        Create a waving motion
        """
        trajectory = []
        dt = 0.01
        steps = int(duration / dt)

        base_position = self.current_position.copy()

        for i in range(steps + 1):
            t = i * dt
            # Create waving motion (circular pattern)
            x_offset = 0
            y_offset = amplitude * np.sin(2 * np.pi * frequency * t)
            z_offset = amplitude * np.cos(2 * np.pi * frequency * t)

            wave_position = base_position + np.array([x_offset, y_offset, z_offset])
            angles = self.ik_solver.solve(wave_position, self.get_current_joint_angles())

            trajectory.append({
                'time': t,
                'joint_angles': angles,
                'position': wave_position
            })

        return trajectory

    def point_to_target(self, target_position):
        """
        Point arm towards a target position
        """
        # Calculate direction vector
        current_pos = self.current_position
        direction = target_position - current_pos
        distance = np.linalg.norm(direction)

        # Normalize and extend to maximum reach
        max_reach = 0.7  # meters
        if distance > max_reach:
            direction = direction / distance * max_reach

        point_position = current_pos + direction

        # Solve for angles to reach this position
        current_angles = self.get_current_joint_angles()
        target_angles = self.ik_solver.solve(point_position, current_angles)

        return target_angles

    def get_current_joint_angles(self):
        """
        Get current joint angles from robot state
        """
        # This would interface with the actual robot
        # For simulation, return some default values
        return np.array([0.0, 0.0, 0.0])  # [shoulder, elbow, wrist]

    def calculate_arm_position(self, angles):
        """
        Calculate end effector position from joint angles
        """
        fk = ForwardKinematics(self.get_arm_dh_params())
        return fk.calculate_position(angles)
```

### Basic Walking Patterns

Implementing simple walking patterns.

```python
class WalkingController:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.gait_generator = GaitPatternGenerator()
        self.balance_controller = BalanceController(robot_model.mass, robot_model.com_height)

    def generate_walk_pattern(self, steps, step_length=0.3, step_height=0.05):
        """
        Generate a walking pattern for a specified number of steps
        """
        pattern = []
        current_x = 0.0
        current_y = 0.0

        for step in range(steps):
            # Generate step pattern for this step
            step_pattern = self.gait_generator.generate_step(
                current_x, current_y, step_length, step_height
            )

            # Add timing information
            for i, pose in enumerate(step_pattern):
                time_from_start = step * 1.0 + i * 0.01  # 1 second per step, 100Hz
                pose['time'] = time_from_start
                pose['step_number'] = step

            pattern.extend(step_pattern)

            # Update current position
            current_x += step_length if step % 2 == 0 else -step_length  # Alternate feet
            current_y = 0.1 if current_y == 0 else 0  # Alternate foot positions

        return pattern

    def generate_turn_pattern(self, angle, steps=4):
        """
        Generate a turning pattern
        """
        pattern = []
        step_length = 0.3
        turn_radius = 0.5  # meters

        for step in range(steps):
            # Calculate turning trajectory
            angle_per_step = angle / steps
            step_angle = step * angle_per_step

            # Calculate step position in arc
            x = turn_radius * np.sin(step_angle)
            y = turn_radius * (1 - np.cos(step_angle))

            step_pattern = self.gait_generator.generate_step(x, y, step_length, 0.05)
            pattern.extend(step_pattern)

        return pattern

    def stand_up(self):
        """
        Generate movement pattern to transition from lying to standing
        """
        pattern = []

        # Phase 1: Push up to hands and knees
        push_up_pattern = self.generate_push_up_sequence()
        pattern.extend(push_up_pattern)

        # Phase 2: Rise to standing position
        rise_pattern = self.generate_rise_sequence()
        pattern.extend(rise_pattern)

        # Phase 3: Adjust to stable standing position
        adjust_pattern = self.generate_adjustment_sequence()
        pattern.extend(adjust_pattern)

        return pattern

    def generate_push_up_sequence(self):
        """
        Generate sequence to get from lying to hands-knees position
        """
        sequence = []

        # Roll to side
        for t in np.linspace(0, 1, 50):
            # Simulate rolling motion
            angles = self.interpolate_to_side_position(t)
            sequence.append({
                'time': t * 2,  # 2 seconds for this phase
                'joint_angles': angles,
                'phase': 'rolling'
            })

        # Push up to hands and knees
        for t in np.linspace(0, 1, 50):
            angles = self.interpolate_to_hands_knees(t)
            sequence.append({
                'time': 2 + t * 3,  # 3 seconds for push-up
                'joint_angles': angles,
                'phase': 'pushing_up'
            })

        return sequence

    def generate_rise_sequence(self):
        """
        Generate sequence to rise from hands-knees to standing
        """
        sequence = []

        # Rise to half-kneel
        for t in np.linspace(0, 1, 40):
            angles = self.interpolate_to_half_kneel(t)
            sequence.append({
                'time': t * 2,
                'joint_angles': angles,
                'phase': 'rising'
            })

        # Rise to standing
        for t in np.linspace(0, 1, 60):
            angles = self.interpolate_to_standing(t)
            sequence.append({
                'time': 2 + t * 3,
                'joint_angles': angles,
                'phase': 'final_rise'
            })

        return sequence

    def generate_adjustment_sequence(self):
        """
        Generate sequence to adjust to stable standing position
        """
        sequence = []

        # Fine-tune balance
        for t in np.linspace(0, 1, 30):
            angles = self.balance_controller.adjust_standing_balance(t)
            sequence.append({
                'time': t * 1,
                'joint_angles': angles,
                'phase': 'balancing'
            })

        return sequence

    def interpolate_to_side_position(self, t):
        """
        Interpolate to side-lying position
        """
        # This would be specific to the robot's joint configuration
        # Simplified example
        return np.array([0, 0, 0, 0, 0, 0])  # Placeholder joint angles

    def interpolate_to_hands_knees(self, t):
        """
        Interpolate to hands and knees position
        """
        # Simplified example
        return np.array([0, 0, 0, 0, 0, 0])  # Placeholder joint angles

    def interpolate_to_half_kneel(self, t):
        """
        Interpolate to half-kneeling position
        """
        # Simplified example
        return np.array([0, 0, 0, 0, 0, 0])  # Placeholder joint angles

    def interpolate_to_standing(self, t):
        """
        Interpolate to full standing position
        """
        # Simplified example
        return np.array([0, 0, 0, 0, 0, 0])  # Placeholder joint angles
```

### Gait Pattern Generation

Creating the actual walking motion patterns.

```python
class GaitPatternGenerator:
    def __init__(self):
        self.step_height = 0.05  # meters
        self.step_duration = 1.0  # seconds
        self.double_support_ratio = 0.2  # 20% of step in double support

    def generate_step(self, start_x, start_y, step_length, step_height=None):
        """
        Generate a single step pattern
        """
        if step_height is None:
            step_height = self.step_height

        pattern = []
        dt = 0.01  # 100Hz control rate
        steps = int(self.step_duration / dt)

        for i in range(steps + 1):
            t = i * dt / self.step_duration  # Normalized time (0 to 1)

            # Calculate foot position
            foot_x = start_x + t * step_length
            foot_y = start_y  # Keep constant for straight line

            # Calculate foot height (parabolic trajectory)
            height_t = t * 2  # Speed up the height calculation
            if height_t > 1:
                height_t = 2 - height_t  # Come back down
            foot_z = step_height * np.sin(np.pi * height_t) if height_t <= 1 else step_height * np.sin(np.pi * (2 - height_t))

            # Calculate CoM trajectory (smooth transition)
            com_x = start_x + t * step_length * 0.9  # CoM moves slightly slower
            com_y = 0  # Maintain balance between feet
            com_z = 0.8  # Maintain constant height

            pattern.append({
                'time': i * dt,
                'swing_foot': np.array([foot_x, foot_y, foot_z]),
                'stance_foot': np.array([start_x - step_length/2, start_y, 0]),  # Previous foot position
                'com_position': np.array([com_x, com_y, com_z]),
                'phase': 'single_support' if self.is_single_support_phase(t) else 'double_support'
            })

        return pattern

    def is_single_support_phase(self, normalized_time):
        """
        Determine if we're in single or double support phase
        """
        double_support_start = self.double_support_ratio / 2
        double_support_end = 1 - self.double_support_ratio / 2

        return double_support_start <= normalized_time <= double_support_end

    def generate_omega_trajectory(self, start_pos, end_pos, duration=1.0, dt=0.01):
        """
        Generate smooth omega-shaped trajectory for foot movement
        This creates a more natural lifting and placing motion
        """
        steps = int(duration / dt)
        trajectory = []

        for i in range(steps + 1):
            t = i / steps  # Normalized time (0 to 1)

            # Use smooth interpolation
            smooth_t = self.smooth_step(t)

            # Calculate position with omega shape
            x = start_pos[0] + smooth_t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + smooth_t * (end_pos[1] - start_pos[1])

            # Add lift for natural foot movement
            lift_factor = np.sin(np.pi * t) * 0.05  # 5cm lift
            z = start_pos[2] + smooth_t * (end_pos[2] - start_pos[2]) + lift_factor

            trajectory.append(np.array([x, y, z]))

        return trajectory

    def smooth_step(self, t):
        """
        Smooth step function (3t² - 2t³) for smooth interpolation
        """
        return 3 * t**2 - 2 * t**3

    def generate_balance_motion(self, amplitude=0.02, frequency=0.5, duration=5.0):
        """
        Generate subtle balance motion to look more natural
        """
        trajectory = []
        dt = 0.01
        steps = int(duration / dt)

        for i in range(steps + 1):
            t = i * dt
            # Generate subtle swaying motion
            sway_x = amplitude * np.sin(2 * np.pi * frequency * t)
            sway_y = amplitude * 0.5 * np.cos(2 * np.pi * frequency * t)

            trajectory.append({
                'time': t,
                'sway_offset': np.array([sway_x, sway_y, 0])
            })

        return trajectory
```

## Movement Execution and Control

### Trajectory Following

Implementing controllers to follow the generated trajectories.

```python
class TrajectoryFollower:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.controllers = {}  # Individual joint controllers
        self.initialize_controllers()

    def initialize_controllers(self):
        """
        Initialize PID controllers for each joint
        """
        joint_names = self.robot.get_joint_names()
        for joint_name in joint_names:
            # Default PID values - should be tuned for specific robot
            self.controllers[joint_name] = PIDController(kp=10, ki=0.1, kd=0.5)

    def follow_trajectory(self, trajectory, controller_type='position'):
        """
        Follow a generated trajectory
        """
        execution_log = []

        for i, waypoint in enumerate(trajectory):
            # Get current time
            current_time = waypoint['time']

            # Get target joint angles
            target_angles = waypoint['joint_angles']

            # Get current joint angles from robot
            current_angles = self.robot.get_current_joint_angles()

            # Calculate control commands
            control_commands = {}
            for joint_idx, joint_name in enumerate(self.robot.get_joint_names()):
                if controller_type == 'position':
                    command = self.controllers[joint_name].compute(
                        target_angles[joint_idx],
                        current_angles[joint_idx]
                    )
                elif controller_type == 'velocity':
                    # Calculate desired velocity from trajectory
                    if i > 0:
                        prev_waypoint = trajectory[i-1]
                        time_diff = waypoint['time'] - prev_waypoint['time']
                        if time_diff > 0:
                            desired_vel = (target_angles[joint_idx] - prev_waypoint['joint_angles'][joint_idx]) / time_diff
                            current_vel = self.robot.get_joint_velocity(joint_name)
                            command = self.controllers[joint_name].compute(desired_vel, current_vel)
                        else:
                            command = 0
                    else:
                        command = 0

                control_commands[joint_name] = command

            # Send commands to robot
            self.robot.send_joint_commands(control_commands)

            # Log execution
            execution_log.append({
                'time': current_time,
                'target_angles': target_angles,
                'current_angles': current_angles,
                'commands': control_commands,
                'error': np.abs(target_angles - current_angles)
            })

            # Wait for next control cycle
            self.robot.wait_for_control_cycle()

        return execution_log

    def blend_trajectories(self, trajectory1, trajectory2, blend_ratio=0.5):
        """
        Blend between two trajectories
        """
        if len(trajectory1) != len(trajectory2):
            raise ValueError("Trajectories must have the same length")

        blended_trajectory = []
        for wp1, wp2 in zip(trajectory1, trajectory2):
            if wp1['time'] != wp2['time']:
                raise ValueError("Trajectory waypoints must be synchronized")

            # Blend joint angles
            blended_angles = (1 - blend_ratio) * wp1['joint_angles'] + blend_ratio * wp2['joint_angles']

            blended_trajectory.append({
                'time': wp1['time'],
                'joint_angles': blended_angles,
                'blended_from': ['trajectory1', 'trajectory2']
            })

        return blended_trajectory
```

### Safety and Error Handling

Implementing safety measures during movement execution.

```python
class MovementSafety:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.emergency_stop = False
        self.safety_limits = self.define_safety_limits()

    def define_safety_limits(self):
        """
        Define safety limits for the robot
        """
        return {
            'joint_limits': {
                'hip_pitch': (-1.57, 1.57),  # radians
                'knee_pitch': (0, 2.35),
                'ankle_pitch': (-0.5, 0.5),
                'shoulder_pitch': (-2.09, 1.57),
                'elbow_pitch': (-2.35, 0),
                'wrist_yaw': (-1.57, 1.57)
            },
            'torque_limits': {
                'hip': 50.0,   # Nm
                'knee': 40.0,
                'ankle': 20.0,
                'shoulder': 30.0,
                'elbow': 15.0
            },
            'velocity_limits': {
                'max_joint_vel': 5.0,  # rad/s
                'max_cartesian_vel': 0.5  # m/s
            },
            'balance_threshold': 0.1  # meters - max ZMP deviation
        }

    def check_safety(self, current_state, desired_state):
        """
        Check if the movement is safe to execute
        """
        issues = []

        # Check joint limits
        joint_angles = current_state.get('joint_angles', [])
        for i, (joint_name, angle) in enumerate(zip(self.robot.get_joint_names(), joint_angles)):
            if joint_name in self.safety_limits['joint_limits']:
                min_limit, max_limit = self.safety_limits['joint_limits'][joint_name]
                if angle < min_limit or angle > max_limit:
                    issues.append(f"Joint {joint_name} limit violation: {angle} rad")

        # Check torque limits
        torques = current_state.get('joint_torques', [])
        for i, (joint_name, torque) in enumerate(zip(self.robot.get_joint_names(), torques)):
            if joint_name in self.safety_limits['torque_limits']:
                max_torque = self.safety_limits['torque_limits'][joint_name]
                if abs(torque) > max_torque:
                    issues.append(f"Torque limit violation for {joint_name}: {torque} Nm")

        # Check balance (ZMP within support polygon)
        zmp = current_state.get('zmp', np.array([0, 0, 0]))
        support_polygon = current_state.get('support_polygon', None)
        if support_polygon is not None:
            if not self.is_zmp_stable(zmp, support_polygon):
                issues.append(f"Balance violation: ZMP {zmp[:2]} outside support polygon")

        return len(issues) == 0, issues

    def is_zmp_stable(self, zmp, support_polygon):
        """
        Check if ZMP is within support polygon
        """
        # Simplified rectangular support polygon check
        x_range, y_range = support_polygon
        x_min, x_max = x_range
        y_min, y_max = y_range

        return x_min <= zmp[0] <= x_max and y_min <= zmp[1] <= y_max

    def emergency_stop(self):
        """
        Execute emergency stop procedure
        """
        self.emergency_stop = True

        # Send zero commands to all joints
        zero_commands = {joint: 0 for joint in self.robot.get_joint_names()}
        self.robot.send_joint_commands(zero_commands)

        # Move to safe position if possible
        try:
            safe_position = self.robot.get_safe_position()
            self.robot.move_to_position(safe_position, duration=1.0)
        except:
            # If can't move to safe position, just stop all motion
            pass

        return "Emergency stop executed"

    def adaptive_control(self, current_state, desired_trajectory, time_step):
        """
        Adaptive control that adjusts based on safety constraints
        """
        # Check current safety status
        is_safe, issues = self.check_safety(current_state, desired_trajectory)

        if not is_safe:
            # Adjust the trajectory to maintain safety
            adjusted_trajectory = self.adjust_trajectory_for_safety(
                desired_trajectory, issues, current_state
            )
            return adjusted_trajectory
        else:
            # Trajectory is safe, return as is
            return desired_trajectory

    def adjust_trajectory_for_safety(self, trajectory, issues, current_state):
        """
        Adjust trajectory to address safety issues
        """
        adjusted = trajectory.copy()

        for issue in issues:
            if "limit violation" in issue:
                # Reduce speed to respect limits
                adjusted = self.reduce_speed(adjusted, factor=0.5)
            elif "torque" in issue:
                # Reduce acceleration to lower torque requirements
                adjusted = self.smooth_trajectory(adjusted)
            elif "balance" in issue:
                # Adjust CoM trajectory to improve balance
                adjusted = self.adjust_balance(adjusted, current_state)

        return adjusted

    def reduce_speed(self, trajectory, factor=0.5):
        """
        Reduce trajectory speed by the given factor
        """
        # Stretch the time dimension
        adjusted = []
        for waypoint in trajectory:
            new_waypoint = waypoint.copy()
            new_waypoint['time'] = waypoint['time'] / factor
            adjusted.append(new_waypoint)
        return adjusted

    def smooth_trajectory(self, trajectory):
        """
        Smooth the trajectory to reduce accelerations
        """
        # Apply simple smoothing (in practice, use more sophisticated methods)
        if len(trajectory) < 3:
            return trajectory

        smoothed = [trajectory[0]]  # Keep first point
        for i in range(1, len(trajectory) - 1):
            prev_angles = trajectory[i-1]['joint_angles']
            curr_angles = trajectory[i]['joint_angles']
            next_angles = trajectory[i+1]['joint_angles']

            # Average with neighbors
            smoothed_angles = (prev_angles + 2*curr_angles + next_angles) / 4

            new_waypoint = trajectory[i].copy()
            new_waypoint['joint_angles'] = smoothed_angles
            smoothed.append(new_waypoint)

        smoothed.append(trajectory[-1])  # Keep last point
        return smoothed
```

## Practical Implementation Tips

### Real-Time Considerations

```python
class RealTimeMovementController:
    def __init__(self, control_frequency=100):  # 100Hz
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.last_control_time = 0

    def wait_for_control_cycle(self):
        """
        Wait for the next control cycle
        """
        import time
        current_time = time.time()
        elapsed = current_time - self.last_control_time

        if elapsed < self.control_period:
            time.sleep(self.control_period - elapsed)

        self.last_control_time = time.time()

    def execute_with_timing(self, movement_function, *args, **kwargs):
        """
        Execute a movement function with proper timing
        """
        import time
        start_time = time.time()

        result = movement_function(*args, **kwargs)

        execution_time = time.time() - start_time
        if execution_time > self.control_period:
            print(f"Warning: Movement function took {execution_time:.3f}s, exceeding control period {self.control_period:.3f}s")

        return result
```

### Calibration and Tuning

```python
class MovementCalibrator:
    def __init__(self, robot_model):
        self.robot = robot_model
        self.calibration_data = {}

    def calibrate_joint_offsets(self):
        """
        Calibrate joint zero position offsets
        """
        print("Starting joint offset calibration...")

        offsets = {}
        for joint_name in self.robot.get_joint_names():
            print(f"Move {joint_name} to zero position and press enter...")
            input()  # Wait for user to position joint

            # Read current encoder value
            current_encoder = self.robot.get_joint_encoder(joint_name)
            offsets[joint_name] = current_encoder

            print(f"Recorded offset for {joint_name}: {current_encoder}")

        self.calibration_data['joint_offsets'] = offsets
        print("Joint offset calibration complete.")
        return offsets

    def tune_pid_parameters(self, joint_name, test_trajectory):
        """
        Automatically tune PID parameters for a joint
        """
        print(f"Tuning PID for {joint_name}...")

        # Try different PID values and evaluate performance
        best_params = None
        best_performance = float('inf')

        # Grid search over PID parameters
        kp_values = [1, 5, 10, 20]
        ki_values = [0.01, 0.1, 0.5, 1.0]
        kd_values = [0.1, 0.5, 1, 2]

        for kp in kp_values:
            for ki in ki_values:
                for kd in kd_values:
                    # Apply parameters
                    self.robot.set_joint_pid(joint_name, kp, ki, kd)

                    # Execute test trajectory
                    start_time = time.time()
                    log = self.execute_test_trajectory(test_trajectory)
                    execution_time = time.time() - start_time

                    # Evaluate performance (simplifed)
                    error = self.calculate_trajectory_error(log)
                    performance = error + 0.1 * execution_time  # Penalize slow execution

                    if performance < best_performance:
                        best_performance = performance
                        best_params = (kp, ki, kd)

        # Apply best parameters
        self.robot.set_joint_pid(joint_name, *best_params)
        print(f"Best parameters for {joint_name}: kp={best_params[0]}, ki={best_params[1]}, kd={best_params[2]}")

        return best_params

    def execute_test_trajectory(self, trajectory):
        """
        Execute a test trajectory and log performance
        """
        # Implementation would execute the trajectory and return logs
        pass

    def calculate_trajectory_error(self, log):
        """
        Calculate total error from execution log
        """
        total_error = 0
        for entry in log:
            total_error += np.sum(np.abs(entry['error']))
        return total_error
```

## Summary and Best Practices

When implementing basic movements in humanoid robots:

1. **Start Simple**: Begin with basic positions and movements before complex patterns
2. **Prioritize Safety**: Always implement safety checks and emergency procedures
3. **Test Gradually**: Increase complexity gradually and test each component
4. **Tune Parameters**: Properly tune control parameters for your specific robot
5. **Consider Real-Time Requirements**: Ensure your implementation meets timing constraints
6. **Implement Feedback Control**: Use sensor feedback to improve movement quality
7. **Plan for Error Recovery**: Have strategies for handling unexpected situations

The next chapter will explore more complex movements and how perception systems integrate with movement control.