---
title: Locomotion Systems
sidebar_position: 3
description: Locomotion principles and implementation for AI humanoid robotics
---

# Locomotion Systems

Locomotion is one of the most challenging aspects of humanoid robotics, requiring sophisticated control of multiple actuators, real-time balance management, and environmental adaptation. This chapter explores the principles and implementation of locomotion in humanoid robots.

## Introduction to Humanoid Locomotion

Humanoid locomotion aims to replicate human-like movement patterns while maintaining stability and efficiency. Key challenges include:
- Maintaining balance during dynamic movement
- Adapting to various terrains and obstacles
- Achieving energy-efficient movement
- Ensuring safety during walking and transitions

## Types of Locomotion

### Static Walking

Static walking maintains stability at all times, with the center of mass always within the support polygon.

**Characteristics:**
- Always statically stable
- Slower but safer
- Simpler control requirements
- Higher energy consumption

**Applications:**
- Unstable terrains
- Initial learning phase
- Safety-critical operations
- Educational robots

### Dynamic Walking

Dynamic walking allows the robot to be temporarily unstable, similar to human walking.

**Characteristics:**
- More human-like
- More energy efficient
- Faster movement
- Requires sophisticated control

**Key Concepts:**
- **Zero Moment Point (ZMP)**: Point where net moment of ground reaction forces is zero
- **Capture Point**: Point where robot needs to step to come to a stop
- **Limit Cycles**: Stable periodic motion patterns

```python
import numpy as np

class ZMPCalculator:
    def __init__(self, robot_mass, gravity=9.81):
        self.mass = robot_mass
        self.gravity = gravity

    def calculate_zmp(self, com_position, com_acceleration, com_height):
        """
        Calculate Zero Moment Point (ZMP) for walking control
        """
        # ZMP_x = CoM_x - (CoM_height / gravity) * CoM_acc_x
        # ZMP_y = CoM_y - (CoM_height / gravity) * CoM_acc_y

        zmp_x = com_position[0] - (com_height / self.gravity) * com_acceleration[0]
        zmp_y = com_position[1] - (com_height / self.gravity) * com_acceleration[1]

        return np.array([zmp_x, zmp_y, 0.0])

    def is_stable(self, zmp, support_polygon):
        """
        Check if ZMP is within support polygon
        """
        # Simple check for rectangular support polygon
        x_min, x_max = support_polygon[0]
        y_min, y_max = support_polygon[1]

        return x_min <= zmp[0] <= x_max and y_min <= zmp[1] <= y_max
```

### Bipedal Gait Patterns

**Double Support Phase:**
- Both feet on ground
- Transfer of weight between legs
- Stability during weight transfer

**Single Support Phase:**
- One foot on ground
- Swing leg moves forward
- Balance control critical

```python
class BipedalGait:
    def __init__(self, step_length=0.3, step_height=0.05, step_duration=1.0):
        self.step_length = step_length
        self.step_height = step_height
        self.step_duration = step_duration
        self.phase = 0  # 0: double support, 1: single support
        self.time_in_phase = 0

    def calculate_foot_trajectory(self, time, support_leg, swing_leg):
        """
        Calculate foot trajectory for walking
        """
        if self.phase == 0:  # Double support
            # Feet remain in place during weight transfer
            support_trajectory = self.get_support_foot_position(support_leg)
            swing_trajectory = self.get_swing_foot_position(swing_leg)
        else:  # Single support
            # Swing foot moves forward
            support_trajectory = self.get_support_foot_position(support_leg)
            swing_trajectory = self.calculate_swing_trajectory(time)

        return support_trajectory, swing_trajectory

    def calculate_swing_trajectory(self, time):
        """
        Calculate trajectory for swing foot
        """
        # Simplified parabolic trajectory
        phase_progress = (time % self.step_duration) / self.step_duration

        x = phase_progress * self.step_length
        z = self.step_height * np.sin(np.pi * phase_progress)  # Parabolic height
        y = 0  # No lateral movement in this simple model

        return np.array([x, y, z])
```

## Balance Control

### Center of Mass (CoM) Control

Maintaining the CoM within the support polygon is crucial for stability.

**CoM Control Strategies:**
- **Ankle Strategy**: Adjust foot orientation for small perturbations
- **Hip Strategy**: Move hips to control CoM for medium perturbations
- **Stepping Strategy**: Take a step to recover balance for large perturbations

```python
class BalanceController:
    def __init__(self, robot_mass, com_height):
        self.mass = robot_mass
        self.com_height = com_height
        self.gravity = 9.81
        self.com_position = np.array([0.0, 0.0, com_height])
        self.com_velocity = np.zeros(3)
        self.com_acceleration = np.zeros(3)

    def calculate_balance_correction(self, zmp_error, current_com_pos):
        """
        Calculate CoM correction to maintain balance
        """
        # Simple inverted pendulum model
        # Desired CoM acceleration to move ZMP to target
        desired_com_acc_x = -self.gravity / self.com_height * zmp_error[0]
        desired_com_acc_y = -self.gravity / self.com_height * zmp_error[1]

        return np.array([desired_com_acc_x, desired_com_acc_y, 0.0])

    def ankle_control(self, roll_error, pitch_error):
        """
        Ankle strategy for small balance corrections
        """
        # PID control for ankle angles
        kp_roll, ki_roll, kd_roll = 10, 0.1, 1
        kp_pitch, ki_pitch, kd_pitch = 15, 0.2, 1.5

        # Calculate desired ankle torques
        ankle_roll_torque = kp_roll * roll_error + ki_roll * 0 + kd_roll * 0
        ankle_pitch_torque = kp_pitch * pitch_error + ki_pitch * 0 + kd_pitch * 0

        return ankle_roll_torque, ankle_pitch_torque
```

### Capture Point Theory

The capture point is the point on the ground where a robot should step to come to a stop.

```python
class CapturePointController:
    def __init__(self, com_height, gravity=9.81):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)

    def calculate_capture_point(self, com_position, com_velocity):
        """
        Calculate capture point based on CoM state
        """
        capture_point_x = com_position[0] + com_velocity[0] / self.omega
        capture_point_y = com_position[1] + com_velocity[1] / self.omega

        return np.array([capture_point_x, capture_point_y, 0.0])

    def should_step(self, current_capture_point, foot_position, safety_margin=0.1):
        """
        Determine if a step is needed based on capture point
        """
        # Calculate distance from capture point to foot
        distance = np.linalg.norm(current_capture_point[:2] - foot_position[:2])

        # Step if capture point is outside safe area around foot
        return distance > safety_margin
```

## Walking Pattern Generation

### Inverted Pendulum Model

The linear inverted pendulum model (LIPM) is commonly used for walking pattern generation.

```python
class LIPMController:
    def __init__(self, com_height, gravity=9.81, sampling_time=0.01):
        self.com_height = com_height
        self.gravity = gravity
        self.omega = np.sqrt(gravity / com_height)
        self.sampling_time = sampling_time

    def calculate_com_trajectory(self, start_com, end_com, duration):
        """
        Calculate CoM trajectory using LIPM
        """
        num_points = int(duration / self.sampling_time)
        time_steps = np.linspace(0, duration, num_points)

        trajectories = []
        for t in time_steps:
            # LIPM solution: CoM follows exponential trajectory
            x = (start_com[0] - end_com[0]) * np.exp(-self.omega * t) + end_com[0]
            y = (start_com[1] - end_com[1]) * np.exp(-self.omega * t) + end_com[1]
            z = self.com_height  # Constant height

            trajectories.append(np.array([x, y, z]))

        return np.array(trajectories)

    def generate_footsteps(self, walking_distance, step_length=0.3):
        """
        Generate footstep pattern for walking
        """
        num_steps = int(walking_distance / step_length)
        footsteps = []

        for i in range(num_steps):
            # Alternate between left and right foot
            x = (i + 1) * step_length
            y = (-1) ** i * 0.1  # Alternate lateral position
            z = 0  # Ground level

            footsteps.append(np.array([x, y, z]))

        return np.array(footsteps)
```

### Walking Controllers

**Cart-Table Model:**
- Simplified model for walking control
- Decouples lateral and forward movement
- Computationally efficient

**Preview Control:**
- Uses future reference trajectory
- Improves tracking performance
- Requires future step planning

```python
class WalkingController:
    def __init__(self, com_height, step_period=1.0, step_length=0.3):
        self.com_height = com_height
        self.step_period = step_period
        self.step_length = step_length
        self.gravity = 9.81
        self.omega = np.sqrt(self.gravity / self.com_height)

        # Walking pattern parameters
        self.current_time = 0
        self.step_count = 0
        self.support_foot = "left"

    def generate_walking_pattern(self, duration):
        """
        Generate complete walking pattern
        """
        patterns = []
        t = 0

        while t < duration:
            # Calculate CoM position for current time
            com_x = self.calculate_com_x(t)
            com_y = self.calculate_com_y(t)

            # Calculate ZMP based on CoM state
            zmp_x = self.calculate_zmp_x(com_x, t)
            zmp_y = self.calculate_zmp_y(com_y, t)

            # Determine support foot
            if t % (2 * self.step_period) < self.step_period:
                support_foot = "left"
            else:
                support_foot = "right"

            pattern = {
                'time': t,
                'com_position': np.array([com_x, com_y, self.com_height]),
                'zmp': np.array([zmp_x, zmp_y, 0.0]),
                'support_foot': support_foot
            }

            patterns.append(pattern)
            t += 0.01  # 100 Hz update rate

        return patterns

    def calculate_com_x(self, t):
        """
        Calculate CoM x position based on walking pattern
        """
        # Simple model: CoM moves forward with walking
        steps_completed = t // self.step_period
        phase_in_step = t % self.step_period

        # CoM moves forward with walking motion
        com_x = steps_completed * self.step_length

        # Add small oscillation for natural movement
        oscillation = 0.01 * np.sin(2 * np.pi * t / self.step_period)

        return com_x + oscillation

    def calculate_com_y(self, t):
        """
        Calculate CoM y position (lateral balance)
        """
        # Lateral CoM oscillates between feet
        phase = (t % (2 * self.step_period)) / (2 * self.step_period)

        # Move CoM laterally toward support foot
        if phase < 0.5:
            # Supporting left foot, CoM moves toward left
            return -0.05 + 0.1 * np.sin(np.pi * phase)
        else:
            # Supporting right foot, CoM moves toward right
            return 0.05 - 0.1 * np.sin(np.pi * (phase - 0.5))
```

## Terrain Adaptation

### Flat Ground Walking

Basic walking on level surfaces.

**Key Considerations:**
- Consistent step timing
- Stable ZMP tracking
- Smooth CoM motion
- Proper foot placement

### Rough Terrain Navigation

Adapting to uneven surfaces.

**Techniques:**
- **Foot placement optimization**: Adjust where to step
- **Ankle adaptation**: Adjust foot orientation
- **Gait modification**: Change walking pattern
- **Vision-based planning**: Pre-plan steps using vision

```python
class RoughTerrainController:
    def __init__(self, base_controller):
        self.base_controller = base_controller
        self.terrain_map = None

    def adapt_to_terrain(self, terrain_data, current_state):
        """
        Adapt walking pattern to terrain conditions
        """
        # Analyze terrain for obstacles and slopes
        obstacles = self.detect_obstacles(terrain_data)
        slopes = self.calculate_slopes(terrain_data)

        # Modify step pattern based on terrain
        modified_pattern = self.base_controller.generate_walking_pattern(10.0)  # 10 seconds

        for i, pattern in enumerate(modified_pattern):
            # Adjust foot height for obstacles
            if self.foot_will_hit_obstacle(pattern, obstacles):
                pattern['foot_height'] += 0.05  # Lift foot higher

            # Adjust foot angle for slopes
            slope_at_foot = self.get_slope_at_location(pattern['zmp'], slopes)
            pattern['foot_angle'] = self.calculate_ankle_angle(slope_at_foot)

        return modified_pattern

    def detect_obstacles(self, terrain_data):
        """
        Detect obstacles in terrain data
        """
        # Simplified obstacle detection
        obstacles = []
        for point in terrain_data:
            if point[2] > 0.05:  # Height threshold for obstacles
                obstacles.append(point)
        return obstacles

    def foot_will_hit_obstacle(self, pattern, obstacles):
        """
        Check if foot placement will hit an obstacle
        """
        # Check if planned foot position has obstacles
        for obstacle in obstacles:
            distance = np.linalg.norm(pattern['zmp'][:2] - obstacle[:2])
            if distance < 0.05:  # 5cm threshold
                return True
        return False
```

### Stair Climbing

Specialized locomotion for stairs.

**Approaches:**
- **Step-by-step**: Treat each step individually
- **Swing-over**: Swing leg over steps
- **Modified gait**: Adjust entire walking pattern

## Energy Efficiency

### Walking Energetics

Understanding energy consumption in walking.

**Energy Sources:**
- **Positive work**: Muscles/actuators doing work
- **Negative work**: Absorbing energy (braking)
- **Passive dynamics**: Gravity-assisted movement

### Efficient Walking Strategies

**Passive Dynamic Walking:**
- Minimize active control
- Use gravity and momentum
- Reduce energy consumption
- More human-like gait

**Optimal Control:**
- Minimize energy consumption
- Optimize actuator usage
- Consider multiple objectives
- Use advanced optimization techniques

```python
class EnergyEfficientController:
    def __init__(self, robot_mass, com_height):
        self.mass = robot_mass
        self.com_height = com_height
        self.gravity = 9.81
        self.minimize_energy = True

    def optimize_step_timing(self, walking_speed):
        """
        Optimize step timing for energy efficiency
        """
        # Optimal step frequency for given speed
        # Based on inverted pendulum dynamics
        optimal_frequency = np.sqrt(self.gravity / self.com_height) / (2 * np.pi)

        # Adjust for desired walking speed
        step_length = walking_speed / optimal_frequency

        # Energy-optimal step length is related to leg length
        leg_length = self.com_height  # Approximation
        energy_optimal_length = 0.4 * np.sqrt(2 * self.gravity * leg_length)

        # Choose step length that balances speed and efficiency
        actual_step_length = min(step_length, energy_optimal_length)

        return actual_step_length, optimal_frequency

    def calculate_energy_consumption(self, joint_trajectories, actuator_efficiencies):
        """
        Estimate energy consumption for a walking pattern
        """
        total_energy = 0

        for joint_name, trajectory in joint_trajectories.items():
            # Calculate energy for each joint
            velocities = np.diff(trajectory) / 0.01  # Assuming 100Hz control
            forces = self.estimate_joint_forces(velocities, trajectory)

            # Energy = sum of (force * velocity) / efficiency
            power = np.abs(forces * velocities)
            energy = np.sum(power) / actuator_efficiencies[joint_name]
            total_energy += energy

        return total_energy

    def estimate_joint_forces(self, velocities, positions):
        """
        Estimate forces required at joints
        """
        # Simplified model - in reality, this would involve
        # inverse dynamics calculations
        forces = np.zeros_like(velocities)

        # Add terms for acceleration, gravity, friction
        for i in range(len(velocities)):
            # Inverse dynamics would calculate this properly
            forces[i] = 0.1 * velocities[i] + 0.05  # Simplified model

        return forces
```

## Safety Considerations

### Fall Prevention

**Early Detection:**
- Monitor balance metrics
- Identify instability early
- Prepare recovery actions
- Alert systems if needed

**Recovery Strategies:**
- **Stepping**: Take corrective steps
- **Ankle adjustment**: Adjust foot orientation
- **Hip strategy**: Move body to recover balance
- **Arm movement**: Use arms for balance

### Safe Fall Strategies

When a fall is unavoidable, minimize injury.

**Techniques:**
- **Crouching**: Lower CoM to reduce impact
- **Spreading**: Spread impact over larger area
- **Rolling**: Convert linear impact to rolling
- **Energy absorption**: Use compliance to absorb impact

```python
class FallPreventionSystem:
    def __init__(self, balance_controller):
        self.balance_controller = balance_controller
        self.fall_threshold = 0.1  # ZMP outside support by 10cm
        self.recovery_time = 0.5  # 500ms to recover

    def detect_instability(self, zmp, support_polygon):
        """
        Detect potential fall situation
        """
        if not self.balance_controller.is_stable(zmp, support_polygon):
            # Calculate distance to stability
            center_support = np.mean(support_polygon, axis=1)
            distance_to_safe = np.linalg.norm(zmp[:2] - center_support)

            if distance_to_safe > self.fall_threshold:
                return True, distance_to_safe

        return False, 0

    def execute_recovery(self, current_state, time_to_fall):
        """
        Execute recovery action
        """
        if time_to_fall < self.recovery_time:
            # Prepare for fall - crouch and protect vital areas
            return self.prepare_for_fall(current_state)
        else:
            # Attempt balance recovery
            return self.attempt_balance_recovery(current_state)

    def prepare_for_fall(self, current_state):
        """
        Prepare robot for safe fall
        """
        # Move CoM lower by crouching
        # Move arms to protect head
        # Activate compliant control
        recovery_action = {
            'crouch': True,
            'protect_head': True,
            'compliant_control': True,
            'arm_position': 'protective'
        }
        return recovery_action
```

## Advanced Locomotion Techniques

### Whole-Body Control

Coordinating all joints for optimal locomotion.

**Approaches:**
- **Task-based control**: Define tasks for different body parts
- **Optimization-based**: Minimize cost functions
- **MPC**: Model predictive control for future planning
- **Learning-based**: Adapt through experience

### Learning-Based Locomotion

Using machine learning for walking control.

**Methods:**
- **Reinforcement Learning**: Learn optimal policies
- **Imitation Learning**: Learn from demonstrations
- **Neural Networks**: Learn complex control mappings
- **Evolutionary Methods**: Optimize through evolution

```python
class LearningBasedWalker:
    def __init__(self):
        # Neural network for locomotion policy
        self.policy_network = self.create_policy_network()
        self.experience_buffer = []
        self.learning_rate = 0.001

    def create_policy_network(self):
        """
        Create neural network for walking policy
        """
        import torch
        import torch.nn as nn

        class WalkingPolicy(nn.Module):
            def __init__(self, input_size, output_size):
                super(WalkingPolicy, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_size),
                    nn.Tanh()  # Output between -1 and 1
                )

            def forward(self, state):
                return self.network(state)

        return WalkingPolicy(20, 12)  # 20 state inputs, 12 joint outputs

    def get_action(self, state):
        """
        Get action from policy network
        """
        import torch

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.policy_network(state_tensor)
        return action.squeeze(0).detach().numpy()

    def update_policy(self, states, actions, rewards):
        """
        Update policy using collected experience
        """
        import torch
        import torch.optim as optim

        optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)

        # Compute loss (simplified - in practice, use proper RL algorithms)
        predicted_actions = self.policy_network(states_tensor)
        loss = torch.nn.functional.mse_loss(predicted_actions, actions_tensor)

        # Add reward-based term to encourage good behavior
        reward_loss = -torch.mean(rewards_tensor)
        total_loss = loss + 0.1 * reward_loss

        # Update network
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss.item()
```

## Control Architectures

### Hierarchical Control

Different levels of control working together.

**Levels:**
- **High Level**: Gait selection, step planning
- **Mid Level**: Balance control, ZMP tracking
- **Low Level**: Joint control, motor commands

### Feedback Control Loops

Multiple feedback loops for robust control.

**Loop Types:**
- **Joint level**: Position/velocity/torque control
- **Balance level**: CoM and ZMP control
- **Gait level**: Step timing and placement
- **Task level**: High-level behavior control

## Implementation Considerations

### Real-Time Requirements

Locomotion control has strict timing requirements.

**Timing Constraints:**
- **High-level planning**: 10-100ms
- **Balance control**: 1-10ms
- **Joint control**: 0.1-1ms
- **Sensor processing**: As fast as possible

### Hardware Limitations

Real robots have physical constraints.

**Considerations:**
- **Actuator limits**: Maximum torque and speed
- **Sensor noise**: Accuracy and reliability
- **Processing power**: Available computational resources
- **Power consumption**: Battery life constraints

## Case Studies

### Honda ASIMO
- Advanced ZMP-based walking
- Predictive control for balance
- Multiple walking patterns
- Stair climbing capability

### Boston Dynamics Atlas
- Dynamic running and jumping
- Complex terrain navigation
- Recovery from disturbances
- High-speed locomotion

### SoftBank NAO
- Simple but stable walking
- Adaptive to different surfaces
- Educational platform for walking
- Robust and safe operation

## Future Trends

### Bio-Inspired Locomotion

**Approaches:**
- Human-like muscle activation patterns
- Adaptive compliance control
- Learning from human walking
- Neural control mechanisms

### AI-Enhanced Locomotion

**Techniques:**
- Deep reinforcement learning
- Sim-to-real transfer learning
- Multi-task learning for different terrains
- Evolutionary optimization

### Advanced Materials

**Applications:**
- Artificial muscles for natural movement
- Variable stiffness actuators
- Energy-harvesting systems
- Self-healing structures

## Summary

Locomotion in humanoid robots is a complex, multi-disciplinary challenge that combines mechanical engineering, control theory, and real-time computing. Successful implementation requires understanding of balance control, dynamic modeling, and safety considerations. Modern approaches increasingly use learning-based methods to adapt to various terrains and conditions while maintaining energy efficiency and safety.

The next chapter will explore implementation aspects, including how to integrate the locomotion system with the AI components discussed in earlier chapters.