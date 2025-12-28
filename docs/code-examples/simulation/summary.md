---
title: Simulation Summary and Best Practices
sidebar_position: 7
description: Comprehensive summary of simulation techniques and best practices for AI humanoid robotics
---

# Simulation Summary and Best Practices

This comprehensive summary consolidates the key concepts, techniques, and best practices for AI humanoid robotics simulation covered throughout this section.

## Key Simulation Concepts

### Physics Simulation Fundamentals

```python
import pybullet as p
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import threading
import queue
import torch

class HumanoidSimulationSystem:
    """
    Complete humanoid simulation system with all optimization techniques
    """
    def __init__(self, config: Dict):
        """
        Initialize complete simulation system

        Args:
            config: Configuration dictionary containing all simulation parameters
        """
        self.config = config
        self.physics_client = p.connect(p.GUI if config.get('gui', True) else p.DIRECT)

        # Set optimized physics parameters
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / config.get('physics_frequency', 240),
            numSolverIterations=config.get('solver_iterations', 10),
            maxNumSolverIterations=config.get('max_solver_iterations', 100),
            numSubSteps=config.get('num_substeps', 1),
            enableConeFriction=True,
            deterministicOverlappingPairs=True,
            contactBreakingThreshold=0.001,
            contactSlop=0.001,
            enableFileCaching=False
        )

        # Initialize environment
        self._initialize_environment()

        # Initialize optimization components
        self.state_cache = {}
        self.transform_cache = {}
        self.collision_cache = {}
        self.trajectory_cache = {}

        # Initialize AI integration
        self.ai_controller = self._initialize_ai_controller()

        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.start_time = time.time()

    def _initialize_environment(self):
        """
        Initialize the simulation environment with all components
        """
        # Load ground plane
        p.setAdditionalSearchPath(p.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

        # Load robot
        robot_urdf = self.config.get('robot_urdf', "r2d2.urdf")  # Placeholder
        self.robot_id = p.loadURDF(
            robot_urdf,
            self.config.get('robot_start_pos', [0, 0, 1]),
            self.config.get('robot_start_orn', [0, 0, 0, 1])
        )

        # Get robot information
        self.joint_info = self._get_joint_info()
        self.link_info = self._get_link_info()

        # Initialize robot-specific parameters
        self.robot_params = {
            'mass': self._calculate_robot_mass(),
            'com_height': self._calculate_com_height(),
            'foot_size': [0.15, 0.1, 0.05],  # Size of feet for balance
            'max_velocity': 5.0  # Maximum joint velocity
        }

    def _get_joint_info(self) -> Dict:
        """
        Get comprehensive joint information
        """
        joint_info = {}
        for i in range(p.getNumJoints(self.robot_id)):
            joint_data = p.getJointInfo(self.robot_id, i)
            joint_name = joint_data[1].decode('utf-8')
            joint_type = joint_data[2]
            joint_lower_limit = joint_data[8]
            joint_upper_limit = joint_data[9]
            joint_max_force = joint_data[10]
            joint_max_velocity = joint_data[11]

            joint_info[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower_limit': joint_lower_limit,
                'upper_limit': joint_upper_limit,
                'max_force': joint_max_force,
                'max_velocity': joint_max_velocity,
                'parent_link': joint_data[16],  # Parent link index
                'axis': joint_data[13]  # Joint axis
            }

        return joint_info

    def _get_link_info(self) -> Dict:
        """
        Get comprehensive link information
        """
        link_info = {}
        for i in range(p.getNumJoints(self.robot_id)):
            link_state = p.getLinkState(self.robot_id, i)
            link_info[i] = {
                'position': link_state[0],
                'orientation': link_state[1],
                'local_inertial_pos': link_state[2],
                'local_inertial_orn': link_state[3],
                'world_position': link_state[4],
                'world_orientation': link_state[5],
                'linear_velocity': link_state[6],
                'angular_velocity': link_state[7]
            }

        return link_info

    def _calculate_robot_mass(self) -> float:
        """
        Calculate total mass of robot
        """
        total_mass = 0
        for i in range(p.getNumJoints(self.robot_id)):
            dynamics_info = p.getDynamicsInfo(self.robot_id, i)
            total_mass += dynamics_info[0]  # mass is first element

        # Add base mass if it has dynamics
        base_dynamics = p.getDynamicsInfo(self.robot_id, -1)
        total_mass += base_dynamics[0]

        return total_mass

    def _calculate_com_height(self) -> float:
        """
        Calculate approximate center of mass height
        """
        # Simplified calculation - in practice, use proper CoM calculation
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return base_pos[2]  # Z position as approximation

    def get_optimized_state(self, use_cache: bool = True) -> Dict:
        """
        Get robot state with optimizations

        Args:
            use_cache: Whether to use cached state if available

        Returns:
            Dictionary containing robot state
        """
        cache_key = f"robot_state_{self.robot_id}"
        current_time = time.time()

        # Check cache first if enabled
        if use_cache and cache_key in self.state_cache:
            cached_time, cached_state = self.state_cache[cache_key]
            cache_duration = current_time - cached_time

            # Use cache if it's recent enough (within 1ms)
            if cache_duration < 0.001:
                return cached_state

        # Calculate fresh state
        start_time = time.time()

        # Get base state efficiently
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)

        # Get joint states in bulk
        joint_indices = list(self.joint_info.keys())
        joint_states = p.getJointStates(self.robot_id, joint_indices)

        # Process joint states efficiently
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_forces = [state[2] for state in joint_states]

        # Get important link states (feet, hands, head)
        important_links = self.config.get('important_links', [8, 12, 14, 15])  # Example indices
        link_states = {}
        for link_idx in important_links:
            if link_idx < p.getNumJoints(self.robot_id):
                link_state = p.getLinkState(self.robot_id, link_idx)
                link_states[link_idx] = {
                    'world_position': link_state[4],
                    'world_orientation': link_state[5],
                    'linear_velocity': link_state[6],
                    'angular_velocity': link_state[7]
                }

        # Calculate derived metrics efficiently
        com_pos = self._calculate_center_of_mass()
        balance_metrics = self._calculate_balance_metrics(base_pos, base_orn)

        # Calculate computation time
        computation_time = time.time() - start_time

        # Create state dictionary
        state = {
            'base': {
                'position': base_pos,
                'orientation': base_orn,
                'linear_velocity': base_lin_vel,
                'angular_velocity': base_ang_vel
            },
            'joints': {
                joint_idx: {
                    'position': joint_states[i][0],
                    'velocity': joint_states[i][1],
                    'force': joint_states[i][2],
                    'applied_torque': joint_states[i][3]
                }
                for i, joint_idx in enumerate(joint_indices)
            },
            'links': link_states,
            'derived': {
                'com_position': com_pos,
                'balance_metrics': balance_metrics,
                'is_balanced': balance_metrics['is_balanced'],
                'torso_height': base_pos[2]
            },
            'metadata': {
                'timestamp': current_time,
                'computation_time_ms': computation_time * 1000,
                'cached': False
            }
        }

        # Cache the result
        self.state_cache[cache_key] = (current_time, state)

        return state

    def _calculate_center_of_mass(self) -> List[float]:
        """
        Calculate center of mass of the robot
        """
        total_mass = 0
        weighted_sum = np.zeros(3)

        # Calculate CoM contribution from each link
        for i in range(p.getNumJoints(self.robot_id)):
            link_state = p.getLinkState(self.robot_id, i)
            dynamics_info = p.getDynamicsInfo(self.robot_id, i)

            mass = dynamics_info[0]
            link_pos = np.array(link_state[0])

            total_mass += mass
            weighted_sum += link_pos * mass

        # Include base link
        base_dynamics = p.getDynamicsInfo(self.robot_id, -1)
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        base_mass = base_dynamics[0]

        total_mass += base_mass
        weighted_sum += np.array(base_pos) * base_mass

        if total_mass > 0:
            com = weighted_sum / total_mass
            return com.tolist()
        else:
            return [0, 0, 0]

    def _calculate_balance_metrics(self, base_pos: List[float], base_orn: List[float]) -> Dict:
        """
        Calculate balance-related metrics
        """
        # Convert quaternion to Euler angles
        euler = p.getEulerFromQuaternion(base_orn)
        roll, pitch, yaw = euler

        # Calculate Zero Moment Point (ZMP) approximation
        com_pos = self._calculate_center_of_mass()
        gravity = 9.81
        com_height = com_pos[2] - base_pos[2]  # Height above ground

        # ZMP approximation (simplified)
        zmp_x = com_pos[0] - (com_height / gravity) * base_ang_vel[1]  # Pitch acceleration effect
        zmp_y = com_pos[1] - (com_height / gravity) * base_ang_vel[0]  # Roll acceleration effect

        # Calculate support polygon (simplified - just feet positions)
        left_foot_pos = self._get_link_position_by_name('left_foot')
        right_foot_pos = self._get_link_position_by_name('right_foot')

        if left_foot_pos and right_foot_pos:
            # Calculate approximate support polygon center and area
            support_center_x = (left_foot_pos[0] + right_foot_pos[0]) / 2
            support_center_y = (left_foot_pos[1] + right_foot_pos[1]) / 2

            # Distance from CoM to support polygon center
            com_to_support_dist = np.sqrt(
                (com_pos[0] - support_center_x)**2 + (com_pos[1] - support_center_y)**2
            )

            # Check if CoM is within approximate support polygon
            support_radius = 0.15  # Approximate support area radius
            is_balanced = com_to_support_dist < support_radius
        else:
            # If no foot positions found, use base position
            support_center_x, support_center_y = base_pos[0], base_pos[1]
            com_to_support_dist = np.sqrt((com_pos[0] - support_center_x)**2 + (com_pos[1] - support_center_y)**2)
            is_balanced = com_to_support_dist < 0.2  # 20cm threshold

        return {
            'roll_angle': roll,
            'pitch_angle': pitch,
            'zmp_x': zmp_x,
            'zmp_y': zmp_y,
            'com_support_distance': com_to_support_dist,
            'is_balanced': is_balanced,
            'torso_height': base_pos[2]
        }

    def _get_link_position_by_name(self, link_name: str) -> Optional[List[float]]:
        """
        Get position of link by name (simplified implementation)
        """
        # In a real implementation, you'd have a mapping of link names to indices
        # For this example, we'll return None to use the fallback
        return None

    def apply_optimized_control(self, commands: Dict[int, float], control_mode: int = p.POSITION_CONTROL):
        """
        Apply optimized control commands to robot

        Args:
            commands: Dictionary mapping joint indices to target values
            control_mode: Control mode (POSITION_CONTROL, VELOCITY_CONTROL, TORQUE_CONTROL)
        """
        for joint_idx, target_value in commands.items():
            if joint_idx in self.joint_info:
                joint_limit_min = self.joint_info[joint_idx]['lower_limit']
                joint_limit_max = self.joint_info[joint_idx]['upper_limit']

                # Clamp target value to joint limits
                clamped_target = max(joint_limit_min, min(joint_limit_max, target_value))

                # Apply control with appropriate parameters
                if control_mode == p.POSITION_CONTROL:
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=control_mode,
                        targetPosition=clamped_target,
                        maxVelocity=self.joint_info[joint_idx]['max_velocity'],
                        force=self.joint_info[joint_idx]['max_force']
                    )
                elif control_mode == p.VELOCITY_CONTROL:
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=control_mode,
                        targetVelocity=clamped_target,
                        force=self.joint_info[joint_idx]['max_force']
                    )
                elif control_mode == p.TORQUE_CONTROL:
                    p.setJointMotorControl2(
                        bodyIndex=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=control_mode,
                        force=clamped_target
                    )

    def detect_collisions(self, robot_only: bool = False) -> List[Dict]:
        """
        Detect collisions with optimizations

        Args:
            robot_only: If True, only detect robot self-collisions

        Returns:
            List of collision information
        """
        if robot_only:
            # Detect self-collisions within robot
            collisions = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=self.robot_id,
                distance=0.05  # 5cm threshold for self-collision
            )
        else:
            # Detect collisions with environment
            collisions = p.getContactPoints(bodyA=self.robot_id)

        collision_info = []
        for contact in collisions:
            collision_detail = {
                'contact_flag': contact[0],
                'body_a_id': contact[1],
                'body_b_id': contact[2],
                'link_index_a': contact[3],
                'link_index_b': contact[4],
                'position_on_a': contact[5],
                'position_on_b': contact[6],
                'contact_normal_on_b': contact[7],
                'contact_distance': contact[8],
                'normal_force': contact[9],
                'lateral_friction_force1': contact[10],
                'lateral_friction_force2': contact[11]
            }
            collision_info.append(collision_detail)

        return collision_info

    def get_performance_metrics(self) -> Dict:
        """
        Get comprehensive performance metrics
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Get physics engine metrics
        physics_metrics = p.getPhysicsEngineParameters()

        # Calculate simulation speed
        if hasattr(self, 'step_count') and self.step_count > 0:
            simulation_time = self.step_count / self.config.get('physics_frequency', 240)
            real_time_speed = simulation_time / elapsed_time if elapsed_time > 0 else 0
        else:
            real_time_speed = 0

        return {
            'elapsed_time': elapsed_time,
            'simulation_speed_ratio': real_time_speed,
            'physics_frequency': self.config.get('physics_frequency', 240),
            'solver_iterations': physics_metrics.get('numSolverIterations', 10),
            'current_step': getattr(self, 'step_count', 0),
            'average_state_computation_time': np.mean(self.performance_monitor.state_computation_times) if self.performance_monitor.state_computation_times else 0,
            'memory_usage_mb': self.performance_monitor.get_memory_usage(),
            'cpu_usage_percent': self.performance_monitor.get_cpu_usage()
        }

    def cleanup(self):
        """
        Clean up simulation resources
        """
        p.disconnect(self.physics_client)
        print("Simulation system cleaned up")

class PerformanceMonitor:
    """
    Monitor and optimize simulation performance
    """
    def __init__(self):
        self.state_computation_times = deque(maxlen=1000)
        self.physics_step_times = deque(maxlen=1000)
        self.total_step_times = deque(maxlen=1000)
        self.frame_count = 0
        self.start_time = time.time()

    def record_state_computation_time(self, computation_time: float):
        """
        Record state computation time for performance monitoring
        """
        self.state_computation_times.append(computation_time)

    def record_physics_step_time(self, step_time: float):
        """
        Record physics step time for performance monitoring
        """
        self.physics_step_times.append(step_time)

    def record_total_step_time(self, total_time: float):
        """
        Record total simulation step time
        """
        self.total_step_times.append(total_time)

    def get_performance_summary(self) -> Dict:
        """
        Get performance summary
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        if not self.total_step_times:
            return {'error': 'No performance data collected'}

        return {
            'frames_per_second': len(self.total_step_times) / elapsed_time if elapsed_time > 0 else 0,
            'average_total_time_ms': np.mean(self.total_step_times) * 1000,
            'average_state_time_ms': np.mean(self.state_computation_times) * 1000 if self.state_computation_times else 0,
            'average_physics_time_ms': np.mean(self.physics_step_times) * 1000 if self.physics_step_times else 0,
            'max_total_time_ms': max(self.total_step_times) * 1000,
            'min_total_time_ms': min(self.total_step_times) * 1000,
            'std_total_time_ms': np.std(self.total_step_times) * 1000,
            'elapsed_real_time': elapsed_time
        }

    def get_cpu_usage(self) -> float:
        """
        Get current CPU usage percentage
        """
        return psutil.cpu_percent()

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB
        """
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

class AIController:
    """
    AI controller for humanoid robot
    """
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        """
        Initialize AI controller

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Initialize neural networks
        self.policy_network = self._create_policy_network()
        self.value_network = self._create_value_network()

        # Initialize training components
        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            lr=config.get('learning_rate', 3e-4)
        )
        self.replay_buffer = deque(maxlen=config.get('replay_buffer_size', 100000))

        # Initialize exploration parameters
        self.exploration_noise = config.get('initial_exploration_noise', 0.1)
        self.exploration_decay = config.get('exploration_decay', 0.999)

    def _create_policy_network(self) -> torch.nn.Module:
        """
        Create policy network for action selection
        """
        class PolicyNetwork(torch.nn.Module):
            def __init__(self, input_size, output_size, hidden_size=256):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, output_size),
                    torch.nn.Tanh()  # Actions bounded to [-1, 1]
                )

            def forward(self, x):
                return self.network(x)

        return PolicyNetwork(self.state_dim, self.action_dim)

    def _create_value_network(self) -> torch.nn.Module:
        """
        Create value network for state evaluation
        """
        class ValueNetwork(torch.nn.Module):
            def __init__(self, input_size, hidden_size=256):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, 1)  # Single value output
                )

            def forward(self, x):
                return self.network(x)

        return ValueNetwork(self.state_dim)

    def get_action(self, state: np.ndarray, add_exploration: bool = True) -> np.ndarray:
        """
        Get action from policy network

        Args:
            state: Current state
            add_exploration: Whether to add exploration noise

        Returns:
            Action array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.policy_network(state_tensor).squeeze(0).numpy()

        if add_exploration and self.exploration_noise > 0.01:
            # Add exploration noise
            noise = np.random.normal(0, self.exploration_noise, action.shape)
            action = np.clip(action + noise, -1, 1)

            # Decay exploration
            self.exploration_noise *= self.exploration_decay

        return action

    def update_policy(self, batch_size: int = 64):
        """
        Update policy using collected experiences

        Args:
            batch_size: Size of training batch
        """
        if len(self.replay_buffer) < batch_size:
            return  # Not enough experiences to train

        # Sample batch from replay buffer
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch_indices]

        # Extract batch data
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch]).unsqueeze(1)

        # Compute target values for value network
        with torch.no_grad():
            next_values = self.value_network(next_states)
            target_values = rewards + (0.99 * next_values * ~dones)  # 0.99 is gamma

        # Update value network
        current_values = self.value_network(states)
        value_loss = torch.nn.MSELoss()(current_values, target_values)

        # Update policy network using advantage
        advantages = target_values - current_values.detach()
        log_probs = torch.log(torch.clamp(self.policy_network(states), 1e-8, 1-1e-8))
        action_log_probs = torch.sum(log_probs * actions, dim=1, keepdim=True)
        policy_loss = -(action_log_probs * advantages.detach()).mean()

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        # Backpropagate
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_network.parameters()) + list(self.value_network.parameters()),
            40
        )
        self.optimizer.step()

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = {
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done
        }
        self.replay_buffer.append(experience)

def simulation_system_example():
    """
    Example of complete simulation system
    """
    print("=== Complete Simulation System Example ===")

    # Configuration
    config = {
        'physics_frequency': 480,  # Hz
        'control_frequency': 100,  # Hz
        'render_frequency': 30,    # Hz
        'gui': True,
        'robot_urdf': 'humanoid.urdf',  # Placeholder
        'robot_start_pos': [0, 0, 1],
        'solver_iterations': 20,
        'max_solver_iterations': 200,
        'num_substeps': 2,
        'learning_rate': 3e-4,
        'replay_buffer_size': 50000,
        'initial_exploration_noise': 0.1,
        'exploration_decay': 0.9995,
        'important_links': [8, 12, 14, 15]  # Example: feet, hands, head
    }

    # Create simulation system
    sim_system = HumanoidSimulationSystem(config)

    print("Simulation system initialized with:")
    print(f"  - Physics frequency: {config['physics_frequency']} Hz")
    print(f"  - Control frequency: {config['control_frequency']} Hz")
    print(f"  - Robot mass: {sim_system.robot_params['mass']:.2f} kg")
    print(f"  - CoM height: {sim_system.robot_params['com_height']:.2f} m")

    # Run simulation loop
    print(f"\nRunning simulation loop for 5 seconds...")
    start_time = time.time()
    step_count = 0

    while time.time() - start_time < 5.0:
        step_start = time.time()

        # Get optimized state
        state = sim_system.get_optimized_state()

        # Apply simple control (for demonstration)
        # In practice, this would come from AI controller
        simple_commands = {}
        for joint_idx in list(sim_system.joint_info.keys())[:5]:  # First 5 joints
            # Simple oscillating pattern
            command = 0.5 * np.sin(time.time() * 2 + joint_idx * 0.5)
            simple_commands[joint_idx] = command

        # Apply control
        sim_system.apply_optimized_control(simple_commands)

        # Step simulation
        p.stepSimulation()

        # Record performance
        step_time = time.time() - step_start
        sim_system.performance_monitor.record_total_step_time(step_time)
        sim_system.performance_monitor.record_state_computation_time(state['metadata']['computation_time_ms'] / 1000)

        step_count += 1

        # Print status periodically
        if step_count % 240 == 0:  # Every second at 240Hz
            perf_metrics = sim_system.get_performance_metrics()
            print(f"Time: {time.time() - start_time:.1f}s, "
                  f"Real-time speed: {perf_metrics['simulation_speed_ratio']:.2f}x, "
                  f"State comp time: {state['metadata']['computation_time_ms']:.2f}ms")

    # Get final performance metrics
    final_metrics = sim_system.performance_monitor.get_performance_summary()
    print(f"\nFinal Performance Metrics:")
    print(f"  Average step time: {final_metrics['average_total_time_ms']:.3f}ms")
    print(f"  Average state computation: {final_metrics['average_state_time_ms']:.3f}ms")
    print(f"  Simulation speed: {final_metrics['frames_per_second']:.1f} FPS")
    print(f"  Real-time speed: {final_metrics['simulation_speed_ratio']:.2f}x")

    # Clean up
    sim_system.cleanup()
    print(f"\nComplete simulation system example completed after {step_count} steps")

if __name__ == "__main__":
    simulation_system_example()
```

## Advanced Simulation Techniques Summary

### Key Optimization Strategies

1. **Physics Optimization**:
   - Fixed time steps for determinism
   - Appropriate solver iterations
   - Contact parameter tuning
   - Sub-stepping for stability

2. **State Caching**:
   - Cache frequently accessed data
   - Invalidate cache based on time or events
   - Use memory pools for frequent allocations

3. **Batch Processing**:
   - Process multiple operations together
   - Reduce function call overhead
   - Use vectorized operations

4. **GPU Acceleration**:
   - Offload computation-intensive tasks
   - Use parallel processing for sensor data
   - Accelerate neural network inference

5. **Parallel Simulation**:
   - Run multiple simulations in parallel
   - Use multi-processing for independent tasks
   - Coordinate between simulations when needed

### Best Practices

1. **Performance Monitoring**:
   - Continuously monitor simulation performance
   - Identify bottlenecks systematically
   - Optimize based on actual measurements

2. **Safety First**:
   - Implement safety checks and limits
   - Use collision detection and avoidance
   - Include emergency stop procedures

3. **Realism vs. Performance**:
   - Balance simulation fidelity with performance
   - Use appropriate level of detail for task
   - Optimize for specific use case requirements

4. **Modular Design**:
   - Separate physics from graphics
   - Use plugin architecture for components
   - Enable easy configuration and extension

5. **Resource Management**:
   - Efficient memory usage
   - Proper cleanup of resources
   - Monitor system resource usage

## AI Integration Techniques

### Reinforcement Learning Integration

```python
class RLSimulationEnvironment:
    """
    Simulation environment for reinforcement learning
    """
    def __init__(self, simulation_system):
        self.sim = simulation_system
        self.state_size = self._calculate_state_size()
        self.action_size = self._calculate_action_size()
        self.max_episode_steps = 1000
        self.current_step = 0
        self.episode_reward = 0.0

    def _calculate_state_size(self) -> int:
        """
        Calculate size of state vector
        """
        # Example: joint positions + velocities + IMU data
        num_joints = len(self.sim.joint_info)
        state_size = (
            num_joints * 2 +  # Positions and velocities
            3 + 3 +           # IMU linear and angular
            3                 # Base position
        )
        return state_size

    def _calculate_action_size(self) -> int:
        """
        Calculate size of action vector
        """
        # One action per controllable joint
        controllable_joints = [idx for idx, info in self.sim.joint_info.items()
                              if info['type'] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]
        return len(controllable_joints)

    def reset(self) -> np.ndarray:
        """
        Reset environment and return initial state
        """
        # Reset simulation to initial state
        self.sim.reset_simulation()

        # Reset episode counters
        self.current_step = 0
        self.episode_reward = 0.0

        # Get initial state
        state = self._get_state_vector()
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment

        Args:
            action: Action vector from agent

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Apply action to simulation
        self._apply_action(action)

        # Step simulation
        p.stepSimulation()

        # Get next state
        next_state = self._get_state_vector()

        # Calculate reward
        reward = self._calculate_reward()

        # Update episode counters
        self.current_step += 1
        self.episode_reward += reward

        # Check if episode is done
        done = self.current_step >= self.max_episode_steps or self._is_terminal_state()

        # Get additional info
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'is_balanced': self._is_robot_balanced(),
            'torso_height': self._get_torso_height()
        }

        return next_state, reward, done, info

    def _get_state_vector(self) -> np.ndarray:
        """
        Get state as vector for neural network
        """
        state = self.sim.get_optimized_state()

        # Extract relevant information and convert to vector
        joint_positions = [state['joints'][idx]['position']
                          for idx in sorted(state['joints'].keys())]
        joint_velocities = [state['joints'][idx]['velocity']
                           for idx in sorted(state['joints'].keys())]

        base_pos = state['base']['position']
        base_orn = state['base']['orientation']
        base_lin_vel = state['base']['linear_velocity']
        base_ang_vel = state['base']['angular_velocity']

        # Convert quaternion to Euler angles for easier learning
        euler_angles = p.getEulerFromQuaternion(base_orn)

        # Combine into state vector
        state_vector = np.concatenate([
            joint_positions,
            joint_velocities,
            base_pos,
            euler_angles,
            base_lin_vel,
            base_ang_vel,
            [state['derived']['balance_metrics']['com_support_distance']],
            [state['derived']['balance_metrics']['is_balanced']]
        ])

        return state_vector

    def _apply_action(self, action: np.ndarray):
        """
        Apply action vector to simulation
        """
        # Map action vector to joint commands
        controllable_joints = [idx for idx, info in self.sim.joint_info.items()
                              if info['type'] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]]

        commands = {}
        for i, joint_idx in enumerate(controllable_joints):
            if i < len(action):
                # Scale action to reasonable joint limits
                scaled_action = np.clip(action[i], -1, 1)  # Actions are in [-1, 1]
                target_position = scaled_action * 1.57  # Scale to reasonable joint range
                commands[joint_idx] = target_position

        # Apply commands
        self.sim.apply_optimized_control(commands, control_mode=p.POSITION_CONTROL)

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current state
        """
        state = self.sim.get_optimized_state()

        reward = 0.0

        # Positive reward for maintaining balance
        balance_metrics = state['derived']['balance_metrics']
        if balance_metrics['is_balanced']:
            reward += 0.1

        # Positive reward for forward progress (if walking task)
        # This would depend on the specific task
        base_pos = state['base']['position']
        if hasattr(self, 'previous_base_x'):
            forward_progress = base_pos[0] - self.previous_base_x
            reward += max(0, forward_progress * 10)  # 10 points per meter forward
            self.previous_base_x = base_pos[0]
        else:
            self.previous_base_x = base_pos[0]

        # Negative reward for falling
        if not balance_metrics['is_balanced'] or base_pos[2] < 0.3:  # Fell down
            reward -= 1.0

        # Negative reward for large control efforts
        # This encourages energy efficiency
        reward -= 0.001 * np.sum(np.abs(action))  # Action is available from step method

        # Small negative reward for each time step (to encourage efficiency)
        reward -= 0.01

        return reward

    def _is_terminal_state(self) -> bool:
        """
        Check if current state is terminal
        """
        state = self.sim.get_optimized_state()

        # Check if robot has fallen
        base_pos = state['base']['position']
        if base_pos[2] < 0.3:  # Robot is on the ground
            return True

        # Check if robot is severely unbalanced
        balance_metrics = state['derived']['balance_metrics']
        if not balance_metrics['is_balanced'] and abs(balance_metrics['pitch_angle']) > 1.0:
            return True

        return False

    def _is_robot_balanced(self) -> bool:
        """
        Check if robot is in balanced state
        """
        state = self.sim.get_optimized_state()
        return state['derived']['balance_metrics']['is_balanced']

    def _get_torso_height(self) -> float:
        """
        Get current torso height
        """
        state = self.sim.get_optimized_state()
        return state['base']['position'][2]

class SimulationTrainer:
    """
    Training system for humanoid robot control
    """
    def __init__(self, env: RLSimulationEnvironment, config: Dict):
        self.env = env
        self.config = config
        self.agent = self._initialize_agent()
        self.episode_rewards = []
        self.episode_lengths = []

    def _initialize_agent(self):
        """
        Initialize reinforcement learning agent
        """
        # This would typically be a PPO, SAC, or DDPG agent
        # For this example, we'll use a placeholder
        return PPOAgent(
            state_dim=self.env.state_size,
            action_dim=self.env.action_size,
            config=self.config
        )

    def train(self, total_timesteps: int = 1000000):
        """
        Train the agent

        Args:
            total_timesteps: Total number of training timesteps
        """
        print(f"Starting training for {total_timesteps} timesteps...")

        obs = self.env.reset()
        ep_reward = 0
        ep_length = 0
        total_steps = 0

        while total_steps < total_timesteps:
            # Get action from agent
            action, _ = self.agent.predict(obs)

            # Take step in environment
            new_obs, reward, done, info = self.env.step(action)

            # Store experience
            self.agent.store_experience(obs, action, reward, new_obs, done)

            # Update counters
            obs = new_obs
            ep_reward += reward
            ep_length += 1
            total_steps += 1

            if done:
                # Episode finished
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)

                # Print progress periodically
                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_length = np.mean(self.episode_lengths[-10:])
                    print(f"Episode {len(self.episode_rewards)}, "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Avg Length: {avg_length:.1f}, "
                          f"Total Steps: {total_steps}")

                # Reset environment
                obs = self.env.reset()
                ep_reward = 0
                ep_length = 0

            # Update agent periodically
            if total_steps % self.config.get('update_frequency', 1000) == 0:
                self.agent.update()

        print("Training completed!")

    def evaluate_agent(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate trained agent

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        success_rates = []

        for episode in range(num_episodes):
            obs = self.env.reset()
            ep_reward = 0
            ep_length = 0
            success = False

            for step in range(self.env.max_episode_steps):
                action, _ = self.agent.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)

                ep_reward += reward
                ep_length += 1

                if done:
                    # Check if episode was successful (based on task-specific criteria)
                    success = info.get('episode_success', False) or ep_length == self.env.max_episode_steps
                    break

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            success_rates.append(success)

        return {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'success_rate': np.mean(success_rates),
            'all_rewards': episode_rewards
        }

def rl_training_example():
    """
    Example of reinforcement learning with simulation
    """
    print("=== RL Training with Simulation Example ===")

    # Configuration for training
    config = {
        'learning_rate': 3e-4,
        'batch_size': 64,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'n_epochs': 10,
        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
        'update_frequency': 2048  # Steps per update
    }

    # Create simulation environment
    sim_config = {
        'physics_frequency': 240,
        'control_frequency': 50,
        'gui': False,  # Headless for training
        'robot_urdf': 'humanoid.urdf',
        'robot_start_pos': [0, 0, 1]
    }

    sim_system = HumanoidSimulationSystem(sim_config)
    rl_env = RLSimulationEnvironment(sim_system)

    # Create trainer
    trainer = SimulationTrainer(rl_env, config)

    print(f"Environment created:")
    print(f"  - State size: {rl_env.state_size}")
    print(f"  - Action size: {rl_env.action_size}")
    print(f"  - Max episode steps: {rl_env.max_episode_steps}")

    # Train agent (for demonstration, we'll train for fewer steps)
    print(f"\nStarting training...")
    trainer.train(total_timesteps=10000)  # Reduced for demonstration

    # Evaluate trained agent
    print(f"\nEvaluating trained agent...")
    eval_results = trainer.evaluate_agent(num_episodes=5)
    print(f"Evaluation results:")
    print(f"  - Average reward: {eval_results['avg_reward']:.2f}")
    print(f"  - Success rate: {eval_results['success_rate']:.2f}")
    print(f"  - Average episode length: {eval_results['avg_length']:.1f}")

    # Clean up
    sim_system.cleanup()
    print(f"\nRL training example completed!")

if __name__ == "__main__":
    rl_training_example()
```

## Deployment Considerations

### Simulation-to-Real Transfer

```python
class DomainRandomization:
    """
    Domain randomization for sim-to-real transfer
    """
    def __init__(self, simulation_system):
        self.sim = simulation_system
        self.randomization_params = {
            'mass_variance': (0.8, 1.2),      # 80% to 120% of original mass
            'friction_variance': (0.5, 2.0),  # 50% to 200% of original friction
            'gravity_variance': (0.9, 1.1),   # 90% to 110% of original gravity
            'actuator_variance': (0.95, 1.05), # 95% to 105% of nominal
            'sensor_noise': (0.0, 0.1),       # Add sensor noise
            'visual_randomization': True       # Randomize visual appearance
        }

    def randomize_domain(self):
        """
        Apply domain randomization to simulation
        """
        # Randomize physical parameters
        self._randomize_masses()
        self._randomize_friction()
        self._randomize_gravity()
        self._randomize_actuators()
        self._add_sensor_noise()
        self._randomize_visuals()

    def _randomize_masses(self):
        """
        Randomize masses of robot links
        """
        for joint_idx in self.sim.joint_info:
            # Get original mass
            dynamics_info = p.getDynamicsInfo(self.sim.robot_id, joint_idx)
            original_mass = dynamics_info[0]

            # Apply randomization
            mass_multiplier = np.random.uniform(
                self.randomization_params['mass_variance'][0],
                self.randomization_params['mass_variance'][1]
            )
            new_mass = original_mass * mass_multiplier

            # Set new mass (this may not work on all PyBullet versions)
            try:
                p.changeDynamics(
                    self.sim.robot_id,
                    joint_idx,
                    mass=new_mass
                )
            except:
                # If mass change fails, continue
                pass

    def _randomize_friction(self):
        """
        Randomize friction coefficients
        """
        for joint_idx in self.sim.joint_info:
            # Get original friction
            dynamics_info = p.getDynamicsInfo(self.sim.robot_id, joint_idx)
            original_lateral_friction = dynamics_info[1]

            # Apply randomization
            friction_multiplier = np.random.uniform(
                self.randomization_params['friction_variance'][0],
                self.randomization_params['friction_variance'][1]
            )
            new_friction = original_lateral_friction * friction_multiplier

            p.changeDynamics(
                self.sim.robot_id,
                joint_idx,
                lateralFriction=new_friction
            )

    def _randomize_gravity(self):
        """
        Randomize gravity
        """
        base_gravity = 9.81
        gravity_multiplier = np.random.uniform(
            self.randomization_params['gravity_variance'][0],
            self.randomization_params['gravity_variance'][1]
        )
        new_gravity = -base_gravity * gravity_multiplier

        p.setGravity(0, 0, new_gravity)

    def _randomize_actuators(self):
        """
        Randomize actuator parameters
        """
        # This would involve changing control parameters
        # For this example, we'll just note that this would happen
        pass

    def _add_sensor_noise(self):
        """
        Add noise to sensor readings
        """
        # This would be implemented in the sensor reading functions
        # The noise level would be randomly selected from the range
        pass

    def _randomize_visuals(self):
        """
        Randomize visual appearance
        """
        if self.randomization_params['visual_randomization']:
            # Randomize colors and textures
            for joint_idx in range(p.getNumJoints(self.sim.robot_id)):
                color = [
                    np.random.uniform(0.1, 1.0),
                    np.random.uniform(0.1, 1.0),
                    np.random.uniform(0.1, 1.0),
                    1.0
                ]
                try:
                    p.changeVisualShape(self.sim.robot_id, joint_idx, rgbaColor=color)
                except:
                    # If visual change fails, continue
                    pass

class CurriculumLearning:
    """
    Curriculum learning for progressive difficulty
    """
    def __init__(self):
        self.current_stage = 0
        self.stage_thresholds = [200, 500, 800, 1000]  # Reward thresholds
        self.stage_descriptions = [
            "Basic standing",
            "Simple walking",
            "Walking with obstacles",
            "Complex locomotion"
        ]

    def update_curriculum(self, current_reward: float) -> bool:
        """
        Update curriculum based on performance

        Args:
            current_reward: Current episode reward

        Returns:
            True if curriculum stage changed
        """
        if (self.current_stage < len(self.stage_thresholds) and
            current_reward > self.stage_thresholds[self.current_stage]):
            self.current_stage += 1
            print(f"Advancing to curriculum stage {self.current_stage}: {self.stage_descriptions[self.current_stage]}")
            return True

        return False

    def get_current_task(self) -> str:
        """
        Get current task based on curriculum stage
        """
        return self.stage_descriptions[min(self.current_stage, len(self.stage_descriptions) - 1)]

def simulation_deployment_example():
    """
    Example of simulation deployment considerations
    """
    print("=== Simulation Deployment Considerations ===")

    # Example: Domain randomization
    print("\n1. Domain Randomization Example:")
    sim_config = {
        'physics_frequency': 240,
        'gui': False,
        'robot_urdf': 'humanoid.urdf'
    }
    sim_system = HumanoidSimulationSystem(sim_config)
    domain_rand = DomainRandomization(sim_system)

    print("  Applying domain randomization...")
    domain_rand.randomize_domain()
    print("  Domain randomization applied successfully")

    # Example: Curriculum learning
    print("\n2. Curriculum Learning Example:")
    curriculum = CurriculumLearning()

    # Simulate some training progress
    rewards = [100, 250, 400, 600, 850, 950]  # Simulated rewards
    for i, reward in enumerate(rewards):
        stage_changed = curriculum.update_curriculum(reward)
        if stage_changed:
            current_task = curriculum.get_current_task()
            print(f"    New task: {current_task}")

    # Example: Performance optimization for deployment
    print("\n3. Performance Optimization for Deployment:")
    perf_monitor = PerformanceMonitor()

    # Simulate performance monitoring
    for i in range(100):
        # Simulate some work
        time.sleep(0.001)  # 1ms of work
        perf_monitor.record_total_step_time(0.001)

    perf_summary = perf_monitor.get_performance_summary()
    print(f"  Average step time: {perf_summary['average_total_time_ms']:.3f}ms")
    print(f"  Estimated real-time speed: {1000/perf_summary['average_total_time_ms']:.1f}x")

    # Clean up
    sim_system.cleanup()
    print("\nSimulation deployment example completed!")

if __name__ == "__main__":
    simulation_deployment_example()
```

## Summary and Best Practices

This comprehensive simulation framework provides:

1. **High-Performance Physics**: Optimized parameters and caching
2. **Realistic Sensor Models**: Noise, latency, and environmental effects
3. **AI Integration**: Direct connection to reinforcement learning
4. **Scalability**: Parallel processing and distributed simulation
5. **Safety**: Built-in safety checks and limits
6. **Transfer Learning**: Domain randomization and curriculum learning

### Key Takeaways

- **Optimize Early**: Performance optimization should be considered from the start
- **Measure Everything**: Continuously monitor performance metrics
- **Balance Fidelity and Speed**: Adjust simulation detail based on requirements
- **Plan for Deployment**: Consider sim-to-real transfer from the beginning
- **Safety First**: Implement comprehensive safety systems
- **Modular Design**: Create flexible, extensible systems

These simulation techniques form the foundation for developing sophisticated AI-powered humanoid robots that can learn and adapt in safe, controlled environments before deployment in the real world.