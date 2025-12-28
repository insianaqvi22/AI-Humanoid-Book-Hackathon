---
title: Advanced Simulation Techniques and Best Practices
sidebar_position: 5
description: Advanced simulation techniques and best practices for AI humanoid robotics
---

# Advanced Simulation Techniques and Best Practices

This section covers advanced simulation techniques that enable sophisticated AI development for humanoid robots, including performance optimization, multi-robot simulation, and integration with machine learning pipelines.

## High-Performance Simulation

### Parallel Simulation Execution

Running multiple simulation instances in parallel to accelerate training:

```python
import multiprocessing as mp
import threading
import queue
import time
import numpy as np
from typing import Dict, List, Callable, Any
import pickle
import zmq

class ParallelSimulationManager:
    """
    Manages multiple parallel simulation instances for accelerated training
    """
    def __init__(self, num_simulations: int = 4, simulation_class: type = None, config: Dict = None):
        """
        Initialize parallel simulation manager

        Args:
            num_simulations: Number of parallel simulation instances
            simulation_class: Class to instantiate for each simulation
            config: Configuration for simulations
        """
        self.num_simulations = num_simulations
        self.simulation_class = simulation_class
        self.config = config or {}
        self.processes = []
        self.result_queues = []
        self.command_queues = []
        self.running = False

        # Communication setup
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://*:5555")

        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect("tcp://localhost:5556")
        self.subscriber.setsockopt(zmq.SUBSCRIBE, b"results")

    def start_simulations(self):
        """
        Start all parallel simulation processes
        """
        for i in range(self.num_simulations):
            result_queue = mp.Queue()
            command_queue = mp.Queue()

            process = mp.Process(
                target=self._simulation_worker,
                args=(i, result_queue, command_queue)
            )
            process.start()

            self.processes.append(process)
            self.result_queues.append(result_queue)
            self.command_queues.append(command_queue)

        self.running = True
        print(f"Started {self.num_simulations} parallel simulations")

    def _simulation_worker(self, worker_id: int, result_queue: mp.Queue, command_queue: mp.Queue):
        """
        Worker function for each simulation process
        """
        # Create simulation instance
        simulation = self.simulation_class(worker_id, self.config)

        try:
            while True:
                # Check for commands
                try:
                    command = command_queue.get_nowait()
                    if command == "STOP":
                        break
                    elif isinstance(command, dict) and command.get('type') == 'STEP':
                        # Execute simulation step
                        result = simulation.step(command.get('data', {}))
                        result_queue.put({
                            'worker_id': worker_id,
                            'result': result,
                            'timestamp': time.time()
                        })
                except queue.Empty:
                    pass

                # Run simulation step
                if hasattr(simulation, 'step'):
                    # Get current state
                    state = simulation.get_state()

                    # Process with AI controller (if applicable)
                    if hasattr(simulation, 'ai_controller'):
                        action = simulation.ai_controller.get_action(state)
                        simulation.apply_action(action)

                    # Step simulation
                    simulation.step_simulation(1)

                time.sleep(0.001)  # Small delay to prevent busy waiting

        except KeyboardInterrupt:
            pass
        finally:
            simulation.cleanup()

    def send_command_to_all(self, command: Dict):
        """
        Send command to all simulation instances

        Args:
            command: Command dictionary to send
        """
        for command_queue in self.command_queues:
            command_queue.put(command)

    def collect_results(self, timeout: float = 1.0) -> List[Dict]:
        """
        Collect results from all simulation instances

        Args:
            timeout: Timeout for collecting results

        Returns:
            List of results from all simulations
        """
        results = []
        start_time = time.time()

        while time.time() - start_time < timeout and len(results) < self.num_simulations:
            for i, result_queue in enumerate(self.result_queues):
                try:
                    result = result_queue.get_nowait()
                    results.append(result)
                except queue.Empty:
                    continue

            if len(results) == self.num_simulations:
                break

            time.sleep(0.001)

        return results

    def run_parallel_training_episode(self, steps_per_episode: int = 1000) -> List[Dict]:
        """
        Run a training episode across all parallel simulations

        Args:
            steps_per_episode: Number of steps per episode

        Returns:
            List of episode results from all simulations
        """
        episode_results = []

        # Send start episode command to all simulations
        self.send_command_to_all({
            'type': 'START_EPISODE',
            'steps': steps_per_episode
        })

        # Run episode in all simulations
        for step in range(steps_per_episode):
            # Send step command to all simulations
            self.send_command_to_all({
                'type': 'STEP',
                'data': {'step': step, 'total_steps': steps_per_episode}
            })

            # Collect results
            step_results = self.collect_results(timeout=0.1)
            episode_results.extend(step_results)

            # Check if all simulations completed
            if len([r for r in step_results if r.get('done', False)]) == self.num_simulations:
                break

        return episode_results

    def stop_all_simulations(self):
        """
        Stop all simulation processes
        """
        self.running = False

        # Send stop command to all processes
        for command_queue in self.command_queues:
            command_queue.put("STOP")

        # Wait for all processes to finish
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()

        # Clean up ZMQ context
        self.context.destroy()

class OptimizedSimulationEnvironment:
    """
    High-performance simulation environment with optimization techniques
    """
    def __init__(self, config: Dict):
        """
        Initialize optimized simulation environment

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.physics_frequency = config.get('physics_frequency', 240)  # Hz
        self.control_frequency = config.get('control_frequency', 100)  # Hz
        self.render_frequency = config.get('render_frequency', 60)     # Hz

        # Time management
        self.physics_dt = 1.0 / self.physics_frequency
        self.control_dt = 1.0 / self.control_frequency
        self.render_dt = 1.0 / self.render_frequency

        # Initialize physics engine with optimizations
        self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless training
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.physics_dt,
            numSolverIterations=10,
            enableConeFriction=True,
            deterministicOverlappingPairs=True,
            contactBreakingThreshold=0.001,
            contactSlop=0.001,
            maxNumCommandsPerStep=1000
        )

        # Set gravity
        p.setGravity(0, 0, -9.81)

        # Initialize robot
        self.robot_id = self._load_robot()
        self.joint_indices = self._get_joint_indices()

        # Initialize optimized components
        self.state_cache = {}  # Cache for frequently accessed state data
        self.transform_cache = {}  # Cache for transformation matrices
        self.collision_masks = {}  # Precomputed collision masks

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.state_history = deque(maxlen=1000)  # Keep last 1000 states

    def _load_robot(self):
        """
        Load robot with optimized parameters
        """
        # Load robot URDF
        robot_path = self.config.get('robot_urdf_path', 'humanoid.urdf')
        robot_id = p.loadURDF(
            robot_path,
            self.config.get('robot_start_pos', [0, 0, 1]),
            self.config.get('robot_start_orn', [0, 0, 0, 1]),
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        # Optimize robot dynamics
        for joint_idx in range(p.getNumJoints(robot_id)):
            p.changeDynamics(
                robot_id,
                joint_idx,
                linearDamping=0.04,
                angularDamping=0.04,
                contactStiffness=10000,
                contactDamping=1000
            )

        return robot_id

    def _get_joint_indices(self) -> List[int]:
        """
        Get indices of controllable joints
        """
        joint_indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode('utf-8')

            # Only include revolute joints (for this example)
            if joint_type == p.JOINT_REVOLUTE:
                joint_indices.append(i)

        return joint_indices

    def get_optimized_state(self) -> Dict:
        """
        Get robot state with optimizations
        """
        start_time = time.time()

        # Use cached transforms if available
        current_time = time.time()
        if current_time - self.state_cache.get('timestamp', 0) < 0.001:  # 1ms
            return self.state_cache['state']

        # Get base state
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, base_ang_vel = p.getBaseVelocity(self.robot_id)

        # Get joint states efficiently
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)

        # Get link states for important links (feet, hands, head)
        important_links = self.config.get('important_links', [-1, 0, 8, 12])  # base, head, left_hand, right_hand
        link_states = []
        for link_idx in important_links:
            link_state = p.getLinkState(self.robot_id, link_idx)
            link_states.append({
                'position': link_state[0],
                'orientation': link_state[1],
                'linear_velocity': link_state[6],
                'angular_velocity': link_state[7]
            })

        # Calculate derived quantities efficiently
        com_pos = self._calculate_center_of_mass()
        zmp = self._calculate_zmp(com_pos, base_vel)

        # Create state dictionary
        state = {
            'base': {
                'position': base_pos,
                'orientation': base_orn,
                'linear_velocity': base_vel,
                'angular_velocity': base_ang_vel
            },
            'joints': {
                joint_idx: {
                    'position': joint_state[0],
                    'velocity': joint_state[1],
                    'force': joint_state[3]
                }
                for joint_idx, joint_state in zip(self.joint_indices, joint_states)
            },
            'links': {
                link_idx: link_state
                for link_idx, link_state in zip(important_links, link_states)
            },
            'derived': {
                'com_position': com_pos,
                'zmp': zmp,
                'is_standing': self._check_standing_state(com_pos, base_orn),
                'balance_metrics': self._calculate_balance_metrics()
            },
            'timestamp': current_time,
            'computation_time': time.time() - start_time
        }

        # Cache the state
        self.state_cache = {
            'state': state,
            'timestamp': current_time
        }

        # Store in history
        self.state_history.append({
            'state': state,
            'computation_time': state['computation_time'],
            'timestamp': current_time
        })

        return state

    def _calculate_center_of_mass(self) -> List[float]:
        """
        Calculate center of mass of the robot
        """
        # This is a simplified calculation
        # In practice, you'd use the robot's mass properties
        total_mass = 0
        weighted_pos = np.zeros(3)

        # Get masses and positions of all links
        for i in range(p.getNumJoints(self.robot_id)):
            link_state = p.getLinkState(self.robot_id, i)
            link_pos = np.array(link_state[0])

            # In this simplified version, assume each link has equal mass
            mass = 1.0  # Placeholder - in reality, get from link dynamics
            total_mass += mass
            weighted_pos += link_pos * mass

        if total_mass > 0:
            com = weighted_pos / total_mass
            return com.tolist()
        else:
            return [0, 0, 0]

    def _calculate_zmp(self, com_pos: List[float], base_vel: List[float]) -> List[float]:
        """
        Calculate Zero Moment Point
        """
        g = 9.81
        com_height = com_pos[2]
        zmp_x = com_pos[0] - (com_height / g) * base_vel[0]
        zmp_y = com_pos[1] - (com_height / g) * base_vel[1]
        return [zmp_x, zmp_y, 0.0]

    def _check_standing_state(self, com_pos: List[float], base_orn: List[float]) -> bool:
        """
        Check if robot is in standing state
        """
        # Check if COM is above support polygon and orientation is upright
        com_height = com_pos[2]
        is_upright = abs(base_orn[2]) < 0.5  # Check if not too tilted
        is_elevated = com_height > 0.5  # Check if not on ground

        return is_upright and is_elevated

    def _calculate_balance_metrics(self) -> Dict:
        """
        Calculate various balance-related metrics
        """
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)

        # Calculate pitch and roll angles
        rotation_matrix = p.getMatrixFromQuaternion(base_orn)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

        # Extract pitch and roll from rotation matrix
        pitch = np.arcsin(-rotation_matrix[2, 0])
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

        # Calculate support polygon (simplified - just feet positions)
        left_foot_pos = p.getLinkState(self.robot_id, self.left_foot_link_index)[0]
        right_foot_pos = p.getLinkState(self.robot_id, self.right_foot_link_index)[0]

        # Calculate distance to support polygon center
        support_center_x = (left_foot_pos[0] + right_foot_pos[0]) / 2
        support_center_y = (left_foot_pos[1] + right_foot_pos[1]) / 2
        com_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        com_distance_to_support = np.sqrt(
            (com_pos[0] - support_center_x)**2 + (com_pos[1] - support_center_y)**2
        )

        return {
            'pitch_angle': pitch,
            'roll_angle': roll,
            'com_distance_to_support': com_distance_to_support,
            'is_balanced': com_distance_to_support < 0.15  # 15cm threshold
        }

    def apply_optimized_control(self, actions: Dict[int, float], dt: float = None):
        """
        Apply control commands with optimizations

        Args:
            actions: Dictionary mapping joint indices to target positions/torques
            dt: Time step for control (if None, uses default)
        """
        if dt is None:
            dt = self.control_dt

        # Use position control with velocity limits
        for joint_idx, target_value in actions.items():
            if joint_idx in self.joint_indices:
                # Apply position control with soft limits
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_value,
                    maxVelocity=self.config.get('max_joint_velocity', 5.0),
                    force=self.config.get('max_joint_torque', 100.0)
                )

    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics for the simulation
        """
        if not self.state_history:
            return {}

        # Calculate average computation time
        avg_comp_time = np.mean([state['computation_time'] for state in self.state_history])

        # Calculate simulation speed
        if len(self.state_history) > 1:
            time_span = (self.state_history[-1]['timestamp'] - self.state_history[0]['timestamp'])
            state_count = len(self.state_history)
            real_time_speed = state_count * self.physics_dt / time_span if time_span > 0 else 0
        else:
            real_time_speed = 0

        return {
            'average_state_computation_time_ms': avg_comp_time * 1000,
            'real_time_speed': real_time_speed,
            'states_per_second': 1 / avg_comp_time if avg_comp_time > 0 else 0,
            'total_states_cached': len(self.state_history)
        }

    def optimize_simulation_parameters(self):
        """
        Dynamically optimize simulation parameters based on performance
        """
        metrics = self.get_performance_metrics()

        # If simulation is too slow, reduce accuracy requirements
        if metrics.get('real_time_speed', 1.0) < 0.5:  # Running at less than 50% real-time
            # Reduce physics accuracy slightly
            p.setPhysicsEngineParameter(
                numSolverIterations=5,  # Reduce from 10
                contactBreakingThreshold=0.002,  # Increase from 0.001
                contactSlop=0.002  # Increase from 0.001
            )
            print("Reduced physics accuracy to improve performance")
        elif metrics.get('real_time_speed', 1.0) > 2.0:  # Running much faster than real-time
            # Can afford to increase accuracy
            p.setPhysicsEngineParameter(
                numSolverIterations=15,  # Increase from 10
                contactBreakingThreshold=0.0005,  # Decrease from 0.001
                contactSlop=0.0005  # Decrease from 0.001
            )
            print("Increased physics accuracy as performance allows")

class SimulationPerformanceOptimizer:
    """
    System for optimizing simulation performance
    """
    def __init__(self):
        self.optimization_history = []
        self.current_settings = {
            'num_solver_iterations': 10,
            'contact_breaking_threshold': 0.001,
            'contact_slop': 0.001,
            'max_num_commands_per_step': 1000,
            'collision_margin': 0.001
        }

    def adaptive_optimization(self, performance_metrics: Dict):
        """
        Adaptively optimize simulation parameters based on performance

        Args:
            performance_metrics: Current performance metrics
        """
        target_real_time_speed = performance_metrics.get('target_real_time_speed', 1.0)
        current_real_time_speed = performance_metrics.get('real_time_speed', 1.0)

        adjustment_factor = current_real_time_speed / target_real_time_speed

        # Adjust parameters based on performance
        if adjustment_factor < 0.8:  # Too slow, reduce accuracy
            self.current_settings['num_solver_iterations'] = max(5, int(self.current_settings['num_solver_iterations'] * 0.9))
            self.current_settings['contact_breaking_threshold'] = min(0.01, self.current_settings['contact_breaking_threshold'] * 1.1)
            self.current_settings['contact_slop'] = min(0.01, self.current_settings['contact_slop'] * 1.1)
        elif adjustment_factor > 1.2:  # Too fast, can increase accuracy
            self.current_settings['num_solver_iterations'] = min(20, int(self.current_settings['num_solver_iterations'] * 1.1))
            self.current_settings['contact_breaking_threshold'] = max(0.0001, self.current_settings['contact_breaking_threshold'] * 0.9)
            self.current_settings['contact_slop'] = max(0.0001, self.current_settings['contact_slop'] * 0.9)

        # Apply new settings
        p.setPhysicsEngineParameter(**self.current_settings)

        # Log optimization
        optimization_record = {
            'timestamp': time.time(),
            'previous_speed': current_real_time_speed,
            'adjustment_factor': adjustment_factor,
            'new_settings': self.current_settings.copy()
        }
        self.optimization_history.append(optimization_record)

    def get_optimization_suggestions(self, current_metrics: Dict) -> List[str]:
        """
        Get suggestions for performance optimization

        Args:
            current_metrics: Current performance metrics

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        if current_metrics.get('average_state_computation_time_ms', 0) > 10:
            suggestions.append("State computation taking too long (>10ms). Consider caching or simplifying state extraction.")

        if current_metrics.get('real_time_speed', 1.0) < 0.5:
            suggestions.append("Simulation running slowly (<50% real-time). Consider reducing physics accuracy or simplifying environment.")

        if current_metrics.get('cpu_percent', 0) > 80:
            suggestions.append("High CPU usage. Consider reducing simulation frequency or optimizing control algorithms.")

        if current_metrics.get('memory_percent', 0) > 80:
            suggestions.append("High memory usage. Consider optimizing data structures or increasing garbage collection.")

        return suggestions

def parallel_simulation_example():
    """
    Example of parallel simulation system
    """
    print("=== Parallel Simulation Example ===")

    # Configuration for simulations
    config = {
        'robot_urdf_path': 'humanoid.urdf',  # In practice, use actual URDF
        'robot_start_pos': [0, 0, 1],
        'physics_frequency': 240,
        'control_frequency': 100,
        'max_steps': 2000  # 10 seconds at 200Hz
    }

    # Create parallel simulation manager
    sim_manager = ParallelSimulationManager(
        num_simulations=4,
        simulation_class=OptimizedSimulationEnvironment,
        config=config
    )

    # Start simulations
    sim_manager.start_simulations()

    # Run parallel training episode
    print("Running parallel training episode...")
    start_time = time.time()

    episode_results = sim_manager.run_parallel_training_episode(steps_per_episode=1000)

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")
    print(f"Collected {len(episode_results)} results from {len(set(r['worker_id'] for r in episode_results))} simulations")

    # Stop all simulations
    sim_manager.stop_all_simulations()

    print("Parallel simulation example completed!")

def optimized_simulation_example():
    """
    Example of optimized simulation environment
    """
    print("=== Optimized Simulation Example ===")

    # Configuration
    config = {
        'physics_frequency': 480,  # Higher frequency for better accuracy
        'control_frequency': 200,
        'robot_start_pos': [0, 0, 1],
        'important_links': [8, 12, 14, 15]  # Example: feet and hands
    }

    # Create optimized simulation
    sim = OptimizedSimulationEnvironment(config)

    print("Running optimized simulation for 5 seconds...")

    start_time = time.time()
    step_count = 0

    while time.time() - start_time < 5.0:
        # Get optimized state
        state = sim.get_optimized_state()

        # Simple control: make joints oscillate
        actions = {}
        for i, joint_idx in enumerate(sim.joint_indices[:10]):  # First 10 joints
            actions[joint_idx] = 0.5 * np.sin(time.time() * 2 + i * 0.5)

        # Apply control
        sim.apply_optimized_control(actions)

        # Step simulation
        p.stepSimulation()

        step_count += 1

        # Print performance metrics periodically
        if step_count % 240 == 0:  # Every second
            metrics = sim.get_performance_metrics()
            print(f"Step {step_count}: Avg comp time: {metrics['average_state_computation_time_ms']:.2f}ms, "
                  f"Real-time speed: {metrics['real_time_speed']:.2f}x")

    # Get final performance metrics
    final_metrics = sim.get_performance_metrics()
    print(f"\nFinal Performance Metrics:")
    print(f"  Average state computation: {final_metrics['average_state_computation_time_ms']:.2f}ms")
    print(f"  Real-time simulation speed: {final_metrics['real_time_speed']:.2f}x")
    print(f"  States per second: {final_metrics['states_per_second']:.1f}")

    # Get optimization suggestions
    optimizer = SimulationPerformanceOptimizer()
    suggestions = optimizer.get_optimization_suggestions(final_metrics)
    if suggestions:
        print(f"\nOptimization suggestions:")
        for suggestion in suggestions:
            print(f"  - {suggestion}")

    print("\nOptimized simulation example completed!")

if __name__ == "__main__":
    # Run examples
    parallel_simulation_example()
    print("\n" + "="*60 + "\n")
    optimized_simulation_example()
```

## Multi-Agent Simulation

Implementing multiple humanoid robots interacting in the same environment:

```python
import numpy as np
import pybullet as p
import time
from typing import List, Dict, Tuple
import threading
import queue

class MultiHumanoidSimulation:
    """
    Simulation environment for multiple humanoid robots
    """
    def __init__(self, num_robots: int = 2, config: Dict = None):
        """
        Initialize multi-robot simulation

        Args:
            num_robots: Number of robots to simulate
            config: Configuration dictionary
        """
        self.num_robots = num_robots
        self.config = config or {}
        self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless training
        p.setGravity(0, 0, -9.81)

        # Set physics parameters
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSolverIterations=10,
            enableConeFriction=True
        )

        # Load ground plane
        p.setAdditionalSearchPath(p.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

        # Initialize robots
        self.robots = []
        self.robot_ids = []
        self.robot_configs = []
        self.robot_positions = []

        # Initialize robot communication network
        self.comm_network = RobotCommunicationNetwork(num_robots)

        # Initialize collision avoidance system
        self.collision_avoidance = CollisionAvoidanceSystem()

        # Initialize formation control
        self.formation_controller = FormationController()

        self._initialize_robots()

    def _initialize_robots(self):
        """
        Initialize multiple robot instances
        """
        # Define starting positions for robots (spaced apart)
        start_positions = self._calculate_start_positions()

        for i in range(self.num_robots):
            # Load robot at specific position
            robot_id = p.loadURDF(
                "humanoid.urdf",  # In practice, use actual humanoid URDF
                start_positions[i],
                [0, 0, 0, 1]  # Default orientation
            )

            # Get robot information
            joint_info = self._get_robot_joint_info(robot_id)
            link_info = self._get_robot_link_info(robot_id)

            robot_config = {
                'id': robot_id,
                'index': i,
                'joint_info': joint_info,
                'link_info': link_info,
                'start_position': start_positions[i],
                'current_position': start_positions[i],
                'communication_range': self.config.get('comm_range', 5.0),
                'collision_radius': self.config.get('collision_radius', 0.5)
            }

            self.robot_ids.append(robot_id)
            self.robot_configs.append(robot_config)
            self.robot_positions.append(start_positions[i])

    def _calculate_start_positions(self) -> List[List[float]]:
        """
        Calculate starting positions to space robots appropriately
        """
        positions = []
        radius = 2.0  # Distance between robots

        for i in range(self.num_robots):
            angle = 2 * np.pi * i / self.num_robots
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 1.0  # Height above ground
            positions.append([x, y, z])

        return positions

    def _get_robot_joint_info(self, robot_id: int) -> Dict:
        """
        Get joint information for a robot
        """
        joint_info = {}
        for i in range(p.getNumJoints(robot_id)):
            joint_data = p.getJointInfo(robot_id, i)
            joint_name = joint_data[1].decode('utf-8')
            joint_type = joint_data[2]
            joint_lower_limit = joint_data[8]
            joint_upper_limit = joint_data[9]

            joint_info[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower_limit': joint_lower_limit,
                'upper_limit': joint_upper_limit
            }

        return joint_info

    def _get_robot_link_info(self, robot_id: int) -> Dict:
        """
        Get link information for a robot
        """
        link_info = {}
        for i in range(p.getNumJoints(robot_id)):
            link_state = p.getLinkState(robot_id, i)
            link_info[i] = {
                'position': link_state[0],
                'orientation': link_state[1],
                'world_position': link_state[4],
                'world_orientation': link_state[5]
            }

        return link_info

    def get_multi_robot_state(self) -> Dict:
        """
        Get state information for all robots
        """
        multi_state = {
            'timestamp': time.time(),
            'robots': {},
            'environment': self._get_environment_state(),
            'communications': self.comm_network.get_recent_messages(),
            'collisions': self._detect_robot_collisions()
        }

        for i, robot_id in enumerate(self.robot_ids):
            # Get base state
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            base_vel, base_ang_vel = p.getBaseVelocity(robot_id)

            # Get joint states
            joint_states = {}
            for joint_idx in self.robot_configs[i]['joint_info']:
                joint_state = p.getJointState(robot_id, joint_idx)
                joint_states[joint_idx] = {
                    'position': joint_state[0],
                    'velocity': joint_state[1],
                    'force': joint_state[2],
                    'applied_torque': joint_state[3]
                }

            # Get important link states
            link_states = {}
            for link_idx in [8, 12, 14, 15]:  # Example: feet, hands
                if link_idx < p.getNumJoints(robot_id):
                    link_state = p.getLinkState(robot_id, link_idx)
                    link_states[link_idx] = {
                        'position': link_state[0],
                        'orientation': link_state[1],
                        'linear_velocity': link_state[6],
                        'angular_velocity': link_state[7]
                    }

            multi_state['robots'][i] = {
                'base': {
                    'position': base_pos,
                    'orientation': base_orn,
                    'linear_velocity': base_vel,
                    'angular_velocity': base_ang_vel
                },
                'joints': joint_states,
                'links': link_states,
                'derived_metrics': self._calculate_robot_derived_metrics(i, base_pos, base_orn)
            }

        return multi_state

    def _calculate_robot_derived_metrics(self, robot_idx: int, base_pos: List, base_orn: List) -> Dict:
        """
        Calculate derived metrics for a specific robot
        """
        # Calculate distance to other robots
        distances_to_others = []
        for other_idx in range(self.num_robots):
            if other_idx != robot_idx:
                other_pos = p.getBasePositionAndOrientation(self.robot_ids[other_idx])[0]
                distance = np.linalg.norm(np.array(base_pos) - np.array(other_pos))
                distances_to_others.append({
                    'robot_index': other_idx,
                    'distance': distance
                })

        # Check if within communication range
        comm_partners = []
        for other_idx in range(self.num_robots):
            if other_idx != robot_idx:
                other_pos = p.getBasePositionAndOrientation(self.robot_ids[other_idx])[0]
                distance = np.linalg.norm(np.array(base_pos) - np.array(other_pos))
                if distance <= self.robot_configs[robot_idx]['communication_range']:
                    comm_partners.append(other_idx)

        return {
            'distances_to_others': distances_to_others,
            'communication_partners': comm_partners,
            'is_colliding': self._is_robot_colliding(robot_idx),
            'balance_metrics': self._calculate_balance_metrics(robot_idx)
        }

    def _is_robot_colliding(self, robot_idx: int) -> bool:
        """
        Check if a robot is colliding with anything
        """
        robot_id = self.robot_ids[robot_idx]
        contacts = p.getContactPoints(bodyA=robot_id)

        # Check for collisions with other robots
        for contact in contacts:
            other_body = contact[2]  # bodyB
            if other_body in self.robot_ids and other_body != robot_id:
                return True

        # Check for collisions with environment
        if len(contacts) > 0:
            return True

        return False

    def _calculate_balance_metrics(self, robot_idx: int) -> Dict:
        """
        Calculate balance metrics for a specific robot
        """
        robot_id = self.robot_ids[robot_idx]
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)

        # Calculate COM position (simplified)
        com_pos = self._calculate_robot_com(robot_idx)

        # Calculate pitch and roll
        rotation_matrix = p.getMatrixFromQuaternion(base_orn)
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

        pitch = np.arcsin(-rotation_matrix[2, 0])
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

        return {
            'com_height': com_pos[2],
            'pitch_angle': pitch,
            'roll_angle': roll,
            'is_balanced': abs(pitch) < 0.5 and abs(roll) < 0.5 and com_pos[2] > 0.5
        }

    def _calculate_robot_com(self, robot_idx: int) -> List[float]:
        """
        Calculate center of mass for a robot (simplified)
        """
        # This is a simplified COM calculation
        # In practice, you'd use the robot's mass properties
        robot_id = self.robot_ids[robot_idx]
        total_mass = 0
        weighted_pos = np.zeros(3)

        # Get masses and positions of all links
        for i in range(p.getNumJoints(robot_id)):
            link_state = p.getLinkState(robot_id, i)
            link_pos = np.array(link_state[0])

            # Assume each link has equal mass for this example
            mass = 1.0
            total_mass += mass
            weighted_pos += link_pos * mass

        if total_mass > 0:
            com = weighted_pos / total_mass
            return com.tolist()
        else:
            return [0, 0, 0]

    def _detect_robot_collisions(self) -> List[Dict]:
        """
        Detect collisions between robots
        """
        collisions = []

        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                # Check for contact points between robots
                contacts = p.getContactPoints(
                    bodyA=self.robot_ids[i],
                    bodyB=self.robot_ids[j]
                )

                if contacts:
                    collision_info = {
                        'robot1_index': i,
                        'robot2_index': j,
                        'robot1_id': self.robot_ids[i],
                        'robot2_id': self.robot_ids[j],
                        'num_contacts': len(contacts),
                        'contact_details': []
                    }

                    for contact in contacts:
                        contact_detail = {
                            'link_index_a': contact[3],
                            'link_index_b': contact[4],
                            'position_on_a': contact[5],
                            'position_on_b': contact[6],
                            'contact_normal': contact[7],
                            'contact_distance': contact[8],
                            'normal_force': contact[9]
                        }
                        collision_info['contact_details'].append(contact_detail)

                    collisions.append(collision_info)

        return collisions

    def apply_multi_robot_control(self, control_commands: Dict[int, Dict]):
        """
        Apply control commands to multiple robots

        Args:
            control_commands: Dictionary mapping robot index to control commands
        """
        for robot_idx, commands in control_commands.items():
            if robot_idx < len(self.robot_ids):
                robot_id = self.robot_ids[robot_idx]

                # Apply joint commands
                if 'joint_commands' in commands:
                    for joint_idx, target_pos in commands['joint_commands'].items():
                        if joint_idx in self.robot_configs[robot_idx]['joint_info']:
                            p.setJointMotorControl2(
                                bodyIndex=robot_id,
                                jointIndex=joint_idx,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_pos,
                                maxVelocity=5.0,
                                force=100.0
                            )

                # Apply base commands if specified
                if 'base_command' in commands:
                    base_cmd = commands['base_command']
                    # This would involve more complex whole-body control
                    # For this example, we'll just apply forces
                    if 'force' in base_cmd:
                        p.applyExternalForce(
                            robot_id,
                            -1,  # Base link
                            base_cmd['force'],
                            base_cmd.get('position', [0, 0, 0]),
                            p.WORLD_FRAME
                        )

    def run_coordinated_behavior(self, duration: float = 10.0, behavior_type: str = "formation"):
        """
        Run coordinated behavior among multiple robots

        Args:
            duration: Duration to run behavior
            behavior_type: Type of coordinated behavior ("formation", "dance", "navigation")
        """
        start_time = time.time()
        step_count = 0

        print(f"Running {behavior_type} behavior with {self.num_robots} robots for {duration} seconds...")

        while time.time() - start_time < duration:
            current_time = time.time()

            # Get current multi-robot state
            multi_state = self.get_multi_robot_state()

            # Generate coordinated commands based on behavior type
            if behavior_type == "formation":
                control_commands = self.formation_controller.generate_formation_commands(
                    multi_state, current_time
                )
            elif behavior_type == "dance":
                control_commands = self._generate_dance_commands(multi_state, current_time)
            elif behavior_type == "navigation":
                control_commands = self._generate_navigation_commands(multi_state, current_time)
            else:
                # Default: simple coordinated movement
                control_commands = self._generate_coordinated_commands(multi_state, current_time)

            # Apply collision avoidance
            control_commands = self.collision_avoidance.avoid_collisions(
                multi_state, control_commands
            )

            # Apply control commands
            self.apply_multi_robot_control(control_commands)

            # Step simulation
            p.stepSimulation()

            step_count += 1

            # Periodically print status
            if step_count % 240 == 0:  # Every second
                avg_distance = self._calculate_average_inter_robot_distance(multi_state)
                collision_count = len(multi_state['collisions'])
                print(f"Time: {current_time - start_time:.1f}s, Avg distance: {avg_distance:.2f}m, "
                      f"Collisions: {collision_count}")

        print(f"Completed {behavior_type} behavior after {step_count} steps")

    def _generate_dance_commands(self, multi_state: Dict, current_time: float) -> Dict[int, Dict]:
        """
        Generate dance-like coordinated commands
        """
        commands = {}

        for robot_idx in range(self.num_robots):
            # Create wave-like motion pattern
            phase_offset = 2 * np.pi * robot_idx / self.num_robots
            dance_pattern = np.sin(current_time * 2 + phase_offset)

            # Apply dance pattern to joints
            joint_commands = {}
            for joint_idx in list(self.robot_configs[robot_idx]['joint_info'].keys())[:10]:  # First 10 joints
                base_pos = 0  # Rest position
                dance_amplitude = 0.3
                target_pos = base_pos + dance_amplitude * dance_pattern
                joint_commands[joint_idx] = target_pos

            commands[robot_idx] = {
                'joint_commands': joint_commands
            }

        return commands

    def _generate_navigation_commands(self, multi_state: Dict, current_time: float) -> Dict[int, Dict]:
        """
        Generate coordinated navigation commands
        """
        commands = {}

        # Define waypoints for formation navigation
        waypoints = [
            [5, 0, 1],    # Waypoint 1
            [5, 5, 1],    # Waypoint 2
            [0, 5, 1],    # Waypoint 3
            [0, 0, 1]     # Waypoint 4 (return to start)
        ]

        # Calculate current waypoint based on time
        waypoint_idx = int((current_time % 20) / 5) % len(waypoints)  # Cycle every 20 seconds
        target_waypoint = waypoints[waypoint_idx]

        for robot_idx in range(self.num_robots):
            # Each robot maintains formation relative to others
            robot_pos = multi_state['robots'][robot_idx]['base']['position']

            # Calculate formation offset for this robot
            formation_radius = 1.0
            formation_angle = 2 * np.pi * robot_idx / self.num_robots
            formation_offset = [
                formation_radius * np.cos(formation_angle),
                formation_radius * np.sin(formation_angle),
                0
            ]

            # Calculate target position
            target_pos = [
                target_waypoint[0] + formation_offset[0],
                target_waypoint[1] + formation_offset[1],
                target_waypoint[2]
            ]

            # Generate commands to move toward target
            commands[robot_idx] = self._generate_navigation_commands_for_robot(
                robot_idx, robot_pos, target_pos
            )

        return commands

    def _generate_navigation_commands_for_robot(self, robot_idx: int, current_pos: List[float],
                                              target_pos: List[float]) -> Dict:
        """
        Generate navigation commands for a single robot
        """
        # Simple proportional controller for navigation
        pos_error = np.array(target_pos) - np.array(current_pos)
        distance_to_target = np.linalg.norm(pos_error)

        if distance_to_target > 0.1:  # If not close to target
            # Normalize direction vector
            direction = pos_error / distance_to_target

            # Generate joint commands to move in direction
            # This is simplified - in reality, you'd use inverse kinematics
            joint_commands = {}
            for joint_idx in list(self.robot_configs[robot_idx]['joint_info'].keys())[:5]:  # First 5 joints
                # Apply small adjustments to move in desired direction
                adjustment = direction[0] * 0.1 * (joint_idx + 1)  # Different for each joint
                joint_commands[joint_idx] = self.robot_configs[robot_idx]['joint_info'][joint_idx]['lower_limit'] + \
                                          (self.robot_configs[robot_idx]['joint_info'][joint_idx]['upper_limit'] -
                                           self.robot_configs[robot_idx]['joint_info'][joint_idx]['lower_limit']) * 0.5 + \
                                          adjustment

            return {'joint_commands': joint_commands}
        else:
            # Near target, hold position
            return {'joint_commands': {}}  # Use current positions

    def _calculate_average_inter_robot_distance(self, multi_state: Dict) -> float:
        """
        Calculate average distance between all robot pairs
        """
        distances = []
        positions = []

        for robot_data in multi_state['robots'].values():
            pos = robot_data['base']['position']
            positions.append(np.array(pos))

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                distances.append(distance)

        return np.mean(distances) if distances else 0.0

    def get_simulation_metrics(self) -> Dict:
        """
        Get comprehensive simulation metrics
        """
        multi_state = self.get_multi_robot_state()

        metrics = {
            'num_robots': self.num_robots,
            'total_collisions': len(multi_state['collisions']),
            'average_distance_between_robots': self._calculate_average_inter_robot_distance(multi_state),
            'robots_in_collision': sum(1 for robot_data in multi_state['robots'].values()
                                      if robot_data['derived_metrics']['is_colliding']),
            'robots_balanced': sum(1 for robot_data in multi_state['robots'].values()
                                  if robot_data['derived_metrics']['balance_metrics']['is_balanced']),
            'communication_efficiency': self.comm_network.get_efficiency_metrics(),
            'formation_stability': self.formation_controller.get_stability_metrics(multi_state)
        }

        return metrics

    def cleanup(self):
        """
        Clean up simulation resources
        """
        p.disconnect(self.physics_client)
        print("Multi-robot simulation cleaned up")

class RobotCommunicationNetwork:
    """
    Simulated communication network between robots
    """
    def __init__(self, num_robots: int):
        self.num_robots = num_robots
        self.message_history = []
        self.connection_matrix = self._initialize_connections()
        self.packet_loss_rate = 0.05  # 5% packet loss
        self.latency_range = (0.01, 0.1)  # 10-100ms latency

    def _initialize_connections(self) -> np.ndarray:
        """
        Initialize connectivity matrix based on robot positions
        """
        connections = np.zeros((self.num_robots, self.num_robots))

        # In a real implementation, this would be based on actual distances
        # For this example, create a random but sparse connectivity matrix
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                # Connect with some probability based on "distance"
                if np.random.random() > 0.3:  # 70% chance of connection
                    connections[i, j] = 1
                    connections[j, i] = 1

        return connections

    def send_message(self, sender: int, receiver: int, message: Dict) -> bool:
        """
        Send message from one robot to another

        Args:
            sender: Sender robot index
            receiver: Receiver robot index
            message: Message dictionary

        Returns:
            True if message was delivered, False if lost due to packet loss
        """
        # Check if robots can communicate
        if self.connection_matrix[sender, receiver] == 0:
            return False

        # Simulate packet loss
        if np.random.random() < self.packet_loss_rate:
            return False

        # Add latency
        latency = np.random.uniform(*self.latency_range)
        delivery_time = time.time() + latency

        # Store message
        message_record = {
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'timestamp': time.time(),
            'delivery_time': delivery_time,
            'delivered': False
        }

        self.message_history.append(message_record)
        return True

    def get_recent_messages(self, time_window: float = 1.0) -> List[Dict]:
        """
        Get messages sent within the specified time window
        """
        current_time = time.time()
        recent_messages = []

        for msg in self.message_history:
            if current_time - msg['timestamp'] <= time_window:
                recent_messages.append(msg)

        return recent_messages

    def get_efficiency_metrics(self) -> Dict:
        """
        Get communication efficiency metrics
        """
        total_messages = len(self.message_history)
        delivered_messages = sum(1 for msg in self.message_history if msg['delivered'])

        return {
            'total_messages_sent': total_messages,
            'messages_delivered': delivered_messages,
            'delivery_rate': delivered_messages / total_messages if total_messages > 0 else 0,
            'average_latency': np.mean([msg['delivery_time'] - msg['timestamp']
                                      for msg in self.message_history]) if self.message_history else 0
        }

class CollisionAvoidanceSystem:
    """
    System for multi-robot collision avoidance
    """
    def __init__(self):
        self.safety_margin = 0.5  # meters
        self.avoidance_strength = 1.0

    def avoid_collisions(self, multi_state: Dict, control_commands: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Modify control commands to avoid collisions between robots

        Args:
            multi_state: Current multi-robot state
            control_commands: Original control commands

        Returns:
            Modified control commands with collision avoidance
        """
        modified_commands = control_commands.copy()

        for robot_idx in range(len(multi_state['robots'])):
            robot_pos = multi_state['robots'][robot_idx]['base']['position']

            # Check distances to other robots
            avoidance_force = np.array([0.0, 0.0, 0.0])

            for other_idx in range(len(multi_state['robots'])):
                if other_idx != robot_idx:
                    other_pos = multi_state['robots'][other_idx]['base']['position']
                    vector_to_other = np.array(other_pos) - np.array(robot_pos)
                    distance = np.linalg.norm(vector_to_other)

                    if 0 < distance < self.safety_margin * 2:  # Within avoidance range
                        # Calculate repulsive force
                        repulsion_strength = max(0, (self.safety_margin * 2 - distance) / (self.safety_margin * 2))
                        repulsion_direction = -vector_to_other / (distance + 1e-6)  # Away from other robot
                        avoidance_force += repulsion_direction * repulsion_strength * self.avoidance_strength

            # Apply avoidance force to modify control commands
            if np.linalg.norm(avoidance_force) > 0.01:  # Only if significant force
                modified_commands = self._apply_avoidance_force(
                    robot_idx, modified_commands, avoidance_force
                )

        return modified_commands

    def _apply_avoidance_force(self, robot_idx: int, commands: Dict[int, Dict],
                              avoidance_force: np.ndarray) -> Dict[int, Dict]:
        """
        Apply avoidance force to modify robot commands
        """
        if robot_idx not in commands:
            commands[robot_idx] = {'joint_commands': {}}

        # Modify joint commands to implement avoidance behavior
        # This is a simplified approach - in reality, you'd use more sophisticated methods
        for joint_idx in list(commands[robot_idx].get('joint_commands', {}).keys())[:5]:  # Modify first 5 joints
            original_command = commands[robot_idx]['joint_commands'][joint_idx]

            # Apply small adjustment based on avoidance force
            adjustment = avoidance_force[0] * 0.05  # Use x-component for this example
            new_command = original_command + adjustment

            # Ensure command stays within limits
            joint_limits = [j['lower_limit'] for j in self.robot_configs[robot_idx]['joint_info'].values()]
            joint_limits = [j['upper_limit'] for j in self.robot_configs[robot_idx]['joint_info'].values()]

            new_command = max(joint_limits[joint_idx]['lower_limit'],
                             min(joint_limits[joint_idx]['upper_limit'], new_command))

            commands[robot_idx]['joint_commands'][joint_idx] = new_command

        return commands

class FormationController:
    """
    Controller for maintaining robot formations
    """
    def __init__(self):
        self.formations = {
            'line': self._line_formation,
            'circle': self._circle_formation,
            'square': self._square_formation,
            'diamond': self._diamond_formation
        }
        self.current_formation = 'circle'
        self.formation_params = {'radius': 2.0}

    def _line_formation(self, robot_idx: int, total_robots: int, params: Dict) -> np.ndarray:
        """
        Calculate target position for line formation
        """
        spacing = params.get('spacing', 1.0)
        x_offset = robot_idx * spacing
        return np.array([x_offset, 0, 0])

    def _circle_formation(self, robot_idx: int, total_robots: int, params: Dict) -> np.ndarray:
        """
        Calculate target position for circular formation
        """
        radius = params.get('radius', 2.0)
        angle = 2 * np.pi * robot_idx / total_robots
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y, 0])

    def _square_formation(self, robot_idx: int, total_robots: int, params: Dict) -> np.ndarray:
        """
        Calculate target position for square formation
        """
        side_length = params.get('side_length', 2.0)
        side_robots = int(np.ceil(total_robots / 4))

        side = robot_idx // side_robots
        pos_on_side = robot_idx % side_robots

        if side == 0:  # Top side
            x = pos_on_side * side_length / (side_robots - 1) - side_length / 2
            y = side_length / 2
        elif side == 1:  # Right side
            x = side_length / 2
            y = pos_on_side * side_length / (side_robots - 1) - side_length / 2
        elif side == 2:  # Bottom side
            x = side_length / 2 - pos_on_side * side_length / (side_robots - 1)
            y = -side_length / 2
        else:  # Left side
            x = -side_length / 2
            y = side_length / 2 - pos_on_side * side_length / (side_robots - 1)

        return np.array([x, y, 0])

    def _diamond_formation(self, robot_idx: int, total_robots: int, params: Dict) -> np.ndarray:
        """
        Calculate target position for diamond formation
        """
        spacing = params.get('spacing', 1.5)
        # Place robots at diamond vertices and along edges
        positions = [
            [0, 2*spacing, 0],      # Top
            [spacing, spacing, 0],  # Upper right
            [0, 0, 0],              # Center
            [-spacing, spacing, 0], # Upper left
            [0, -2*spacing, 0],     # Bottom
            [spacing, -spacing, 0], # Lower right
            [-spacing, -spacing, 0] # Lower left
        ]

        if robot_idx < len(positions):
            return np.array(positions[robot_idx])
        else:
            # For additional robots, place randomly around the formation
            angle = 2 * np.pi * (robot_idx - len(positions)) / total_robots
            radius = 3 * spacing
            return np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

    def generate_formation_commands(self, multi_state: Dict, current_time: float) -> Dict[int, Dict]:
        """
        Generate commands to maintain current formation
        """
        commands = {}
        total_robots = len(multi_state['robots'])

        # Calculate center of formation (average of all robot positions)
        positions = []
        for robot_data in multi_state['robots'].values():
            pos = robot_data['base']['position']
            positions.append(np.array(pos))

        formation_center = np.mean(positions, axis=0) if positions else np.array([0, 0, 0])

        for robot_idx in range(total_robots):
            # Calculate desired position in formation
            desired_offset = self.formations[self.current_formation](
                robot_idx, total_robots, self.formation_params
            )
            desired_position = formation_center + desired_offset

            # Calculate current position
            current_position = np.array(multi_state['robots'][robot_idx]['base']['position'])

            # Calculate position error
            pos_error = desired_position - current_position

            # Generate control commands to move toward desired position
            commands[robot_idx] = self._generate_formation_control_commands(
                robot_idx, pos_error, current_position, desired_position
            )

        return commands

    def _generate_formation_control_commands(self, robot_idx: int, pos_error: np.ndarray,
                                           current_pos: np.ndarray, desired_pos: np.ndarray) -> Dict:
        """
        Generate control commands for formation maintenance
        """
        distance_to_desired = np.linalg.norm(pos_error)

        if distance_to_desired > 0.1:  # If significantly away from desired position
            # Normalize direction vector
            direction = pos_error / distance_to_desired

            # Generate joint commands to move in direction
            joint_commands = {}
            for joint_idx in list(self.robot_configs[robot_idx]['joint_info'].keys())[:5]:  # First 5 joints
                # Apply small adjustments to move in desired direction
                adjustment = direction[0] * 0.1 * (joint_idx + 1)  # Different for each joint
                joint_commands[joint_idx] = self.robot_configs[robot_idx]['joint_info'][joint_idx]['lower_limit'] + \
                                          (self.robot_configs[robot_idx]['joint_info'][joint_idx]['upper_limit'] -
                                           self.robot_configs[robot_idx]['joint_info'][joint_idx]['lower_limit']) * 0.5 + \
                                          adjustment

            return {'joint_commands': joint_commands}
        else:
            # Close to desired position, hold formation
            return {'joint_commands': {}}  # Use current positions

    def get_stability_metrics(self, multi_state: Dict) -> Dict:
        """
        Get metrics about formation stability
        """
        positions = []
        for robot_data in multi_state['robots'].values():
            pos = robot_data['base']['position']
            positions.append(np.array(pos))

        if not positions:
            return {'formation_error': float('inf'), 'stability': 0.0}

        formation_center = np.mean(positions, axis=0)

        # Calculate deviation from ideal formation
        total_deviation = 0
        for robot_idx, current_pos in enumerate(positions):
            desired_offset = self.formations[self.current_formation](
                robot_idx, len(positions), self.formation_params
            )
            desired_pos = formation_center + desired_offset

            deviation = np.linalg.norm(current_pos - desired_pos)
            total_deviation += deviation

        avg_deviation = total_deviation / len(positions)

        # Stability is inversely related to deviation (normalized)
        max_acceptable_deviation = 1.0  # meters
        stability = max(0, 1 - avg_deviation / max_acceptable_deviation)

        return {
            'formation_error': avg_deviation,
            'stability': stability,
            'center_position': formation_center.tolist()
        }

def multi_robot_simulation_example():
    """
    Example of multi-robot simulation with coordination
    """
    print("=== Multi-Robot Simulation Example ===")

    # Create multi-robot simulation
    multi_sim = MultiHumanoidSimulation(num_robots=3)

    # Run different coordinated behaviors
    print("\n1. Running formation behavior...")
    multi_sim.run_coordinated_behavior(duration=10.0, behavior_type="formation")

    print("\n2. Running dance behavior...")
    multi_sim.run_coordinated_behavior(duration=10.0, behavior_type="dance")

    print("\n3. Running navigation behavior...")
    multi_sim.run_coordinated_behavior(duration=15.0, behavior_type="navigation")

    # Get final metrics
    metrics = multi_sim.get_simulation_metrics()
    print(f"\nFinal Simulation Metrics:")
    print(f"  - Number of robots: {metrics['num_robots']}")
    print(f"  - Total collisions: {metrics['total_collisions']}")
    print(f"  - Robots in collision: {metrics['robots_in_collision']}")
    print(f"  - Robots balanced: {metrics['robots_balanced']}")
    print(f"  - Average inter-robot distance: {metrics['average_distance_between_robots']:.2f}m")
    print(f"  - Communication delivery rate: {metrics['communication_efficiency']['delivery_rate']:.2f}")
    print(f"  - Formation stability: {metrics['formation_stability']['stability']:.2f}")

    # Clean up
    multi_sim.cleanup()
    print("\nMulti-robot simulation example completed!")

if __name__ == "__main__":
    multi_robot_simulation_example()
```

## Integration with Machine Learning

### Simulation-to-Real Transfer Learning

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import random

class DomainRandomization:
    """
    Domain randomization techniques for sim-to-real transfer
    """
    def __init__(self, simulation_env):
        self.sim_env = simulation_env
        self.randomization_params = {
            'mass_variance': (0.8, 1.2),      # 80% to 120% of original mass
            'friction_variance': (0.5, 2.0),  # 50% to 200% of original friction
            'gravity_variance': (0.9, 1.1),   # 90% to 110% of original gravity
            'visual_textures': True,          # Randomize visual appearance
            'lighting_conditions': True,      # Randomize lighting
            'sensor_noise': (0.0, 0.1),       # Add sensor noise
            'actuator_dynamics': (0.9, 1.1)   # Randomize actuator dynamics
        }

    def randomize_domain(self):
        """
        Apply domain randomization to the simulation
        """
        # Randomize physical parameters
        self._randomize_masses()
        self._randomize_friction()
        self._randomize_gravity()
        self._randomize_visual_appearance()
        self._randomize_lighting()
        self._add_sensor_noise()
        self._randomize_actuator_dynamics()

    def _randomize_masses(self):
        """
        Randomize masses of robot links
        """
        for robot_id in self.sim_env.robot_ids:
            for joint_idx in range(p.getNumJoints(robot_id)):
                # Get original mass
                link_state = p.getDynamicsInfo(robot_id, joint_idx)
                original_mass = link_state[0]

                # Apply randomization
                mass_multiplier = random.uniform(*self.randomization_params['mass_variance'])
                new_mass = original_mass * mass_multiplier

                # Set new mass (this is simplified - in reality, you'd need to recreate the body)
                try:
                    p.changeDynamics(robot_id, joint_idx, mass=new_mass)
                except:
                    # If direct mass change fails, continue
                    pass

    def _randomize_friction(self):
        """
        Randomize friction coefficients
        """
        for robot_id in self.sim_env.robot_ids:
            for joint_idx in range(p.getNumJoints(robot_id)):
                # Get original friction
                link_state = p.getDynamicsInfo(robot_id, joint_idx)
                original_lateral_friction = link_state[1]

                # Apply randomization
                friction_multiplier = random.uniform(*self.randomization_params['friction_variance'])
                new_friction = original_lateral_friction * friction_multiplier

                p.changeDynamics(robot_id, joint_idx, lateralFriction=new_friction)

    def _randomize_gravity(self):
        """
        Randomize gravity
        """
        base_gravity = 9.81
        gravity_multiplier = random.uniform(*self.randomization_params['gravity_variance'])
        new_gravity = base_gravity * gravity_multiplier

        p.setGravity(0, 0, -new_gravity)

    def _randomize_visual_appearance(self):
        """
        Randomize visual appearance (textures, colors)
        """
        if self.randomization_params['visual_textures']:
            for robot_id in self.sim_env.robot_ids:
                for link_idx in range(p.getNumJoints(robot_id)):
                    # Randomize color
                    color = [
                        random.uniform(0.1, 1.0),  # R
                        random.uniform(0.1, 1.0),  # G
                        random.uniform(0.1, 1.0),  # B
                        1.0  # Alpha
                    ]
                    try:
                        p.changeVisualShape(robot_id, link_idx, rgbaColor=color)
                    except:
                        # If visual shape change fails, continue
                        pass

    def _randomize_lighting(self):
        """
        Randomize lighting conditions
        """
        if self.randomization_params['lighting_conditions']:
            # In PyBullet, lighting is controlled by the renderer
            # For this example, we'll just note that lighting randomization
            # would happen in the rendering pipeline
            pass

    def _add_sensor_noise(self):
        """
        Add noise to sensor readings
        """
        # This would be implemented in the sensor reading functions
        # For simulation, we can add noise during state extraction
        pass

    def _randomize_actuator_dynamics(self):
        """
        Randomize actuator dynamics
        """
        # This would involve changing actuator parameters
        # such as gear ratios, motor constants, etc.
        pass

    def get_randomized_observation(self, base_observation: np.ndarray) -> np.ndarray:
        """
        Add domain randomization effects to observation

        Args:
            base_observation: Base observation from simulation

        Returns:
            Randomized observation
        """
        randomized_obs = base_observation.copy()

        # Add sensor noise
        noise_level = random.uniform(*self.randomization_params['sensor_noise'])
        noise = np.random.normal(0, noise_level, randomized_obs.shape)
        randomized_obs += noise

        # Apply other randomization effects
        # This could include time delays, quantization, etc.

        return randomized_obs

class CurriculumLearning:
    """
    Curriculum learning for humanoid robot training
    """
    def __init__(self, initial_difficulty=0.1, max_difficulty=1.0):
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = initial_difficulty
        self.performance_history = []
        self.performance_threshold = 0.7  # 70% success rate to advance

    def update_difficulty(self, episode_performance: float):
        """
        Update difficulty based on performance

        Args:
            episode_performance: Performance score for current episode (0-1)
        """
        self.performance_history.append(episode_performance)

        # Keep only recent performance (last 10 episodes)
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]

        # Calculate average recent performance
        avg_performance = np.mean(self.performance_history)

        # Adjust difficulty based on performance
        if avg_performance >= self.performance_threshold and len(self.performance_history) >= 5:
            # Performance is good, increase difficulty
            self.current_difficulty = min(self.max_difficulty, self.current_difficulty * 1.1)
            print(f"Increasing difficulty to {self.current_difficulty:.2f}")
        elif avg_performance < self.performance_threshold * 0.8:
            # Performance is poor, decrease difficulty
            self.current_difficulty = max(self.initial_difficulty, self.current_difficulty * 0.95)
            print(f"Decreasing difficulty to {self.current_difficulty:.2f}")

    def get_current_task(self) -> Dict:
        """
        Get current task based on difficulty level

        Returns:
            Task configuration dictionary
        """
        if self.current_difficulty < 0.3:
            # Basic task: standing still
            task = {
                'type': 'balance',
                'duration': 5.0,
                'target': 'maintain_upright_position',
                'difficulty': 'easy',
                'rewards': {
                    'balance_bonus': 1.0,
                    'fall_penalty': -10.0
                }
            }
        elif self.current_difficulty < 0.6:
            # Intermediate task: simple walking
            task = {
                'type': 'locomotion',
                'duration': 10.0,
                'target': 'walk_forward_2m',
                'difficulty': 'medium',
                'rewards': {
                    'forward_progress': 0.1,
                    'balance_bonus': 0.5,
                    'fall_penalty': -10.0,
                    'energy_efficiency': -0.01
                }
            }
        elif self.current_difficulty < 0.8:
            # Advanced task: walking with obstacles
            task = {
                'type': 'navigation',
                'duration': 15.0,
                'target': 'navigate_to_goal_with_obstacles',
                'difficulty': 'hard',
                'rewards': {
                    'goal_reached': 10.0,
                    'obstacle_avoidance': 2.0,
                    'balance_bonus': 0.3,
                    'fall_penalty': -10.0,
                    'energy_efficiency': -0.01
                }
            }
        else:
            # Expert task: complex maneuvers
            task = {
                'type': 'complex_behavior',
                'duration': 20.0,
                'target': 'perform_complex_manipulation_while_balancing',
                'difficulty': 'expert',
                'rewards': {
                    'task_completion': 20.0,
                    'precision_bonus': 5.0,
                    'balance_maintenance': 2.0,
                    'fall_penalty': -10.0,
                    'energy_efficiency': -0.01
                }
            }

        return task

class SimulationTrainer:
    """
    Training system that integrates simulation with machine learning
    """
    def __init__(self, policy_network, value_network=None, config=None):
        self.policy_network = policy_network
        self.value_network = value_network
        self.config = config or {}
        self.optimizer = torch.optim.Adam(
            list(policy_network.parameters()) +
            (list(value_network.parameters()) if value_network else []),
            lr=config.get('learning_rate', 3e-4)
        )
        self.memory = []
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)  # Soft update parameter

        # Domain randomization
        self.domain_rand = DomainRandomization(None)  # Will be set when simulation is provided

        # Curriculum learning
        self.curriculum = CurriculumLearning()

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []

    def collect_experience(self, simulation, num_episodes=1000):
        """
        Collect experience from simulation

        Args:
            simulation: Simulation environment
            num_episodes: Number of episodes to collect
        """
        for episode in range(num_episodes):
            # Apply domain randomization for this episode
            self.domain_rand.randomize_domain()

            # Get current task from curriculum
            current_task = self.curriculum.get_current_task()

            # Run episode in simulation
            episode_data = self.run_episode(simulation, current_task)

            # Store experience
            self.memory.extend(episode_data)

            # Update curriculum based on performance
            success_rate = self.calculate_episode_success(episode_data)
            self.curriculum.update_difficulty(success_rate)

            # Periodically train
            if len(self.memory) > self.batch_size * 10:
                self.train_step()

            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Difficulty: {self.curriculum.current_difficulty:.2f}")

    def run_episode(self, simulation, task_config) -> List[Dict]:
        """
        Run a single episode in simulation

        Args:
            simulation: Simulation environment
            task_config: Task configuration

        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        episode_data = []
        state = simulation.reset()

        for step in range(int(task_config['duration'] * 240)):  # Assuming 240Hz simulation
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy_network(state_tensor)
                action = torch.multinomial(action_probs, 1).item()

            # Apply domain randomization to observation
            randomized_state = self.domain_rand.get_randomized_observation(state)

            # Take action in simulation
            next_state, reward, done, info = simulation.step(action)

            # Apply domain randomization to next state
            randomized_next_state = self.domain_rand.get_randomized_observation(next_state)

            # Store experience
            episode_data.append({
                'state': randomized_state,
                'action': action,
                'reward': reward,
                'next_state': randomized_next_state,
                'done': done
            })

            state = next_state

            if done:
                break

        return episode_data

    def train_step(self):
        """
        Perform a single training step
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]

        # Extract batch data
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.BoolTensor([exp['done'] for exp in batch])

        # Compute value targets (for actor-critic)
        if self.value_network:
            with torch.no_grad():
                next_values = self.value_network(next_states)
                targets = rewards + (self.gamma * next_values.squeeze() * ~dones)

            # Update value network
            current_values = self.value_network(states).squeeze()
            value_loss = nn.MSELoss()(current_values, targets)

            # Update policy network
            action_probs = self.policy_network(states)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

            # Compute advantage
            advantages = targets - current_values.detach()
            policy_loss = -(log_probs * advantages.detach()).mean()

            # Total loss
            loss = policy_loss + 0.5 * value_loss

        else:
            # Simple policy gradient
            action_probs = self.policy_network(states)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
            loss = -(log_probs * rewards).mean()

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy_network.parameters()) +
                                     (list(self.value_network.parameters()) if self.value_network else []),
                                     40)
        self.optimizer.step()

    def save_model(self, filepath: str):
        """
        Save trained model

        Args:
            filepath: Path to save the model
        """
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict() if self.value_network else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'curriculum_difficulty': self.curriculum.current_difficulty,
            'episode_rewards': self.episode_rewards,
            'config': self.config
        }, filepath)

    def load_model(self, filepath: str):
        """
        Load trained model

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        if self.value_network and checkpoint['value_state_dict']:
            self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.curriculum.current_difficulty = checkpoint['curriculum_difficulty']
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"Model loaded from {filepath}")

def ml_integration_example():
    """
    Example of machine learning integration with simulation
    """
    print("=== ML Integration with Simulation Example ===")

    # Define simple neural network for policy
    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(PolicyNetwork, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            return self.network(x)

    # Initialize networks
    policy_net = PolicyNetwork(state_dim=100, action_dim=20)  # Example dimensions
    value_net = PolicyNetwork(state_dim=100, action_dim=1)    # For value estimation

    # Configuration
    config = {
        'learning_rate': 3e-4,
        'batch_size': 64,
        'gamma': 0.99,
        'tau': 0.005
    }

    # Create trainer
    trainer = SimulationTrainer(policy_net, value_net, config)

    # In a real implementation, you would connect this to an actual simulation
    # For this example, we'll just demonstrate the structure
    print("ML integration system initialized with:")
    print(f"  - Policy network: {sum(p.numel() for p in policy_net.parameters())} parameters")
    print(f"  - Value network: {sum(p.numel() for p in value_net.parameters())} parameters")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Batch size: {config['batch_size']}")

    # Simulate training process
    print("\nSimulating training process...")
    for epoch in range(5):  # Simulate 5 training epochs
        print(f"  Epoch {epoch + 1}/5: Collecting experience and training...")
        time.sleep(0.5)  # Simulate training time

    print("\nML integration example completed!")
    print("In a real implementation, this would connect to actual simulation environments")
    print("for data collection and training of humanoid robot control policies.")

if __name__ == "__main__":
    ml_integration_example()
```

## Summary

Advanced simulation techniques are essential for developing sophisticated AI humanoid robots. These techniques include:

1. **High-Performance Simulation**: Optimizing physics engines, using parallel processing, and implementing efficient algorithms
2. **Multi-Robot Simulation**: Coordinating multiple robots with communication and collision avoidance
3. **Machine Learning Integration**: Domain randomization, curriculum learning, and sim-to-real transfer techniques
4. **Realistic Sensor Simulation**: Accurate modeling of cameras, LIDAR, IMUs, and other sensors
5. **Environmental Effects**: Simulating real-world conditions like lighting, weather, and dynamic obstacles

These advanced techniques enable researchers and developers to create more realistic, efficient, and effective simulation environments that can accelerate the development of AI-powered humanoid robots while ensuring safety and reducing costs associated with physical prototyping.