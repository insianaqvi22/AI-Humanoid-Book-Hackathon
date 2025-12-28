---
title: Integration Examples and Best Practices
sidebar_position: 5
description: Integration examples and best practices for AI humanoid robotics simulation
---

# Integration Examples and Best Practices

This section provides practical examples of integrating simulation environments with real-world robotics systems and demonstrates best practices for developing robust simulation frameworks.

## Real-to-Sim Integration

### System Architecture for Real-to-Sim Integration

```python
import asyncio
import websockets
import json
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Callable
import queue

class RealToSimIntegration:
    """
    System for integrating real robot data with simulation environment
    """
    def __init__(self, robot_interface, sim_environment):
        """
        Initialize real-to-sim integration system

        Args:
            robot_interface: Interface to real robot hardware
            sim_environment: Simulation environment instance
        """
        self.robot_interface = robot_interface
        self.sim_environment = sim_environment

        # Data synchronization queues
        self.real_sensor_queue = queue.Queue(maxsize=100)
        self.sim_command_queue = queue.Queue(maxsize=100)

        # State synchronization
        self.real_state = {}
        self.sim_state = {}
        self.state_lock = threading.Lock()

        # Configuration
        self.sync_frequency = 100  # Hz
        self.time_warp_factor = 1.0  # Real-time by default
        self.calibration_data = {}

        # Connection status
        self.connected = False
        self.sync_thread = None
        self.command_thread = None

    def connect_to_robot(self):
        """
        Establish connection to real robot
        """
        try:
            # Connect to robot
            self.robot_interface.connect()

            # Initialize calibration
            self.calibrate_systems()

            # Start synchronization threads
            self.start_sync_threads()

            self.connected = True
            print("Successfully connected to real robot and started sync threads")
            return True

        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False

    def calibrate_systems(self):
        """
        Calibrate real and simulated systems for accurate mapping
        """
        print("Starting system calibration...")

        # Calibrate joint positions
        real_joints = self.robot_interface.get_joint_positions()
        sim_joints = self.sim_environment.get_joint_positions()

        # Calculate offset between real and simulated positions
        self.calibration_data['joint_offsets'] = {}
        for joint_name, real_pos in real_joints.items():
            if joint_name in sim_joints:
                offset = real_pos - sim_joints[joint_name]
                self.calibration_data['joint_offsets'][joint_name] = offset

        # Calibrate IMU data
        real_imu = self.robot_interface.get_imu_data()
        sim_imu = self.sim_environment.get_imu_data()

        self.calibration_data['imu_offset'] = {
            'acceleration': np.array(real_imu['acceleration']) - np.array(sim_imu['acceleration']),
            'angular_velocity': np.array(real_imu['angular_velocity']) - np.array(sim_imu['angular_velocity'])
        }

        print("System calibration completed")

    def start_sync_threads(self):
        """
        Start synchronization threads
        """
        # Thread for syncing real sensor data to simulation
        self.sync_thread = threading.Thread(target=self.sync_real_to_sim, daemon=True)
        self.sync_thread.start()

        # Thread for syncing simulation commands to real robot
        self.command_thread = threading.Thread(target=self.sync_sim_to_real, daemon=True)
        self.command_thread.start()

    def sync_real_to_sim(self):
        """
        Continuously sync real robot sensor data to simulation
        """
        while self.connected:
            try:
                # Get real sensor data
                real_data = self.robot_interface.get_all_sensor_data()

                # Apply calibration
                calibrated_data = self.apply_calibration(real_data, 'sensor')

                # Update simulation state
                self.update_simulation_with_real_data(calibrated_data)

                # Add to queue for other systems to consume
                if not self.real_sensor_queue.full():
                    self.real_sensor_queue.put({
                        'timestamp': time.time(),
                        'data': calibrated_data,
                        'sync_time': time.time()
                    })

                # Control loop frequency
                time.sleep(1.0 / self.sync_frequency)

            except Exception as e:
                print(f"Error in real-to-sim sync: {e}")
                time.sleep(0.1)  # Brief pause before retrying

    def sync_sim_to_real(self):
        """
        Continuously sync simulation commands to real robot
        """
        while self.connected:
            try:
                # Get commands from simulation
                if not self.sim_command_queue.empty():
                    command_data = self.sim_command_queue.get_nowait()

                    # Apply calibration
                    calibrated_command = self.apply_calibration(command_data['data'], 'command')

                    # Send to real robot
                    self.robot_interface.send_commands(calibrated_command)

                    # Update real state
                    with self.state_lock:
                        self.real_state.update({
                            'last_command': calibrated_command,
                            'command_timestamp': command_data['timestamp']
                        })

                time.sleep(1.0 / self.sync_frequency)

            except queue.Empty:
                # No commands to process, continue
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in sim-to-real sync: {e}")
                time.sleep(0.1)

    def apply_calibration(self, data, data_type):
        """
        Apply calibration offsets to data

        Args:
            data: Raw sensor data or command data
            data_type: 'sensor' or 'command'

        Returns:
            Calibrated data
        """
        calibrated = data.copy()

        if data_type == 'sensor':
            # Apply sensor calibration
            if 'joints' in calibrated:
                for joint_name, position in calibrated['joints'].items():
                    if joint_name in self.calibration_data['joint_offsets']:
                        calibrated['joints'][joint_name] = position - self.calibration_data['joint_offsets'][joint_name]

            if 'imu' in calibrated:
                if 'acceleration' in calibrated['imu']:
                    calibrated['imu']['acceleration'] = np.array(calibrated['imu']['acceleration']) - self.calibration_data['imu_offset']['acceleration']
                if 'angular_velocity' in calibrated['imu']:
                    calibrated['imu']['angular_velocity'] = np.array(calibrated['imu']['angular_velocity']) - self.calibration_data['imu_offset']['angular_velocity']

        elif data_type == 'command':
            # Apply command calibration
            if 'joint_commands' in calibrated:
                for joint_name, command in calibrated['joint_commands'].items():
                    if joint_name in self.calibration_data['joint_offsets']:
                        calibrated['joint_commands'][joint_name] = command + self.calibration_data['joint_offsets'][joint_name]

        return calibrated

    def update_simulation_with_real_data(self, real_data):
        """
        Update simulation environment with real robot data

        Args:
            real_data: Calibrated real robot sensor data
        """
        # Update joint positions in simulation
        if 'joints' in real_data:
            self.sim_environment.set_joint_positions(real_data['joints'])

        # Update IMU data in simulation
        if 'imu' in real_data:
            self.sim_environment.set_imu_data(real_data['imu'])

        # Update other sensor data as needed
        if 'force_torque' in real_data:
            self.sim_environment.set_force_torque_data(real_data['force_torque'])

        # Update simulation state
        with self.state_lock:
            self.sim_state.update({
                'last_real_update': time.time(),
                'real_data': real_data
            })

    def get_synchronized_state(self) -> Dict:
        """
        Get synchronized state between real robot and simulation

        Returns:
            Dictionary containing synchronized state
        """
        with self.state_lock:
            return {
                'real_state': self.real_state.copy(),
                'sim_state': self.sim_state.copy(),
                'calibration_data': self.calibration_data.copy(),
                'sync_status': {
                    'connected': self.connected,
                    'sync_frequency': self.sync_frequency,
                    'time_warp_factor': self.time_warp_factor
                }
            }

    def switch_control_mode(self, mode: str):
        """
        Switch between different control modes

        Args:
            mode: 'teleoperation', 'autonomous', or 'mixed_reality'
        """
        if mode == 'teleoperation':
            # In teleoperation, real robot follows simulation exactly
            self.time_warp_factor = 1.0
            self.sync_frequency = 200  # Higher frequency for teleoperation
        elif mode == 'autonomous':
            # In autonomous mode, simulation runs independently
            self.time_warp_factor = 1.0
            self.sync_frequency = 50  # Lower frequency for autonomous
        elif mode == 'mixed_reality':
            # Mixed reality: real robot provides physical constraints, sim provides intelligence
            self.time_warp_factor = 1.0
            self.sync_frequency = 100
        else:
            raise ValueError(f"Unknown control mode: {mode}")

        print(f"Switched to control mode: {mode}")

    def run_mixed_reality_session(self, duration=60.0):
        """
        Run a mixed reality session where real robot and simulation work together

        Args:
            duration: Duration of the session in seconds
        """
        print(f"Starting mixed reality session for {duration} seconds...")

        # Switch to mixed reality mode
        self.switch_control_mode('mixed_reality')

        start_time = time.time()
        last_update = start_time

        try:
            while time.time() - start_time < duration:
                current_time = time.time()

                # Get synchronized state
                state = self.get_synchronized_state()

                # In mixed reality, use AI from simulation to enhance real robot
                if 'real_data' in state['sim_state']:
                    real_data = state['sim_state']['real_data']

                    # Use simulation AI to interpret real sensor data
                    ai_interpretation = self.sim_environment.interpret_sensory_data(real_data)

                    # Generate enhanced commands based on AI interpretation
                    enhanced_commands = self.generate_enhanced_commands(ai_interpretation, real_data)

                    # Send enhanced commands to real robot
                    self.robot_interface.send_commands(enhanced_commands)

                # Log status periodically
                if current_time - last_update > 1.0:  # Every second
                    print(f"Mixed reality session time: {current_time - start_time:.1f}s")
                    last_update = current_time

                time.sleep(0.01)  # Small delay

        except KeyboardInterrupt:
            print("Mixed reality session interrupted by user")

        print("Mixed reality session completed")

    def generate_enhanced_commands(self, ai_interpretation, real_data):
        """
        Generate enhanced commands based on AI interpretation and real data

        Args:
            ai_interpretation: AI interpretation of sensory data
            real_data: Real robot sensor data

        Returns:
            Enhanced command dictionary
        """
        # This would typically involve:
        # 1. Using AI from simulation to understand the environment
        # 2. Planning optimal actions based on understanding
        # 3. Generating safe, feasible commands for the real robot

        enhanced_commands = {
            'joint_commands': {},
            'gripper_commands': {},
            'locomotion_commands': {}
        }

        # Example: Use AI interpretation to adjust joint stiffness
        if 'object_detected' in ai_interpretation:
            # If object detected, increase gripper stiffness
            enhanced_commands['gripper_commands']['stiffness'] = 0.8
        else:
            # Otherwise, use compliant control
            enhanced_commands['gripper_commands']['stiffness'] = 0.3

        # Example: Adjust walking pattern based on terrain analysis
        if 'terrain_analysis' in ai_interpretation:
            terrain_type = ai_interpretation['terrain_analysis']['type']
            if terrain_type == 'uneven':
                # Use more cautious walking pattern
                enhanced_commands['locomotion_commands']['step_height'] = 0.15
                enhanced_commands['locomotion_commands']['step_width'] = 0.3
            elif terrain_type == 'slippery':
                # Use smaller steps for better traction
                enhanced_commands['locomotion_commands']['step_length'] = 0.2
                enhanced_commands['locomotion_commands']['walking_speed'] = 0.3

        return enhanced_commands

    def disconnect(self):
        """
        Disconnect from robot and stop synchronization
        """
        self.connected = False

        # Wait for threads to finish
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=2.0)

        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=2.0)

        # Disconnect from robot
        if hasattr(self.robot_interface, 'disconnect'):
            self.robot_interface.disconnect()

        print("Disconnected from robot and stopped synchronization")

def real_to_sim_integration_example():
    """
    Example of real-to-sim integration
    """
    print("=== Real-to-Sim Integration Example ===")

    # In a real implementation, you would have:
    # real_robot_interface = RealRobotInterface(robot_ip="192.168.1.100")
    # sim_env = SimulationEnvironment(sim_type="pybullet")

    # For this example, we'll create mock interfaces
    class MockRobotInterface:
        def connect(self):
            print("Connected to mock robot")

        def get_joint_positions(self):
            return {f'joint_{i}': 0.1*i for i in range(10)}

        def get_imu_data(self):
            return {
                'acceleration': [0.01, 0.02, 9.81],
                'angular_velocity': [0.001, 0.002, 0.003]
            }

        def get_all_sensor_data(self):
            return {
                'joints': self.get_joint_positions(),
                'imu': self.get_imu_data(),
                'force_torque': {'left_foot': [10, 5, 2], 'right_foot': [12, 3, 1]}
            }

        def send_commands(self, commands):
            print(f"Sending commands to mock robot: {len(commands)} items")

        def disconnect(self):
            print("Disconnected from mock robot")

    class MockSimulationEnvironment:
        def get_joint_positions(self):
            return {f'joint_{i}': 0.1*i + 0.01 for i in range(10)}

        def get_imu_data(self):
            return {
                'acceleration': [0.02, 0.01, 9.82],
                'angular_velocity': [0.002, 0.001, 0.004]
            }

        def set_joint_positions(self, positions):
            print(f"Updating sim joints: {len(positions)} joints")

        def set_imu_data(self, imu_data):
            print("Updating sim IMU data")

        def set_force_torque_data(self, ft_data):
            print("Updating sim force/torque data")

        def interpret_sensory_data(self, real_data):
            return {'object_detected': True, 'terrain_analysis': {'type': 'flat'}}

    # Create mock interfaces
    mock_robot = MockRobotInterface()
    mock_sim = MockSimulationEnvironment()

    # Create integration system
    integration = RealToSimIntegration(mock_robot, mock_sim)

    # Connect to robot
    if integration.connect_to_robot():
        print("\n1. Running mixed reality session...")
        integration.run_mixed_reality_session(duration=5.0)

        print("\n2. Getting synchronized state...")
        state = integration.get_synchronized_state()
        print(f"Synchronized state retrieved: {len(state)} keys")

        print("\n3. Switching control modes...")
        integration.switch_control_mode('teleoperation')
        integration.switch_control_mode('autonomous')

        # Disconnect
        integration.disconnect()
        print("\nReal-to-sim integration example completed")
    else:
        print("Failed to connect to robot for integration example")

if __name__ == "__main__":
    real_to_sim_integration_example()
```

## AI Training Integration

### Simulation-to-Training Pipeline

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import gzip
from datetime import datetime
import os

class SimulationDataset(Dataset):
    """
    Dataset for training AI models using simulation data
    """
    def __init__(self, data_path, transform=None):
        """
        Initialize simulation dataset

        Args:
            data_path: Path to simulation data file
            transform: Optional transform to be applied on samples
        """
        self.data_path = data_path
        self.transform = transform
        self.data = self.load_simulation_data()
        self.episode_indices = self.create_episode_indices()

    def load_simulation_data(self):
        """
        Load simulation data from file
        """
        try:
            with gzip.open(self.data_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded simulation data: {len(data)} samples")
            return data
        except Exception as e:
            print(f"Error loading simulation data: {e}")
            return []

    def create_episode_indices(self):
        """
        Create indices for each episode in the dataset
        """
        episode_indices = []
        current_episode_start = 0

        for i, sample in enumerate(self.data):
            if sample.get('episode_done', False):
                episode_indices.append((current_episode_start, i))
                current_episode_start = i + 1

        # Add the last episode if it wasn't terminated
        if current_episode_start < len(self.data):
            episode_indices.append((current_episode_start, len(self.data) - 1))

        print(f"Created {len(episode_indices)} episodes")
        return episode_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        """
        sample = self.data[idx]

        # Extract state, action, reward, next_state
        state = torch.FloatTensor(sample.get('state', np.zeros(100)))
        action = torch.FloatTensor(sample.get('action', np.zeros(10)))
        reward = torch.FloatTensor([sample.get('reward', 0.0)])
        next_state = torch.FloatTensor(sample.get('next_state', np.zeros(100)))
        done = torch.BoolTensor([sample.get('done', False)])

        if self.transform:
            state = self.transform(state)
            next_state = self.transform(next_state)

        return {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

class AITrainingPipeline:
    """
    Pipeline for training AI models using simulation data
    """
    def __init__(self, model_class, config):
        """
        Initialize AI training pipeline

        Args:
            model_class: Class of the neural network model
            config: Configuration dictionary
        """
        self.model_class = model_class
        self.config = config

        # Initialize model
        self.model = self.create_model()
        self.target_model = self.create_model()  # For DQN-style algorithms
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.criterion = nn.MSELoss()

        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.training_history = []

        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.target_model.to(self.device)

        # Update target model
        self.update_target_model()

    def create_model(self):
        """
        Create neural network model instance
        """
        # This would typically be a more complex model based on the task
        # For example: Policy network, Value network, or Actor-Critic
        return self.model_class(**self.config.get('model_params', {}))

    def update_target_model(self):
        """
        Update target model with current model weights
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def train_epoch(self, dataloader):
        """
        Train for one epoch

        Args:
            dataloader: DataLoader for training data

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move data to device
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            next_states = batch['next_state'].to(self.device)
            dones = batch['done'].to(self.device)

            # Forward pass
            current_q_values = self.model(states)

            # Calculate target Q values (for DQN-style training)
            with torch.no_grad():
                next_q_values = self.target_model(next_states)
                max_next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]
                target_q_values = rewards + (self.config.get('gamma', 0.99) * max_next_q_values * (1 - dones.float()))

            # Select Q values for taken actions
            action_indices = torch.argmax(actions, dim=1, keepdim=True)
            current_q_selected = torch.gather(current_q_values, 1, action_indices)

            # Calculate loss
            loss = self.criterion(current_q_selected, target_q_values)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))

            # Update weights
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def validate(self, dataloader):
        """
        Validate the model

        Args:
            dataloader: DataLoader for validation data

        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                states = batch['state'].to(self.device)
                actions = batch['action'].to(self.device)
                rewards = batch['reward'].to(self.device)
                next_states = batch['next_state'].to(self.device)
                dones = batch['done'].to(self.device)

                current_q_values = self.model(states)

                # Calculate target Q values
                next_q_values = self.target_model(next_states)
                max_next_q_values = torch.max(next_q_values, dim=1, keepdim=True)[0]
                target_q_values = rewards + (self.config.get('gamma', 0.99) * max_next_q_values * (1 - dones.float()))

                action_indices = torch.argmax(actions, dim=1, keepdim=True)
                current_q_selected = torch.gather(current_q_values, 1, action_indices)

                loss = self.criterion(current_q_selected, target_q_values)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return {'avg_loss': avg_loss}

    def train(self, train_dataset, val_dataset=None, num_epochs=100):
        """
        Train the AI model

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            num_epochs: Number of training epochs
        """
        print(f"Starting training for {num_epochs} epochs...")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                num_workers=self.config.get('num_workers', 4)
            )

        # Training loop
        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train one epoch
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader)

            # Update target model periodically
            if epoch % self.config.get('target_update_freq', 10) == 0:
                self.update_target_model()

            # Record metrics
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_metrics.get('avg_loss', 0),
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(metrics)

            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_metrics.get('avg_loss', 0):.4f}")

            # Save best model
            if val_metrics.get('avg_loss', train_loss) < self.best_loss:
                self.best_loss = val_metrics.get('avg_loss', train_loss)
                self.save_model(f"best_model_epoch_{epoch}.pth")

        print("Training completed!")

    def save_model(self, filepath):
        """
        Save the trained model

        Args:
            filepath: Path to save the model
        """
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': self.config
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model

        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        self.config = checkpoint['config']

        print(f"Model loaded from {filepath}")

    def evaluate_model(self, test_dataset):
        """
        Evaluate the model on test data

        Args:
            test_dataset: Test dataset

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False
        )

        total_reward = 0
        num_episodes = 0
        episode_rewards = []

        with torch.no_grad():
            for batch in test_loader:
                states = batch['state'].to(self.device)

                # Get model predictions
                q_values = self.model(states)
                actions = torch.argmax(q_values, dim=1)

                # Calculate episode rewards (simplified)
                rewards = batch['reward']
                total_reward += rewards.sum().item()
                num_episodes += len(rewards)
                episode_rewards.extend(rewards.flatten().tolist())

        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0
        std_reward = np.std(episode_rewards) if episode_rewards else 0

        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'num_episodes': num_episodes,
            'total_reward': total_reward
        }

class PolicyNetwork(nn.Module):
    """
    Policy network for humanoid robot control
    """
    def __init__(self, state_dim=100, action_dim=10, hidden_dim=256):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are bounded to [-1, 1]
        )

    def forward(self, state):
        return self.network(state)

def ai_training_pipeline_example():
    """
    Example of AI training pipeline
    """
    print("=== AI Training Pipeline Example ===")

    # Configuration for training
    config = {
        'learning_rate': 1e-4,
        'batch_size': 64,
        'num_workers': 4,
        'gamma': 0.99,
        'target_update_freq': 10,
        'max_grad_norm': 1.0,
        'model_params': {
            'state_dim': 100,
            'action_dim': 10,
            'hidden_dim': 256
        }
    }

    # Create mock dataset (in reality, this would be loaded from simulation data)
    print("Creating mock simulation dataset...")

    # Generate mock simulation data
    mock_data = []
    for episode in range(10):  # 10 episodes
        episode_length = np.random.randint(50, 200)  # Random episode length
        for step in range(episode_length):
            sample = {
                'state': np.random.randn(100).astype(np.float32),
                'action': np.random.randn(10).astype(np.float32),
                'reward': np.random.uniform(-1, 1).astype(np.float32),
                'next_state': np.random.randn(100).astype(np.float32),
                'done': step == episode_length - 1  # Done at end of episode
            }
            mock_data.append(sample)

    # Save mock data to simulate loading from file
    with gzip.open('mock_simulation_data.pkl.gz', 'wb') as f:
        pickle.dump(mock_data, f)

    # Create datasets
    train_dataset = SimulationDataset('mock_simulation_data.pkl.gz')
    val_dataset = SimulationDataset('mock_simulation_data.pkl.gz')  # Use same for example

    # Create training pipeline
    pipeline = AITrainingPipeline(PolicyNetwork, config)

    # Train the model
    print("\nStarting training...")
    pipeline.train(train_dataset, val_dataset, num_epochs=20)

    # Evaluate the model
    print("\nEvaluating model...")
    eval_metrics = pipeline.evaluate_model(val_dataset)
    print(f"Evaluation results: {eval_metrics}")

    # Save the model
    pipeline.save_model('humanoid_robot_policy.pth')

    # Clean up mock data
    os.remove('mock_simulation_data.pkl.gz')

    print("\nAI training pipeline example completed!")

if __name__ == "__main__":
    ai_training_pipeline_example()
```

## Performance Optimization

### Efficient Simulation Techniques

```python
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import wraps
import cProfile
import pstats
from io import StringIO

class PerformanceOptimizer:
    """
    Class for optimizing simulation performance
    """
    def __init__(self):
        self.performance_metrics = {
            'simulation_steps_per_second': 0,
            'cpu_usage': 0,
            'memory_usage': 0,
            'render_time': 0,
            'physics_time': 0,
            'total_time': 0
        }
        self.profiling_enabled = False
        self.profile_stats = None

    def profile_simulation_step(func):
        """
        Decorator for profiling simulation steps
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.profiling_enabled:
                return func(self, *args, **kwargs)

            # Start profiling
            pr = cProfile.Profile()
            pr.enable()

            # Execute function
            result = func(self, *args, **kwargs)

            # Stop profiling
            pr.disable()

            # Store stats
            s = StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions
            self.profile_stats = s.getvalue()

            return result
        return wrapper

    def benchmark_simulation(self, simulation_func, *args, **kwargs):
        """
        Benchmark a simulation function

        Args:
            simulation_func: Function to benchmark
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function

        Returns:
            Dictionary with benchmark results
        """
        # Monitor system resources
        start_cpu_percent = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().percent

        start_time = time.time()

        # Run simulation
        result = simulation_func(*args, **kwargs)

        end_time = time.time()

        # Monitor resources after
        end_cpu_percent = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().percent

        benchmark_results = {
            'execution_time': end_time - start_time,
            'cpu_usage_avg': (start_cpu_percent + end_cpu_percent) / 2,
            'memory_usage_avg': (start_memory + end_memory) / 2,
            'throughput': len(args) / (end_time - start_time) if args else 0  # Simplified
        }

        return benchmark_results

    def optimize_physics_simulation(self, simulation_step_func):
        """
        Optimize physics simulation by adjusting parameters

        Args:
            simulation_step_func: Function that performs one simulation step

        Returns:
            Optimized function
        """
        def optimized_step(*args, **kwargs):
            # Use multithreading for parallel physics calculations
            with ThreadPoolExecutor(max_workers=4) as executor:
                # This is a simplified example - in reality, physics engines
                # have their own optimization strategies
                result = simulation_step_func(*args, **kwargs)
            return result

        return optimized_step

    def parallel_simulation_execution(self, simulation_instances, steps_per_instance=100):
        """
        Execute multiple simulation instances in parallel

        Args:
            simulation_instances: List of simulation instances
            steps_per_instance: Number of steps to run per instance

        Returns:
            List of results from each simulation instance
        """
        def run_simulation_instance(instance_data):
            sim_instance, steps = instance_data
            results = []

            for step in range(steps):
                # Run simulation step
                step_result = sim_instance.step()
                results.append(step_result)

                # Yield control periodically
                if step % 10 == 0:
                    time.sleep(0.001)  # Small yield to prevent blocking

            return results

        # Prepare data for parallel execution
        instance_data = [(instance, steps_per_instance) for instance in simulation_instances]

        # Execute in parallel
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            results = list(executor.map(run_simulation_instance, instance_data))

        return results

    def adaptive_time_stepping(self, base_time_step=1/240, target_frequency=240):
        """
        Implement adaptive time stepping based on performance

        Args:
            base_time_step: Base time step (seconds)
            target_frequency: Target simulation frequency (Hz)

        Returns:
            Function decorator for adaptive time stepping
        """
        def adaptive_decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                start_time = time.time()

                # Execute the function
                result = func(self, *args, **kwargs)

                # Calculate actual execution time
                execution_time = time.time() - start_time

                # Adjust time step based on performance
                target_time = 1.0 / target_frequency
                if execution_time > target_time * 1.1:  # 10% over target
                    # Slow down simulation to maintain stability
                    adjusted_time_step = base_time_step * 0.9
                elif execution_time < target_time * 0.9:  # 10% under target
                    # Can speed up slightly
                    adjusted_time_step = min(base_time_step * 1.1, base_time_step * 2)
                else:
                    # Performance is good, maintain current time step
                    adjusted_time_step = base_time_step

                # Store performance metrics
                self.performance_metrics['physics_time'] = execution_time
                self.performance_metrics['adaptive_time_step'] = adjusted_time_step

                return result
            return wrapper
        return adaptive_decorator

    def memory_optimized_data_processing(self, data_generator, batch_size=1000):
        """
        Process large amounts of simulation data efficiently

        Args:
            data_generator: Generator that yields simulation data
            batch_size: Size of batches to process

        Yields:
            Processed data batches
        """
        batch = []

        for data in data_generator:
            batch.append(data)

            if len(batch) >= batch_size:
                # Process batch efficiently
                processed_batch = self.process_data_batch(batch)
                yield processed_batch
                batch = []  # Clear batch to free memory

        # Process remaining data
        if batch:
            processed_batch = self.process_data_batch(batch)
            yield processed_batch

    def process_data_batch(self, batch):
        """
        Process a batch of simulation data efficiently

        Args:
            batch: List of data items to process

        Returns:
            Processed batch
        """
        # Convert to NumPy arrays for efficient processing
        if batch and isinstance(batch[0], dict):
            # Handle dictionary-based data
            processed = {}
            for key in batch[0].keys():
                values = [item[key] for item in batch]
                if isinstance(values[0], (int, float)):
                    processed[key] = np.array(values)
                else:
                    processed[key] = values
        else:
            # Handle other data types
            processed = np.array(batch) if batch else np.array([])

        return processed

    def get_performance_report(self):
        """
        Generate performance optimization report

        Returns:
            Performance report dictionary
        """
        report = {
            'timestamp': time.time(),
            'performance_metrics': self.performance_metrics.copy(),
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available
            },
            'optimization_suggestions': self.generate_optimization_suggestions()
        }

        if self.profile_stats:
            report['profile_stats'] = self.profile_stats

        return report

    def generate_optimization_suggestions(self):
        """
        Generate suggestions for performance optimization

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # CPU usage suggestions
        if self.performance_metrics['cpu_usage'] > 80:
            suggestions.append("High CPU usage detected - consider parallelization or algorithm optimization")

        # Memory usage suggestions
        if self.performance_metrics['memory_usage'] > 80:
            suggestions.append("High memory usage detected - implement memory pooling or data streaming")

        # Performance suggestions
        if self.performance_metrics['simulation_steps_per_second'] < 100:
            suggestions.append("Low simulation throughput - consider time step optimization or parallel execution")

        # Default suggestions
        if not suggestions:
            suggestions.append("Performance looks good! Consider running benchmarks to establish baselines")

        return suggestions

def performance_optimization_example():
    """
    Example of performance optimization techniques
    """
    print("=== Performance Optimization Example ===")

    # Create optimizer
    optimizer = PerformanceOptimizer()

    # Example: Benchmark a simple simulation step
    def mock_simulation_step():
        """Mock simulation step for benchmarking"""
        # Simulate some computation
        result = 0
        for i in range(1000000):
            result += i * 0.001
        return result

    print("Benchmarking simulation step...")
    benchmark_results = optimizer.benchmark_simulation(mock_simulation_step)
    print(f"Benchmark results: {benchmark_results}")

    # Example: Demonstrate adaptive time stepping
    class MockSimulation:
        def __init__(self):
            self.time_step = 1/240

        @optimizer.adaptive_time_stepping(base_time_step=1/240, target_frequency=240)
        def step(self):
            """Simulation step with adaptive time stepping"""
            # Simulate physics update
            time.sleep(0.001)  # Simulate computation time
            return "step_complete"

    print("\nTesting adaptive time stepping...")
    mock_sim = MockSimulation()
    for i in range(5):
        result = mock_sim.step()
        print(f"Step {i+1}: {result}, time_step: {mock_sim.time_step:.6f}")

    # Example: Demonstrate memory-optimized data processing
    def mock_data_generator():
        """Mock data generator for testing"""
        for i in range(100):
            yield {'step': i, 'data': np.random.rand(100), 'value': i * 0.1}

    print("\nTesting memory-optimized data processing...")
    processed_count = 0
    for batch in optimizer.memory_optimized_data_processing(mock_data_generator(), batch_size=20):
        processed_count += len(batch['step']) if isinstance(batch, dict) else len(batch)
        print(f"Processed batch with {len(batch['step']) if isinstance(batch, dict) else len(batch)} items")

    print(f"Total processed items: {processed_count}")

    # Generate performance report
    report = optimizer.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"  - System CPUs: {report['system_info']['cpu_count']}")
    print(f"  - Memory: {report['system_info']['memory_total'] / (1024**3):.1f} GB total")
    print(f"  - Suggestions: {len(report['optimization_suggestions'])}")

    print("\nPerformance optimization example completed!")

if __name__ == "__main__":
    performance_optimization_example()
```

## Deployment and Production Considerations

### Production-Ready Simulation System

```python
import logging
import sys
from logging.handlers import RotatingFileHandler
import traceback
from contextlib import contextmanager
import signal
import atexit
from pathlib import Path

class ProductionSimulationSystem:
    """
    Production-ready simulation system with proper error handling,
    logging, and deployment considerations
    """
    def __init__(self, config_path="config.yaml", log_level=logging.INFO):
        """
        Initialize production simulation system

        Args:
            config_path: Path to configuration file
            log_level: Logging level
        """
        self.config = self.load_configuration(config_path)
        self.logger = self.setup_logging(log_level)
        self.health_check_interval = self.config.get('health_check_interval', 60)  # seconds
        self.error_recovery_enabled = self.config.get('enable_error_recovery', True)
        self.max_restart_attempts = self.config.get('max_restart_attempts', 3)

        # System state
        self.running = False
        self.restart_count = 0
        self.last_error = None

        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.logger.info("Production simulation system initialized")

    def load_configuration(self, config_path):
        """
        Load configuration from file

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        # In a real implementation, you'd load from YAML/JSON
        # For this example, we'll create a default configuration
        default_config = {
            'simulation_steps_per_second': 240,
            'max_simulation_time': 3600,  # 1 hour
            'checkpoint_interval': 300,   # 5 minutes
            'backup_enabled': True,
            'log_level': 'INFO',
            'enable_error_recovery': True,
            'max_restart_attempts': 3,
            'health_check_interval': 60,
            'resource_limits': {
                'cpu_percent': 80,
                'memory_percent': 80,
                'disk_space_gb': 10
            }
        }

        # Try to load from file, use defaults if not found
        try:
            # This would actually load from the specified config file
            config = default_config  # Placeholder
            self.logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}, using defaults: {e}")
            config = default_config

        return config

    def setup_logging(self, level):
        """
        Setup logging system

        Args:
            level: Logging level

        Returns:
            Logger instance
        """
        logger = logging.getLogger('ProductionSimulation')
        logger.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler with rotation
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = RotatingFileHandler(
            log_dir / "simulation.log",
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def health_check(self):
        """
        Perform system health check

        Returns:
            Health status dictionary
        """
        import psutil

        health_status = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'process_count': len(psutil.pids()),
            'uptime': time.time() - getattr(self, '_start_time', time.time()),
            'errors_since_start': getattr(self, '_error_count', 0),
            'warnings_since_start': getattr(self, '_warning_count', 0)
        }

        # Check resource limits
        resource_limits = self.config.get('resource_limits', {})
        health_status['resource_issues'] = []

        if health_status['cpu_percent'] > resource_limits.get('cpu_percent', 80):
            health_status['resource_issues'].append('high_cpu_usage')

        if health_status['memory_percent'] > resource_limits.get('memory_percent', 80):
            health_status['resource_issues'].append('high_memory_usage')

        if health_status['disk_percent'] > (100 - resource_limits.get('disk_space_gb', 10)):
            health_status['resource_issues'].append('low_disk_space')

        # Log health status if there are issues
        if health_status['resource_issues']:
            self.logger.warning(f"Health issues detected: {health_status['resource_issues']}")

        return health_status

    def run_simulation_with_monitoring(self, simulation_func, duration=3600):
        """
        Run simulation with monitoring and error recovery

        Args:
            simulation_func: Simulation function to run
            duration: Duration to run simulation (seconds)
        """
        self.logger.info(f"Starting simulation with monitoring for {duration} seconds")
        self._start_time = time.time()
        self._error_count = 0
        self._warning_count = 0

        start_time = time.time()
        health_check_time = start_time

        try:
            self.running = True

            while self.running and (time.time() - start_time) < duration:
                try:
                    # Perform health check periodically
                    current_time = time.time()
                    if current_time - health_check_time > self.health_check_interval:
                        health_status = self.health_check()
                        health_check_time = current_time

                        # Log health status periodically
                        if current_time - start_time > 300:  # Every 5 minutes after startup
                            self.logger.info(f"Health status: CPU={health_status['cpu_percent']:.1f}%, "
                                           f"Memory={health_status['memory_percent']:.1f}%")

                    # Execute simulation step
                    simulation_func()

                    # Small delay to prevent busy waiting
                    time.sleep(0.001)

                except Exception as e:
                    self._error_count += 1
                    self.logger.error(f"Simulation error: {e}")
                    self.logger.debug(traceback.format_exc())

                    if self.error_recovery_enabled:
                        recovery_success = self.attempt_error_recovery(e)
                        if not recovery_success:
                            self.logger.error("Error recovery failed, shutting down")
                            break
                    else:
                        raise  # Re-raise if recovery disabled

        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal simulation error: {e}")
            self.logger.debug(traceback.format_exc())
        finally:
            self.running = False
            self.logger.info("Simulation finished")

    def attempt_error_recovery(self, error):
        """
        Attempt to recover from simulation error

        Args:
            error: The error that occurred

        Returns:
            True if recovery was successful, False otherwise
        """
        self.last_error = error

        if self.restart_count >= self.max_restart_attempts:
            self.logger.error(f"Maximum restart attempts ({self.max_restart_attempts}) reached")
            return False

        self.logger.info(f"Attempting recovery, attempt {self.restart_count + 1}")

        try:
            # Perform recovery steps
            self.perform_recovery_steps()

            # Reset error counter after successful recovery
            self.restart_count = 0
            self._error_count = max(0, self._error_count - 1)  # Reduce error count

            self.logger.info("Recovery successful")
            return True

        except Exception as recovery_error:
            self.restart_count += 1
            self.logger.error(f"Recovery failed: {recovery_error}")
            self.logger.debug(traceback.format_exc())

            if self.restart_count >= self.max_restart_attempts:
                self.logger.critical("Too many restart attempts, system unstable")
                return False

            # Wait before next recovery attempt
            time.sleep(2 ** self.restart_count)  # Exponential backoff
            return False

    def perform_recovery_steps(self):
        """
        Perform specific recovery steps
        """
        # Clear any problematic state
        self.clear_problematic_state()

        # Reset simulation to known good state
        self.reset_to_checkpoint()

        # Reinitialize critical components
        self.reinitialize_components()

        self.logger.info("Recovery steps completed")

    def clear_problematic_state(self):
        """
        Clear any state that might be causing problems
        """
        # Implementation would clear problematic simulation state
        pass

    def reset_to_checkpoint(self):
        """
        Reset simulation to last known good checkpoint
        """
        # Implementation would load from last checkpoint
        pass

    def reinitialize_components(self):
        """
        Reinitialize critical simulation components
        """
        # Implementation would reinitialize components
        pass

    def signal_handler(self, signum, frame):
        """
        Handle system signals for graceful shutdown

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.running = False

    def cleanup(self):
        """
        Cleanup resources on shutdown
        """
        if hasattr(self, 'running') and self.running:
            self.logger.info("Performing cleanup during shutdown")

            # Save current state
            self.save_checkpoint()

            # Close any open files/resources
            # (Implementation would close specific resources)

            self.logger.info("Cleanup completed")

    def save_checkpoint(self):
        """
        Save current simulation state to checkpoint
        """
        # Implementation would save current state
        checkpoint_time = time.time()
        self.logger.info(f"Checkpoint saved at {checkpoint_time}")

    def get_system_status(self):
        """
        Get current system status

        Returns:
            System status dictionary
        """
        return {
            'running': self.running,
            'restart_count': self.restart_count,
            'last_error': str(self.last_error) if self.last_error else None,
            'start_time': getattr(self, '_start_time', None),
            'uptime': time.time() - getattr(self, '_start_time', time.time()) if hasattr(self, '_start_time') else 0,
            'error_count': getattr(self, '_error_count', 0),
            'warning_count': getattr(self, '_warning_count', 0)
        }

def production_simulation_example():
    """
    Example of production simulation system
    """
    print("=== Production Simulation System Example ===")

    # Create production simulation system
    prod_system = ProductionSimulationSystem()

    # Mock simulation function
    def mock_simulation_step():
        """Mock simulation step"""
        # Simulate normal operation
        time.sleep(0.01)  # 10ms per step (100Hz)

        # Occasionally simulate an error for recovery testing
        if time.time() % 30 < 0.1:  # Every 30 seconds, simulate error
            raise Exception("Simulated simulation error for testing")

    print("Running production simulation with monitoring...")

    try:
        # Run simulation for 60 seconds
        prod_system.run_simulation_with_monitoring(mock_simulation_step, duration=60)
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    except Exception as e:
        print(f"Simulation error: {e}")

    # Get system status
    status = prod_system.get_system_status()
    print(f"\nSystem Status:")
    print(f"  - Running: {status['running']}")
    print(f"  - Restart count: {status['restart_count']}")
    print(f"  - Uptime: {status['uptime']:.1f}s")
    print(f"  - Errors: {status['error_count']}")
    print(f"  - Warnings: {status['warning_count']}")

    print("\nProduction simulation example completed!")

if __name__ == "__main__":
    production_simulation_example()
```

## Summary

Integration examples and best practices for AI humanoid robotics simulation include:

1. **Real-to-Sim Integration**: Seamless synchronization between real robots and simulation environments
2. **AI Training Pipelines**: Efficient systems for training AI models using simulation data
3. **Performance Optimization**: Techniques for maximizing simulation efficiency and throughput
4. **Production Considerations**: Robust systems for deployment with proper error handling and monitoring

These integration patterns ensure that simulation systems can effectively support the development, training, and deployment of AI-powered humanoid robots in real-world applications.