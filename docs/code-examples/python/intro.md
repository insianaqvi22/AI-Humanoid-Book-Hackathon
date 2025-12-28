---
title: Python Examples for AI Humanoid Robotics
sidebar_position: 1
description: Python code examples for AI humanoid robotics implementation
---

# Python Examples for AI Humanoid Robotics

This section provides practical Python code examples for implementing various aspects of AI humanoid robotics. These examples demonstrate how to implement the concepts discussed in the book using Python, which is widely used in robotics and AI development.

## Setting Up the Environment

Before diving into the examples, you'll need to set up your Python environment with the necessary libraries:

```bash
pip install numpy scipy matplotlib torch torchvision tensorflow opencv-python
pip install robotframework pybullet transforms3d
pip install speech-recognition face-recognition pygame
pip install transformers torchtext datasets
```

## Basic Movement Control Example

Let's start with a basic example of controlling robot movements:

```python
import numpy as np
import math
import time

class RobotController:
    """
    A basic controller for humanoid robot movements
    """
    def __init__(self, robot_model="nao", simulation=True):
        self.robot_model = robot_model
        self.simulation = simulation
        self.joint_positions = {}
        self.joint_limits = self.define_joint_limits()

    def define_joint_limits(self):
        """
        Define joint limits for the robot
        """
        if self.robot_model == "nao":
            return {
                'HeadYaw': (-2.0857, 2.0857),
                'HeadPitch': (-0.7068, 0.4469),
                'LShoulderPitch': (-2.0857, 2.0857),
                'LShoulderRoll': (-0.3142, 1.3265),
                'LElbowYaw': (-2.0857, 2.0857),
                'LElbowRoll': (-1.5446, -0.0349),
                'LWristYaw': (-1.8238, 1.8238),
                'LHipYawPitch': (-1.1453, 0.7408),
                'LHipRoll': (-0.3794, 0.7905),
                'LHipPitch': (-1.5358, 0.4840),
                'LKneePitch': (-0.0923, 2.1125),
                'LAnklePitch': (-1.1895, 0.9329),
                'LAnkleRoll': (-0.3900, 0.7408)
            }
        else:
            # Default limits for generic humanoid
            return {
                'hip_pitch': (-1.57, 1.57),
                'knee_pitch': (0, 2.35),
                'ankle_pitch': (-0.5, 0.5),
                'shoulder_pitch': (-2.09, 1.57),
                'elbow_pitch': (-2.35, 0),
                'wrist_yaw': (-1.57, 1.57)
            }

    def move_to_position(self, joint_name, target_position, duration=1.0):
        """
        Move a specific joint to a target position over a given duration

        Args:
            joint_name (str): Name of the joint to move
            target_position (float): Target position in radians
            duration (float): Duration of the movement in seconds
        """
        if joint_name not in self.joint_limits:
            raise ValueError(f"Unknown joint: {joint_name}")

        # Check joint limits
        min_limit, max_limit = self.joint_limits[joint_name]
        if target_position < min_limit or target_position > max_limit:
            print(f"Warning: Position {target_position} exceeds limits for {joint_name}")
            target_position = max(min_limit, min(max_limit, target_position))

        # Get current position
        current_position = self.joint_positions.get(joint_name, 0.0)

        # Calculate movement parameters
        steps = int(duration * 100)  # 100Hz control rate
        delta = target_position - current_position

        # Execute movement with smooth interpolation
        for i in range(steps + 1):
            t = i / steps  # Normalized time (0 to 1)

            # Use cubic interpolation for smooth motion
            # This creates a smooth S-curve for acceleration/deceleration
            smooth_t = 3 * t**2 - 2 * t**3

            intermediate_position = current_position + smooth_t * delta

            # Update joint position
            self.joint_positions[joint_name] = intermediate_position

            # In simulation, just update internal state
            # In real robot, send command to actual joint
            if self.simulation:
                self.simulate_joint_movement(joint_name, intermediate_position)
            else:
                self.send_joint_command(joint_name, intermediate_position)

            time.sleep(duration / steps)

        print(f"Successfully moved {joint_name} to {target_position:.3f} radians")

    def simulate_joint_movement(self, joint_name, position):
        """
        Simulate joint movement (for demonstration purposes)
        """
        print(f"  {joint_name}: {position:.3f} rad", end='\r')

    def send_joint_command(self, joint_name, position):
        """
        Send actual command to robot joint (implementation depends on robot platform)
        """
        # This would interface with the actual robot
        pass

    def move_multiple_joints(self, joint_targets, duration=1.0):
        """
        Move multiple joints simultaneously

        Args:
            joint_targets (dict): Dictionary of {joint_name: target_position}
            duration (float): Duration of the movement in seconds
        """
        # Check all joints are valid
        for joint_name in joint_targets:
            if joint_name not in self.joint_limits:
                raise ValueError(f"Unknown joint: {joint_name}")

        # Get starting positions
        start_positions = {}
        for joint_name in joint_targets:
            start_positions[joint_name] = self.joint_positions.get(joint_name, 0.0)

        # Execute simultaneous movement
        steps = int(duration * 100)  # 100Hz control rate

        for i in range(steps + 1):
            t = i / steps  # Normalized time (0 to 1)
            smooth_t = 3 * t**2 - 2 * t**3  # Smooth interpolation

            for joint_name, target_position in joint_targets.items():
                start_pos = start_positions[joint_name]
                delta = target_position - start_pos
                intermediate_position = start_pos + smooth_t * delta

                # Update joint position
                self.joint_positions[joint_name] = intermediate_position

                # Send command (simulation or real)
                if self.simulation:
                    self.simulate_joint_movement(joint_name, intermediate_position)
                else:
                    self.send_joint_command(joint_name, intermediate_position)

            time.sleep(duration / steps)

        print("\nSuccessfully moved multiple joints")

    def get_current_positions(self):
        """
        Get current positions of all joints
        """
        return self.joint_positions.copy()

    def wave_hand(self, duration=2.0):
        """
        Execute a waving motion with the right arm
        """
        print("Executing wave hand motion...")

        # Define wave motion sequence
        wave_sequence = [
            {'RShoulderPitch': -0.5, 'RElbowRoll': -1.0, 'RWristYaw': 0.0},  # Raise arm
            {'RShoulderPitch': -0.5, 'RElbowRoll': -1.0, 'RWristYaw': 0.5},  # Wave right
            {'RShoulderPitch': -0.5, 'RElbowRoll': -1.0, 'RWristYaw': -0.5}, # Wave left
            {'RShoulderPitch': -0.5, 'RElbowRoll': -1.0, 'RWristYaw': 0.5},  # Wave right
            {'RShoulderPitch': -0.5, 'RElbowRoll': -1.0, 'RWristYaw': 0.0},  # Return to center
            {'RShoulderPitch': 0.0, 'RElbowRoll': 0.0, 'RWristYaw': 0.0}     # Lower arm
        ]

        # Execute wave sequence
        for i, target in enumerate(wave_sequence):
            print(f"Wave motion step {i+1}/{len(wave_sequence)}")
            self.move_multiple_joints(target, duration=duration/len(wave_sequence))
            time.sleep(0.2)  # Small pause between steps

    def bow(self, duration=1.0):
        """
        Execute a bowing motion
        """
        print("Executing bow motion...")

        # Bow forward
        bow_targets = {'LHipPitch': -0.5, 'RHipPitch': -0.5, 'LAnklePitch': 0.3, 'RAnklePitch': 0.3}
        self.move_multiple_joints(bow_targets, duration=duration)

        # Hold the bow
        time.sleep(1.0)

        # Return to standing position
        stand_targets = {'LHipPitch': 0.0, 'RHipPitch': 0.0, 'LAnklePitch': 0.0, 'RAnklePitch': 0.0}
        self.move_multiple_joints(stand_targets, duration=duration)

        print("Bow completed")

# Example usage
if __name__ == "__main__":
    print("=== Humanoid Robot Controller Example ===")

    # Create robot controller instance
    robot = RobotController(simulation=True)

    print("\n1. Moving head to look left...")
    robot.move_to_position('HeadYaw', 0.5, duration=1.0)

    print("\n2. Moving head to look right...")
    robot.move_to_position('HeadYaw', -0.5, duration=1.0)

    print("\n3. Returning head to center...")
    robot.move_to_position('HeadYaw', 0.0, duration=1.0)

    print("\n4. Waving hand...")
    robot.wave_hand(duration=3.0)

    print("\n5. Bowing...")
    robot.bow(duration=1.0)

    print("\n6. Current joint positions:")
    positions = robot.get_current_positions()
    for joint, position in positions.items():
        print(f"  {joint}: {position:.3f} radians")
```

## Computer Vision Example

Here's an example of implementing computer vision for object detection:

```python
import cv2
import numpy as np
import time

class VisionSystem:
    """
    Computer vision system for humanoid robot
    """
    def __init__(self):
        self.cap = None
        self.object_detector = self.initialize_object_detector()

    def initialize_object_detector(self):
        """
        Initialize object detection model (using OpenCV DNN module)
        """
        # For this example, we'll use a pre-trained MobileNet SSD
        # In practice, you might use YOLO, SSD, or other models
        try:
            # Load the model (you'll need to download these files)
            net = cv2.dnn.readNetFromCaffe(
                'deploy.prototxt',  # Model configuration
                'mobilenet_iter_73000.caffemodel'  # Pre-trained weights
            )
            return net
        except:
            print("Warning: Could not load pre-trained model. Using basic detection.")
            return None

    def start_camera(self, camera_id=0):
        """
        Start camera capture
        """
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("Camera started successfully")

    def detect_objects(self, image):
        """
        Detect objects in image using computer vision
        """
        if self.object_detector is None:
            # Fallback to basic color-based detection
            return self.basic_color_detection(image)

        # Create blob from image
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # Pass blob through network
        self.object_detector.setInput(blob)
        detections = self.object_detector.forward()

        # Process detections
        objects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter by confidence
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])

                # Calculate bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Get label (simplified - in reality, you'd have a class mapping)
                label = f"Object {idx}"

                objects.append({
                    'label': label,
                    'confidence': confidence,
                    'bbox': (startX, startY, endX, endY),
                    'center': ((startX + endX) // 2, (startY + endY) // 2)
                })

        return objects

    def basic_color_detection(self, image):
        """
        Basic color-based object detection as fallback
        """
        # Convert to HSV for easier color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges (in HSV)
        color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255])
        }

        objects = []

        for color_name, (lower, upper) in color_ranges.items():
            # Create mask for this color
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by minimum area
                if area > 500:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    objects.append({
                        'label': color_name,
                        'confidence': 0.7,  # Assumed confidence for color detection
                        'bbox': (x, y, x+w, y+h),
                        'center': (x + w//2, y + h//2)
                    })

        return objects

    def detect_faces(self, image):
        """
        Detect faces in image using OpenCV's Haar cascade
        """
        # Load face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_objects = []
        for (x, y, w, h) in faces:
            face_objects.append({
                'label': 'face',
                'confidence': 0.9,  # High confidence for face detection
                'bbox': (x, y, x+w, y+h),
                'center': (x + w//2, y + h//2)
            })

        return face_objects

    def process_video_stream(self, callback=None):
        """
        Process video stream continuously

        Args:
            callback: Function to call with each processed frame
        """
        if self.cap is None:
            raise RuntimeError("Camera not started. Call start_camera() first.")

        print("Starting video processing... Press 'q' to quit")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            # Process frame
            objects = self.detect_objects(frame)
            faces = self.detect_faces(frame)

            # Draw detections on frame
            processed_frame = self.draw_detections(frame, objects, faces)

            # Call callback if provided
            if callback:
                callback(processed_frame, objects, faces)

            # Display frame
            cv2.imshow('Robot Vision', processed_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()

    def draw_detections(self, image, objects, faces):
        """
        Draw detection results on image
        """
        output_image = image.copy()

        # Draw object detections
        for obj in objects:
            bbox = obj['bbox']
            label = f"{obj['label']}: {obj['confidence']:.2f}"

            # Draw bounding box
            cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Draw label
            cv2.putText(output_image, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw face detections
        for face in faces:
            bbox = face['bbox']
            label = f"{face['label']}: {face['confidence']:.2f}"

            # Draw bounding box (different color for faces)
            cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # Draw label
            cv2.putText(output_image, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return output_image

# Example usage for vision system
def vision_example():
    print("=== Robot Vision System Example ===")

    # Create vision system
    vision = VisionSystem()

    # Start camera (uncomment to run with actual camera)
    # vision.start_camera()

    # For demonstration, we'll create a sample image
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add some colored rectangles to simulate objects
    cv2.rectangle(sample_image, (100, 100), (200, 200), (0, 0, 255), -1)  # Red rectangle
    cv2.rectangle(sample_image, (300, 150), (400, 250), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(sample_image, (500, 300), 50, (0, 255, 0), -1)  # Green circle

    print("Sample image created with colored shapes")

    # Detect objects in sample image
    objects = vision.detect_objects(sample_image)
    faces = vision.detect_faces(sample_image)  # Will find none in this example

    print(f"\nDetected {len(objects)} objects:")
    for i, obj in enumerate(objects):
        print(f"  Object {i+1}: {obj['label']} at {obj['center']} with confidence {obj['confidence']:.2f}")

    print(f"\nDetected {len(faces)} faces")

    # Draw detections on image
    result_image = vision.draw_detections(sample_image, objects, faces)

    # Display result (uncomment to show image)
    # cv2.imshow('Detection Result', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run movement example
    print("Running movement controller example...")
    # The movement example is already defined above

    print("\n" + "="*50 + "\n")

    # Run vision example
    vision_example()
```

## AI Integration Example

Here's an example of integrating AI with the robot system:

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import json

class RobotAI:
    """
    AI system for humanoid robot decision making
    """
    def __init__(self):
        self.perception_model = self.create_perception_model()
        self.decision_model = self.create_decision_model()
        self.scaler = StandardScaler()
        self.action_history = []

    def create_perception_model(self):
        """
        Create a neural network for processing sensory input
        """
        class PerceptionModel(nn.Module):
            def __init__(self, input_size=20, output_size=10):
                super(PerceptionModel, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, output_size),
                    nn.Softmax(dim=1)  # For classification
                )

            def forward(self, x):
                return self.network(x)

        return PerceptionModel()

    def create_decision_model(self):
        """
        Create a neural network for decision making
        """
        class DecisionModel(nn.Module):
            def __init__(self, state_size=10, action_size=5):
                super(DecisionModel, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, action_size)
                )

            def forward(self, x):
                return self.network(x)

        return DecisionModel()

    def process_sensory_input(self, sensory_data):
        """
        Process sensory input through AI model

        Args:
            sensory_data: Dictionary containing sensor readings

        Returns:
            Processed perception output
        """
        # Convert sensory data to numerical features
        features = self.extract_features(sensory_data)

        # Normalize features
        features_scaled = self.scaler.fit_transform(features.reshape(1, -1))

        # Convert to tensor
        features_tensor = torch.FloatTensor(features_scaled)

        # Process through perception model
        with torch.no_grad():
            perception_output = self.perception_model(features_tensor)

        return perception_output.numpy()[0]

    def extract_features(self, sensory_data):
        """
        Extract numerical features from sensory data
        """
        features = np.zeros(20)  # 20 feature dimensions

        # Extract features from different sensors
        idx = 0

        # Joint positions (assuming up to 10 joints)
        if 'joint_positions' in sensory_data:
            positions = list(sensory_data['joint_positions'].values())
            for i, pos in enumerate(positions[:10]):  # Take first 10 joints
                features[idx + i] = pos
            idx += 10

        # IMU data (acceleration, gyroscope)
        if 'imu' in sensory_data:
            imu = sensory_data['imu']
            features[idx] = imu.get('accel_x', 0)
            features[idx + 1] = imu.get('accel_y', 0)
            features[idx + 2] = imu.get('accel_z', 0)
            features[idx + 3] = imu.get('gyro_x', 0)
            features[idx + 4] = imu.get('gyro_y', 0)
            features[idx + 5] = imu.get('gyro_z', 0)
            idx += 6

        # Force/torque sensors
        if 'force_sensors' in sensory_data:
            forces = list(sensory_data['force_sensors'].values())
            for i, force in enumerate(forces[:4]):  # Take first 4 force sensors
                features[idx + i] = force
            idx += 4

        # Fill remaining features with zeros if needed
        if idx < 20:
            features[idx:] = 0

        return features

    def make_decision(self, state_vector):
        """
        Make decision based on state vector

        Args:
            state_vector: Processed perception output

        Returns:
            Action probabilities or values
        """
        # Convert to tensor
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)

        # Get action values from decision model
        with torch.no_grad():
            action_values = self.decision_model(state_tensor)

        return action_values.numpy()[0]

    def select_action(self, action_values, method='greedy'):
        """
        Select action from action values

        Args:
            action_values: Values for each possible action
            method: Selection method ('greedy', 'softmax', 'epsilon_greedy')

        Returns:
            Selected action index
        """
        if method == 'greedy':
            return np.argmax(action_values)
        elif method == 'softmax':
            # Apply softmax to get probabilities
            exp_values = np.exp(action_values - np.max(action_values))  # For numerical stability
            probabilities = exp_values / np.sum(exp_values)
            return np.random.choice(len(action_values), p=probabilities)
        elif method == 'epsilon_greedy':
            epsilon = 0.1  # Exploration rate
            if np.random.random() < epsilon:
                return np.random.choice(len(action_values))
            else:
                return np.argmax(action_values)
        else:
            raise ValueError(f"Unknown selection method: {method}")

    def update_model(self, state, action, reward, next_state, done=False):
        """
        Update AI model with experience (for learning)

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # This would implement learning algorithms like DQN, A3C, etc.
        # For this example, we'll just store the experience
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        self.action_history.append(experience)

        # If we have enough experiences, we could train the model
        if len(self.action_history) >= 1000:
            self.train_model()

    def train_model(self):
        """
        Train the AI models with collected experiences
        """
        print("Training AI models with collected experiences...")

        # This would implement actual training
        # For this example, we'll just print a message
        print(f"Trained on {len(self.action_history)} experiences")

        # Clear old experiences to prevent memory issues
        if len(self.action_history) > 10000:
            self.action_history = self.action_history[-5000:]  # Keep last 5000

class RobotWithAI:
    """
    Robot class that integrates AI for decision making
    """
    def __init__(self):
        self.ai_system = RobotAI()
        self.controller = RobotController(simulation=True)
        self.action_names = [
            'move_forward',
            'move_backward',
            'turn_left',
            'turn_right',
            'wave_hand'
        ]
        self.current_state = self.get_initial_state()

    def get_initial_state(self):
        """
        Get initial state for the robot
        """
        return {
            'joint_positions': self.controller.get_current_positions(),
            'imu': {'accel_x': 0, 'accel_y': 0, 'accel_z': 9.81,  # Gravity
                   'gyro_x': 0, 'gyro_y': 0, 'gyro_z': 0},
            'force_sensors': {'left_foot': 0, 'right_foot': 0, 'left_hand': 0, 'right_hand': 0}
        }

    def sense_environment(self):
        """
        Sense the current environment
        """
        # In a real robot, this would interface with actual sensors
        # For simulation, we'll return a modified current state
        state = self.current_state.copy()

        # Simulate some environmental changes
        # For example, add some noise to IMU readings
        import random
        state['imu']['accel_x'] += random.uniform(-0.1, 0.1)
        state['imu']['accel_y'] += random.uniform(-0.1, 0.1)
        state['imu']['gyro_z'] += random.uniform(-0.05, 0.05)

        return state

    def execute_action(self, action_index):
        """
        Execute the selected action

        Args:
            action_index: Index of the action to execute
        """
        action_name = self.action_names[action_index]
        print(f"Executing action: {action_name}")

        # Execute different actions
        if action_name == 'move_forward':
            # Move forward (simulate with joint movements)
            self.controller.move_to_position('LHipPitch', -0.1, 0.5)
            self.controller.move_to_position('RHipPitch', -0.1, 0.5)
            time.sleep(0.5)
            self.controller.move_to_position('LHipPitch', 0.0, 0.5)
            self.controller.move_to_position('RHipPitch', 0.0, 0.5)

        elif action_name == 'move_backward':
            # Move backward
            self.controller.move_to_position('LHipPitch', 0.1, 0.5)
            self.controller.move_to_position('RHipPitch', 0.1, 0.5)
            time.sleep(0.5)
            self.controller.move_to_position('LHipPitch', 0.0, 0.5)
            self.controller.move_to_position('RHipPitch', 0.0, 0.5)

        elif action_name == 'turn_left':
            # Turn left
            self.controller.move_to_position('LHipRoll', 0.1, 0.5)
            self.controller.move_to_position('RHipRoll', -0.1, 0.5)
            time.sleep(0.5)
            self.controller.move_to_position('LHipRoll', 0.0, 0.5)
            self.controller.move_to_position('RHipRoll', 0.0, 0.5)

        elif action_name == 'turn_right':
            # Turn right
            self.controller.move_to_position('LHipRoll', -0.1, 0.5)
            self.controller.move_to_position('RHipRoll', 0.1, 0.5)
            time.sleep(0.5)
            self.controller.move_to_position('LHipRoll', 0.0, 0.5)
            self.controller.move_to_position('RHipRoll', 0.0, 0.5)

        elif action_name == 'wave_hand':
            # Wave hand
            self.controller.wave_hand(duration=2.0)

        # Update current state after action
        self.current_state = self.sense_environment()

    def calculate_reward(self, action_index, next_state):
        """
        Calculate reward for the action taken

        Args:
            action_index: Index of the action taken
            next_state: State after the action

        Returns:
            Reward value
        """
        reward = 0.0

        # Positive rewards for stable behavior
        imu_data = next_state.get('imu', {})
        if abs(imu_data.get('accel_x', 0)) < 0.5 and abs(imu_data.get('accel_y', 0)) < 0.5:
            reward += 0.1  # Bonus for stability

        # Negative rewards for dangerous behavior
        if abs(imu_data.get('gyro_z', 0)) > 1.0:
            reward -= 0.5  # Penalty for spinning too fast

        # Small negative reward for each step (to encourage efficiency)
        reward -= 0.01

        return reward

    def run_interaction_loop(self, max_steps=50):
        """
        Run the main interaction loop with AI decision making
        """
        print("Starting AI-powered robot interaction loop...")

        for step in range(max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")

            # Sense environment
            current_state = self.sense_environment()
            print(f"Current state sensed")

            # Process through AI perception
            perception_output = self.ai_system.process_sensory_input(current_state)
            print(f"Perception processed: {perception_output[:5]}...")  # Show first 5 values

            # Make decision
            action_values = self.ai_system.make_decision(perception_output)
            action_index = self.ai_system.select_action(action_values, method='epsilon_greedy')
            action_name = self.action_names[action_index]

            print(f"AI decided: {action_name} (index {action_index})")

            # Execute action
            self.execute_action(action_index)

            # Get next state
            next_state = self.sense_environment()

            # Calculate reward
            reward = self.calculate_reward(action_index, next_state)
            print(f"Reward: {reward:.3f}")

            # Update AI model with experience
            self.ai_system.update_model(
                perception_output,  # State
                action_index,       # Action
                reward,             # Reward
                self.ai_system.process_sensory_input(next_state),  # Next state
                done=False          # Done flag
            )

            # Small pause between steps
            time.sleep(0.1)

        print(f"\nInteraction loop completed after {max_steps} steps")
        print(f"AI learned from {len(self.ai_system.action_history)} experiences")

# Example usage
if __name__ == "__main__":
    print("=== AI-Integrated Robot Example ===")

    # Create robot with AI
    robot_with_ai = RobotWithAI()

    # Run the interaction loop
    robot_with_ai.run_interaction_loop(max_steps=10)  # Reduced steps for demo
```

## Running the Examples

To run these examples, you'll need to have the required Python packages installed. You can install them using pip:

```bash
pip install numpy scipy matplotlib torch torchvision opencv-python
```

These examples demonstrate:
1. Basic robot movement control with smooth interpolation
2. Computer vision for object and face detection
3. AI integration for decision making
4. Sensor processing and action selection

The code is designed to be educational and demonstrates key concepts in humanoid robotics programming. In a real robot, you would connect these systems to actual hardware interfaces and sensors.