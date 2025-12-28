---
title: Sensors and Actuators
sidebar_position: 2
description: Sensing and actuation systems for AI humanoid robotics
---

# Sensors and Actuators

Sensors and actuators form the nervous system of humanoid robots, enabling them to perceive their environment and interact with it. This chapter covers the essential sensing and actuation technologies used in humanoid robotics, including selection criteria, integration challenges, and system design principles.

## Introduction to Robot Sensors and Actuators

The sensor-actuator system in humanoid robots enables:
- **Perception**: Understanding the environment through various sensors
- **Control**: Executing precise movements through actuators
- **Interaction**: Safe and effective engagement with humans and objects
- **Adaptation**: Responding to changing conditions and tasks

## Sensor Systems for Humanoid Robots

### Inertial Measurement Units (IMUs)

IMUs are critical for balance and motion control in humanoid robots.

**Components:**
- **Accelerometer**: Measures linear acceleration
- **Gyroscope**: Measures angular velocity
- **Magnetometer**: Measures magnetic field (compass)

**Applications:**
- Balance control and posture maintenance
- Motion tracking and navigation
- Fall detection and recovery
- Orientation estimation

```python
import numpy as np

class IMU:
    def __init__(self):
        self.accelerometer_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.orientation = np.array([0, 0, 0, 1])  # Quaternion [x, y, z, w]

    def read_sensors(self):
        # Simulate IMU readings (in real implementation, this would interface with hardware)
        accelerometer_raw = np.random.normal(0, 0.01, 3)  # m/s²
        gyroscope_raw = np.random.normal(0, 0.001, 3)    # rad/s
        magnetometer_raw = np.random.normal(0, 0.1, 3)   # µT

        # Apply bias correction
        accelerometer = accelerometer_raw - self.accelerometer_bias
        gyroscope = gyroscope_raw - self.gyro_bias

        return {
            'accelerometer': accelerometer,
            'gyroscope': gyroscope,
            'magnetometer': magnetometer_raw
        }

    def estimate_orientation(self, dt, gyro_reading, accel_reading):
        """
        Estimate orientation using gyroscope and accelerometer
        """
        # Simple complementary filter
        # Integrate gyroscope for rotation
        gyro_rotation = gyro_reading * dt
        gyro_quat = self.axis_angle_to_quaternion(gyro_rotation)

        # Update orientation
        new_orientation = self.quaternion_multiply(self.orientation, gyro_quat)

        # Correct with accelerometer when robot is not accelerating
        accel_quat = self.accelerometer_to_quaternion(accel_reading)
        # Blend the two estimates (simplified)
        self.orientation = self.orientation * 0.95 + accel_quat * 0.05
        self.orientation = self.orientation / np.linalg.norm(self.orientation)

        return self.orientation

    def axis_angle_to_quaternion(self, axis_angle):
        # Convert axis-angle representation to quaternion
        angle = np.linalg.norm(axis_angle)
        if angle == 0:
            return np.array([0, 0, 0, 1])

        axis = axis_angle / angle
        half_angle = angle / 2
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)

        return np.array([
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
            cos_half
        ])

    def quaternion_multiply(self, q1, q2):
        # Multiply two quaternions
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([x, y, z, w])
```

### Vision Sensors

Vision systems provide crucial environmental awareness.

**Camera Types:**
- **RGB Cameras**: Color imaging for object recognition
- **Depth Cameras**: Depth information for 3D understanding
- **Stereo Cameras**: 3D reconstruction capabilities
- **Event Cameras**: High-speed dynamic vision

**Applications:**
- Object recognition and tracking
- Human detection and recognition
- Navigation and obstacle avoidance
- Manipulation and grasping

```python
import cv2
import numpy as np

class RobotVisionSystem:
    def __init__(self, camera_matrix, distortion_coeffs):
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs

    def undistort_image(self, image):
        """Correct lens distortion"""
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)

    def detect_objects(self, image):
        """Detect objects in the image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2)
                })

        return objects

    def estimate_depth(self, left_image, right_image):
        """Estimate depth using stereo vision"""
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(
            cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        )
        return disparity
```

### Tactile Sensors

Tactile sensors enable safe and precise interaction.

**Types:**
- **Force/Torque Sensors**: Measure forces at joints
- **Pressure Sensors**: Detect contact and pressure distribution
- **Tactile Arrays**: High-resolution contact sensing
- **Proximity Sensors**: Detect nearby objects

**Applications:**
- Safe human-robot interaction
- Grasping and manipulation
- Surface exploration
- Contact detection and localization

```python
class TactileSensorArray:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.pressure_map = np.zeros((rows, cols))
        self.threshold = 0.1  # Minimum pressure to register contact

    def read_pressure(self):
        """Read pressure values from all sensors"""
        # In real implementation, this would interface with hardware
        # Simulate sensor readings with some noise
        self.pressure_map = np.random.normal(0, 0.01, (self.rows, self.cols))
        return self.pressure_map

    def detect_contacts(self):
        """Detect contact points above threshold"""
        contacts = np.where(self.pressure_map > self.threshold)
        contact_points = list(zip(contacts[0], contacts[1]))
        return contact_points

    def calculate_contact_force(self):
        """Calculate total contact force"""
        return np.sum(self.pressure_map[self.pressure_map > self.threshold])

    def find_contact_center(self):
        """Find center of pressure"""
        if np.sum(self.pressure_map) == 0:
            return None

        total_force = np.sum(self.pressure_map)
        row_indices, col_indices = np.mgrid[:self.rows, :self.cols]

        center_row = np.sum(row_indices * self.pressure_map) / total_force
        center_col = np.sum(col_indices * self.pressure_map) / total_force

        return (center_row, center_col)
```

### Proprioceptive Sensors

Proprioceptive sensors provide information about the robot's own state.

**Joint Sensors:**
- **Encoders**: Measure joint angles with high precision
- **Potentiometers**: Alternative angle measurement
- **Resolvers**: Robust angle sensors for harsh environments
- **Torque Sensors**: Measure forces in joints

**Applications:**
- Joint position control
- Motion planning and execution
- Collision detection
- System monitoring and diagnostics

```python
class JointSensor:
    def __init__(self, joint_name, encoder_resolution=4096):
        self.joint_name = joint_name
        self.encoder_resolution = encoder_resolution
        self.position = 0.0
        self.velocity = 0.0
        self.torque = 0.0
        self.position_offset = 0.0

    def read_position(self):
        """Read joint position from encoder"""
        # Simulate encoder reading (real implementation would interface with hardware)
        raw_encoder = np.random.randint(0, self.encoder_resolution)
        self.position = (raw_encoder / self.encoder_resolution) * 2 * np.pi - np.pi
        self.position += self.position_offset
        return self.position

    def read_velocity(self):
        """Estimate joint velocity"""
        # In real system, this might use a tachometer or velocity estimation
        # from position changes
        return self.velocity

    def read_torque(self):
        """Read joint torque (if available)"""
        # Simulate torque reading
        self.torque = np.random.normal(0, 0.1)  # Nm
        return self.torque

    def check_limits(self, min_pos, max_pos):
        """Check if joint is within safe limits"""
        return min_pos <= self.position <= max_pos
```

## Actuator Systems

### Types of Actuators

**Servo Motors:**
- Precise position control
- Integrated controllers
- Feedback systems
- Common in small to medium robots

**Brushless DC Motors:**
- High efficiency and power density
- Long lifespan
- Require external controllers
- Good for high-performance applications

**Series Elastic Actuators (SEAs):**
- Built-in compliance
- Force control capabilities
- Safe human interaction
- More complex but safer

**Pneumatic and Hydraulic Systems:**
- High power-to-weight ratio
- Natural compliance
- More complex control
- Used in some large robots

### Actuator Control

**Position Control:**
- PID control for precise positioning
- Trajectory following
- Smooth motion profiles
- Anti-windup and saturation handling

```python
class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-np.inf, np.inf)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral = 0
        self.previous_error = 0
        self.previous_time = None

    def compute(self, setpoint, measured_value, dt=None):
        """Compute control output using PID"""
        if dt is None:
            current_time = time.time()
            if self.previous_time is None:
                dt = 0.01  # Default time step
            else:
                dt = current_time - self.previous_time
            self.previous_time = current_time
        else:
            current_time = None

        error = setpoint - measured_value

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        # Anti-windup
        i_term = self.ki * self.integral

        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0
        d_term = self.kd * derivative

        # Store error for next iteration
        self.previous_error = error

        # Compute output
        output = p_term + i_term + d_term

        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return output

class JointController:
    def __init__(self, joint_name):
        self.joint_name = joint_name
        self.position_controller = PIDController(kp=10, ki=0.1, kd=0.5)
        self.velocity_controller = PIDController(kp=2, ki=0.05, kd=0.1)
        self.torque_controller = PIDController(kp=1, ki=0.01, kd=0.05)
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.current_torque = 0.0

    def control_position(self, target_position, current_position, dt):
        """Control joint position"""
        control_output = self.position_controller.compute(
            target_position, current_position, dt
        )
        return control_output

    def control_velocity(self, target_velocity, current_velocity, dt):
        """Control joint velocity"""
        control_output = self.velocity_controller.compute(
            target_velocity, current_velocity, dt
        )
        return control_output

    def control_torque(self, target_torque, current_torque, dt):
        """Control joint torque"""
        control_output = self.torque_controller.compute(
            target_torque, current_torque, dt
        )
        return control_output
```

### Force Control

Force control is crucial for safe interaction:

**Impedance Control:**
- Control robot's mechanical impedance
- Achieve desired interaction behavior
- Safe human-robot interaction
- Compliance with environment

**Admittance Control:**
- Control motion based on applied forces
- Mimic human-like responses
- Adaptive behavior
- Natural interaction

```python
class ImpedanceController:
    def __init__(self, mass=1.0, damping=10.0, stiffness=100.0):
        self.mass = mass
        self.damping = damping
        self.stiffness = stiffness
        self.velocity = 0.0
        self.position = 0.0

    def update(self, external_force, dt):
        """Update impedance controller state"""
        # Calculate acceleration: F = ma
        acceleration = (external_force - self.damping * self.velocity -
                       self.stiffness * self.position) / self.mass

        # Integrate to get velocity and position
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        return self.position, self.velocity

class ForceController:
    def __init__(self, joint_controller):
        self.joint_controller = joint_controller
        self.target_force = 0.0
        self.current_force = 0.0
        self.position_offset = 0.0
        self.stiffness = 1000.0  # N/m or Nm/rad

    def control(self, target_force, current_force, current_position, dt):
        """Control joint to achieve target force"""
        self.target_force = target_force
        self.current_force = current_force

        # Calculate force error
        force_error = self.target_force - self.current_force

        # Adjust position to achieve target force (simple stiffness control)
        position_correction = force_error / self.stiffness
        target_position = current_position + position_correction

        # Use position controller to achieve target position
        control_output = self.joint_controller.control_position(
            target_position, current_position, dt
        )

        return control_output
```

## Sensor Fusion and Integration

### Data Fusion Techniques

**Kalman Filtering:**
- Optimal estimation from multiple sensors
- Handles sensor noise and uncertainty
- Recursive algorithm suitable for real-time
- Good for tracking and estimation

**Extended Kalman Filter (EKF):**
- Handles nonlinear sensor models
- More complex but more accurate
- Used for pose estimation
- Common in SLAM applications

**Particle Filtering:**
- Non-parametric approach
- Handles multimodal distributions
- More computationally intensive
- Good for complex estimation problems

```python
class SensorFusion:
    def __init__(self):
        # State: [position, velocity, acceleration]
        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 1000  # Initial uncertainty

    def predict(self, dt, control_input=0):
        """Predict state forward in time"""
        # State transition matrix (constant acceleration model)
        F = np.array([
            [1, dt, 0.5*dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])

        # Process noise matrix
        Q = np.eye(3) * 0.1

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, measurement, sensor_type):
        """Update state with measurement"""
        if sensor_type == "position":
            H = np.array([[1, 0, 0]])  # Measurement matrix
            R = 0.1  # Measurement noise
            z = np.array([measurement])  # Measurement vector

        elif sensor_type == "velocity":
            H = np.array([[0, 1, 0]])
            R = 0.2
            z = np.array([measurement])

        # Kalman gain
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        y = z - H @ self.state  # Innovation
        self.state = self.state + K @ y

        # Update covariance
        I = np.eye(len(self.state))
        self.covariance = (I - K @ H) @ self.covariance
```

### Sensor Calibration

**Importance of Calibration:**
- Ensures accurate sensor readings
- Compensates for manufacturing variations
- Maintains system performance
- Critical for precise control

**Calibration Procedures:**
- **IMU Calibration**: Bias and scale factor correction
- **Camera Calibration**: Intrinsic and extrinsic parameters
- **Joint Calibration**: Zero positions and range verification
- **Force Sensor Calibration**: Reference load measurements

## Safety and Redundancy

### Safety Systems

**Emergency Stops:**
- Hardware-based emergency stops
- Software safety limits
- Watchdog timers
- Safe state procedures

**Fault Detection:**
- Sensor failure detection
- Actuator malfunction detection
- Communication failure detection
- Graceful degradation

### Redundancy Strategies

**Sensor Redundancy:**
- Multiple sensors for critical functions
- Cross-validation of readings
- Fail-safe operation
- Improved accuracy through fusion

**Actuator Redundancy:**
- Backup actuators for critical joints
- Alternative motion paths
- Graceful degradation
- Increased reliability

## Integration Challenges

### Communication Protocols

**Common Protocols:**
- **CAN Bus**: Robust for automotive/industrial applications
- **EtherCAT**: High-speed real-time communication
- **RS-485**: Simple, reliable for motor control
- **Ethernet**: High-bandwidth for vision systems

### Real-Time Requirements

**Timing Constraints:**
- Control loops: 1-10ms for position control
- Balance control: &lt;5ms for stability
- Vision processing: &lt;33ms for real-time
- Communication: &lt;1ms for critical data

### Power Management

**Power Considerations:**
- Sensor power consumption
- Actuator power requirements
- Battery life optimization
- Thermal management

## Advanced Sensing Technologies

### LIDAR Systems

**Applications:**
- 3D environment mapping
- Obstacle detection and avoidance
- Navigation and localization
- Safe operation verification

**Types:**
- **2D LIDAR**: Ground plane scanning
- **3D LIDAR**: Full 3D mapping
- **Solid-state**: No moving parts, more reliable

### Audio Systems

**Components:**
- **Microphone Arrays**: Sound source localization
- **Speakers**: Audio output and communication
- **Audio Processing**: Speech recognition and synthesis

**Applications:**
- Voice commands and responses
- Sound source localization
- Environmental monitoring
- Human-robot interaction

### Environmental Sensors

**Additional Sensors:**
- **Temperature Sensors**: Environmental monitoring
- **Humidity Sensors**: Environmental awareness
- **Gas Sensors**: Safety monitoring
- **Light Sensors**: Adaptive behavior

## Sensor Selection Guidelines

### Criteria for Selection

**Performance Requirements:**
- Accuracy and precision needed
- Response time requirements
- Operating range
- Environmental conditions

**System Integration:**
- Communication interface compatibility
- Power consumption
- Physical size and weight
- Cost considerations

### Trade-offs

**Accuracy vs. Cost:**
- Higher accuracy sensors are more expensive
- Determine required precision for application
- Consider sensor fusion for enhanced performance

**Speed vs. Accuracy:**
- Fast sensors may have lower accuracy
- Balance requirements based on application
- Use different sensors for different tasks

## Case Studies

### Boston Dynamics Robots
- Advanced IMU systems for dynamic balance
- Multiple cameras for 360-degree awareness
- High-precision joint encoders
- Custom actuator designs for efficiency

### SoftBank Pepper
- Multiple cameras for face recognition
- Tactile sensors on head for interaction
- Microphone array for voice processing
- Actuators optimized for safety

### Honda ASIMO
- Force sensors in feet for balance
- Multiple cameras for object recognition
- High-resolution joint encoders
- Advanced sensor fusion for walking

## Future Trends

### Emerging Technologies

**Bio-inspired Sensors:**
- Human-like sensory capabilities
- Adaptive sensitivity
- Energy-efficient designs
- Self-calibrating systems

**Advanced Materials:**
- Flexible sensors for conformal mounting
- Self-healing sensor materials
- Smart materials with sensing capabilities
- Nanotechnology sensors

### AI-Enhanced Sensing

**Learning-based Sensors:**
- Adaptive sensor parameters
- Context-aware sensing
- Predictive sensor maintenance
- Intelligent sensor fusion

## Summary

Sensors and actuators are the critical interface between humanoid robots and their environment. The selection, integration, and calibration of these components directly impact the robot's capabilities, safety, and performance. Modern humanoid robots require sophisticated sensor fusion and real-time control systems to achieve human-like perception and action capabilities.

The next chapter will explore locomotion principles, which heavily depend on the sensor-actuator systems described here.