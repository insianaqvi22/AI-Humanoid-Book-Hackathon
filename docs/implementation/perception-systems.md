---
title: Perception Systems Implementation
sidebar_position: 2
description: Implementing perception systems for AI humanoid robotics
---

# Perception Systems Implementation

Perception systems enable humanoid robots to understand their environment and interact effectively with humans and objects. This chapter covers the practical implementation of various perception systems, including sensor integration, data processing, and real-time perception algorithms.

## Introduction to Perception Systems

Perception systems in humanoid robots integrate multiple sensors to create a comprehensive understanding of the environment. Key components include:
- Visual perception (cameras, depth sensors)
- Auditory perception (microphones, speech recognition)
- Tactile perception (force/torque sensors, touch sensors)
- Proprioceptive perception (joint encoders, IMUs)
- Environmental perception (temperature, humidity, etc.)

## Visual Perception Implementation

### Camera Integration and Calibration

Proper camera integration is fundamental to visual perception.

```python
import cv2
import numpy as np
import json

class CameraSystem:
    def __init__(self, camera_id=0, resolution=(640, 480), fps=30):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.is_initialized = False

    def initialize(self):
        """Initialize camera and load calibration parameters"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Load camera calibration
        try:
            self.load_calibration("camera_calibration.json")
        except FileNotFoundError:
            print("Warning: Camera calibration file not found. Using default parameters.")
            self.camera_matrix = np.eye(3)
            self.dist_coeffs = np.zeros((5, 1))

        self.is_initialized = True
        return True

    def load_calibration(self, calibration_file):
        """Load camera calibration parameters"""
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)

        self.camera_matrix = np.array(calib_data['camera_matrix'])
        self.dist_coeffs = np.array(calib_data['dist_coeffs']).reshape(-1, 1)

    def capture_frame(self):
        """Capture and return a single frame"""
        if not self.is_initialized:
            raise RuntimeError("Camera not initialized")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")

        return frame

    def undistort_image(self, image):
        """Correct lens distortion"""
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
        return image

    def get_camera_parameters(self):
        """Get camera intrinsic parameters"""
        return {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'resolution': self.resolution
        }
```

### Image Preprocessing Pipeline

Real-time image preprocessing for perception tasks.

```python
class ImagePreprocessor:
    def __init__(self):
        self.brightness_adjustment = 0
        self.contrast_adjustment = 1.0
        self.gamma = 1.0

    def preprocess(self, image, target_size=(224, 224)):
        """Apply preprocessing pipeline to image"""
        # Undistort image (if camera calibration is available)
        # image = self.undistort(image)

        # Adjust brightness and contrast
        image = self.adjust_brightness_contrast(image)

        # Resize image
        image = cv2.resize(image, target_size)

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        return image

    def adjust_brightness_contrast(self, image, brightness=None, contrast=None):
        """Adjust brightness and contrast of image"""
        if brightness is None:
            brightness = self.brightness_adjustment
        if contrast is None:
            contrast = self.contrast_adjustment

        # Apply brightness and contrast adjustment
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted

    def apply_gamma_correction(self, image, gamma=None):
        """Apply gamma correction"""
        if gamma is None:
            gamma = self.gamma

        # Build a lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    def denoise_image(self, image, method='bilateral'):
        """Apply noise reduction"""
        if method == 'bilateral':
            # Bilateral filter preserves edges while reducing noise
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        else:
            return image
```

### Object Detection and Recognition

Implementing real-time object detection for humanoid robots.

```python
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class ObjectDetector:
    def __init__(self, confidence_threshold=0.5, model_path=None):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transforms = T.Compose([
            T.ToTensor(),
        ])
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def load_model(self, model_path):
        """Load object detection model"""
        if model_path:
            # Load custom trained model
            model = torch.load(model_path)
        else:
            # Load pre-trained model
            model = fasterrcnn_resnet50_fpn(pretrained=True)

        model.eval()
        model.to(self.device)
        return model

    def detect_objects(self, image):
        """Detect objects in image"""
        # Preprocess image
        image_tensor = self.transforms(image).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Run detection
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Process results
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Filter by confidence threshold
        confident_detections = scores > self.confidence_threshold

        results = []
        for i in range(len(boxes)):
            if confident_detections[i]:
                result = {
                    'bbox': boxes[i],
                    'label': self.class_names[labels[i]] if labels[i] < len(self.class_names) else f'unknown_{labels[i]}',
                    'confidence': scores[i],
                    'center': ((boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2)
                }
                results.append(result)

        return results

    def draw_detections(self, image, detections):
        """Draw detection results on image"""
        output_image = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']

            # Draw bounding box
            cv2.rectangle(output_image,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])),
                         (0, 255, 0), 2)

            # Draw label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(output_image, text,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_image
```

### Face Detection and Recognition

Enabling social interaction through face perception.

```python
import face_recognition

class FacePerception:
    def __init__(self, face_encoding_file="known_faces.json"):
        self.known_faces = {}
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces(face_encoding_file)

    def load_known_faces(self, encoding_file):
        """Load known face encodings"""
        try:
            with open(encoding_file, 'r') as f:
                data = json.load(f)
                self.known_face_encodings = [np.array(encoding) for encoding in data['encodings']]
                self.known_face_names = data['names']
        except FileNotFoundError:
            print(f"Face encoding file {encoding_file} not found. Starting with empty face database.")

    def detect_faces(self, image):
        """Detect faces in image"""
        # Convert image from BGR to RGB (OpenCV uses BGR, face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_image)

        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        faces = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"
            confidence = 0

            # Find the best match
            face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]

            faces.append({
                'bbox': (left, top, right, bottom),
                'name': name,
                'confidence': confidence,
                'encoding': encoding
            })

        return faces

    def register_new_face(self, image, name):
        """Register a new face"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_image)

        if len(face_encodings) > 0:
            # Register the first detected face
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            return True
        return False

    def draw_face_annotations(self, image, faces):
        """Draw face annotations on image"""
        output_image = image.copy()

        for face in faces:
            bbox = face['bbox']
            name = face['name']
            confidence = face['confidence']

            # Draw bounding box
            cv2.rectangle(output_image,
                         (bbox[0], bbox[1]),
                         (bbox[2], bbox[3]),
                         (0, 255, 0), 2)

            # Draw label
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
            cv2.putText(output_image, label,
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_image
```

## Auditory Perception Implementation

### Audio Input and Processing

Implementing audio perception for human-robot interaction.

```python
import pyaudio
import numpy as np
import webrtcvad
import speech_recognition as sr

class AudioSystem:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(2)  # Aggressiveness mode 2
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def start_audio_stream(self):
        """Start audio stream for real-time processing"""
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        return stream

    def capture_audio(self, duration=1.0):
        """Capture audio for specified duration"""
        stream = self.start_audio_stream()
        frames = []

        num_chunks = int(self.sample_rate * duration / self.chunk_size)
        for _ in range(num_chunks):
            data = stream.read(self.chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        return b''.join(frames)

    def detect_voice_activity(self, audio_data):
        """Detect voice activity in audio data"""
        # Convert audio data to the right format for VAD
        # VAD expects 16-bit PCM audio at 8kHz, 16kHz, or 32kHz
        # with frame lengths of 10, 20, or 30 ms
        frame_duration = 20  # ms
        frame_size = int(self.sample_rate * frame_duration / 1000) * 2  # 2 bytes per sample

        frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]

        voice_activity = []
        for frame in frames:
            if len(frame) == frame_size:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                voice_activity.append(is_speech)

        return voice_activity

    def recognize_speech(self, audio_data):
        """Recognize speech from audio data"""
        try:
            # Convert raw audio data to AudioData object
            audio = sr.AudioData(audio_data, self.sample_rate, 2)  # 2 bytes per sample

            # Use Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service; {e}")
            return None

    def listen_for_speech(self, timeout=None, phrase_time_limit=None):
        """Listen for speech using the microphone"""
        try:
            with self.microphone as source:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            try:
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from speech recognition service; {e}")
                return None
        except sr.WaitTimeoutError:
            print("Listening timed out")
            return None
```

### Sound Source Localization

Determining the direction of sounds for spatial awareness.

```python
class SoundLocalization:
    def __init__(self, microphone_array):
        self.microphones = microphone_array
        self.speed_of_sound = 343  # m/s

    def estimate_direction_of_arrival(self, audio_signals, sample_rate):
        """
        Estimate direction of arrival using time difference of arrival (TDOA)
        """
        # This is a simplified example - real implementation would be more complex
        # and require multiple synchronized microphones

        # Calculate cross-correlation between microphone signals
        # to find time delays
        delays = []
        for i in range(1, len(audio_signals)):
            correlation = np.correlate(audio_signals[0], audio_signals[i], mode='full')
            delay_samples = np.argmax(correlation) - len(audio_signals[i]) + 1
            delay_time = delay_samples / sample_rate
            delays.append(delay_time)

        # Convert time delays to angles (simplified geometry)
        # This assumes a linear microphone array
        angles = []
        for delay in delays:
            # For a linear array with known spacing, convert delay to angle
            # This is a simplified calculation
            angle = np.arcsin(delay * self.speed_of_sound / 0.1)  # 0.1m spacing
            angles.append(angle)

        return angles

    def track_sound_source(self, audio_stream):
        """
        Track moving sound source
        """
        # Implementation would continuously estimate DoA and track source
        pass
```

## Tactile Perception Implementation

### Force/Torque Sensing

Implementing tactile perception for safe interaction.

```python
class ForceTorqueSensor:
    def __init__(self, sensor_id, calibration_matrix=None):
        self.sensor_id = sensor_id
        self.calibration_matrix = calibration_matrix or np.eye(6)  # 6-axis: 3 forces, 3 torques
        self.bias = np.zeros(6)  # Initial bias values
        self.is_calibrated = False

    def read_raw_data(self):
        """Read raw sensor data (simulated)"""
        # In real implementation, this would interface with hardware
        # For simulation, return random data with some bias
        raw_data = np.random.normal(0, 0.1, 6) + self.bias
        return raw_data

    def calibrate_sensor(self, num_samples=100):
        """Calibrate sensor by measuring bias in zero-load condition"""
        print("Calibrating force/torque sensor. Ensure no external forces are applied.")
        raw_readings = []

        for _ in range(num_samples):
            raw_data = self.read_raw_data()
            raw_readings.append(raw_data)
            time.sleep(0.01)  # Small delay between readings

        # Calculate average bias
        self.bias = np.mean(raw_readings, axis=0)
        self.is_calibrated = True
        print(f"Calibration complete. Bias: {self.bias}")

    def get_force_torque(self):
        """Get calibrated force/torque measurements"""
        if not self.is_calibrated:
            raise RuntimeError("Sensor not calibrated")

        raw_data = self.read_raw_data()
        calibrated_data = raw_data - self.bias
        force_torque = self.calibration_matrix @ calibrated_data

        return {
            'forces': force_torque[:3],   # Fx, Fy, Fz
            'torques': force_torque[3:],  # Tx, Ty, Tz
            'magnitude': np.linalg.norm(force_torque[:3])
        }

    def detect_contact(self, threshold=5.0):
        """Detect contact based on force magnitude"""
        force_data = self.get_force_torque()
        return force_data['magnitude'] > threshold

    def detect_slip(self, force_threshold=2.0, time_window=0.1):
        """Detect slip based on force changes"""
        # Simplified slip detection based on rapid force changes
        current_force = self.get_force_torque()
        # In real implementation, would track force over time window
        return current_force['magnitude'] > force_threshold
```

### Tactile Sensor Arrays

Implementing distributed tactile sensing.

```python
class TactileSensorArray:
    def __init__(self, rows, cols, sensor_type='pressure'):
        self.rows = rows
        self.cols = cols
        self.sensor_type = sensor_type
        self.pressure_map = np.zeros((rows, cols))
        self.threshold = 0.1  # Minimum pressure to register contact
        self.max_pressure = 100.0  # Maximum pressure in arbitrary units

    def read_pressure_map(self):
        """Read pressure values from all sensors"""
        # In real implementation, this would interface with tactile sensor hardware
        # For simulation, generate pressure map with some noise
        self.pressure_map = np.random.normal(0, 0.01, (self.rows, self.cols))

        # Add some contact points for demonstration
        # In real use, this would come from actual sensor readings
        self.pressure_map[10:15, 20:25] += 2.0  # Simulated contact area
        self.pressure_map[5:8, 30:35] += 1.5    # Another contact area

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

    def get_pressure_gradient(self):
        """Calculate pressure gradient for slip detection"""
        grad_y, grad_x = np.gradient(self.pressure_map)
        return np.sqrt(grad_x**2 + grad_y**2)

    def visualize_pressure(self, save_path=None):
        """Visualize pressure distribution"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.imshow(self.pressure_map, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Pressure (arbitrary units)')
        plt.title('Tactile Sensor Array Pressure Distribution')

        # Mark contact points
        contacts = self.detect_contacts()
        if contacts:
            contact_rows, contact_cols = zip(*contacts)
            plt.scatter(contact_cols, contact_rows, c='red', s=10, alpha=0.7, label='Contacts')
            plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()
```

## Integration with Robot Control

### Sensor Fusion

Combining multiple sensor inputs for robust perception.

```python
class SensorFusion:
    def __init__(self):
        self.visual_system = None
        self.auditory_system = None
        self.tactile_system = None
        self.imu_data = None
        self.joint_data = None

    def initialize_systems(self, camera_sys, audio_sys, tactile_sys):
        """Initialize all perception systems"""
        self.visual_system = camera_sys
        self.auditory_system = audio_sys
        self.tactile_system = tactile_sys

    def get_fused_perception(self):
        """Get fused perception data from all systems"""
        perception_data = {}

        # Visual perception
        if self.visual_system:
            try:
                frame = self.visual_system.capture_frame()
                objects = self.visual_system.object_detector.detect_objects(frame)
                faces = self.visual_system.face_perception.detect_faces(frame)
                perception_data['visual'] = {
                    'objects': objects,
                    'faces': faces,
                    'timestamp': time.time()
                }
            except Exception as e:
                print(f"Visual perception error: {e}")

        # Auditory perception
        if self.auditory_system:
            try:
                # For continuous audio processing, this would be more complex
                # This is a simplified example
                perception_data['auditory'] = {
                    'sound_detected': True,
                    'timestamp': time.time()
                }
            except Exception as e:
                print(f"Auditory perception error: {e}")

        # Tactile perception
        if self.tactile_system:
            try:
                tactile_data = self.tactile_system.get_pressure_map()
                perception_data['tactile'] = {
                    'pressure_map': tactile_data,
                    'contacts': self.tactile_system.detect_contacts(),
                    'timestamp': time.time()
                }
            except Exception as e:
                print(f"Tactile perception error: {e}")

        # Add IMU and joint data if available
        if self.imu_data:
            perception_data['proprioceptive'] = {
                'imu': self.imu_data,
                'joints': self.joint_data,
                'timestamp': time.time()
            }

        return perception_data

    def track_object(self, object_id, max_age=10):
        """Track objects across frames"""
        # Implementation would maintain object trajectories
        pass

    def maintain_world_model(self, perception_data):
        """Maintain a world model based on perception data"""
        world_model = {
            'objects': {},
            'humans': {},
            'obstacles': {},
            'safe_zones': {},
            'timestamp': time.time()
        }

        # Process visual data
        if 'visual' in perception_data:
            for obj in perception_data['visual']['objects']:
                obj_id = f"{obj['label']}_{hash(str(obj['bbox'])) % 10000}"
                world_model['objects'][obj_id] = {
                    'type': obj['label'],
                    'position': obj['center'],
                    'bbox': obj['bbox'],
                    'confidence': obj['confidence']
                }

            for face in perception_data['visual']['faces']:
                face_id = f"person_{hash(str(face['bbox'])) % 10000}"
                world_model['humans'][face_id] = {
                    'name': face['name'],
                    'position': ((face['bbox'][0] + face['bbox'][2]) / 2,
                                (face['bbox'][1] + face['bbox'][3]) / 2),
                    'bbox': face['bbox'],
                    'confidence': face['confidence']
                }

        return world_model
```

### Real-Time Processing Considerations

Implementing efficient real-time perception.

```python
import threading
import queue
from collections import deque

class RealTimePerceptionSystem:
    def __init__(self, max_fps=30):
        self.max_fps = max_fps
        self.frame_interval = 1.0 / max_fps
        self.running = False
        self.processing_thread = None

        # Queues for multi-threaded processing
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)

        # Frame history for temporal processing
        self.frame_history = deque(maxlen=5)

        # Processing modules
        self.camera_system = None
        self.object_detector = None
        self.face_perceptor = None

    def start_processing(self):
        """Start real-time perception processing"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()

    def stop_processing(self):
        """Stop real-time perception processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()

    def _processing_loop(self):
        """Main processing loop"""
        last_process_time = time.time()

        while self.running:
            current_time = time.time()

            # Control frame rate
            if current_time - last_process_time < self.frame_interval:
                time.sleep(0.001)
                continue

            try:
                # Capture frame
                frame = self.camera_system.capture_frame()

                # Add to history
                self.frame_history.append(frame)

                # Process frame
                results = self.process_frame(frame)

                # Put results in output queue
                try:
                    self.output_queue.put_nowait(results)
                except queue.Full:
                    # Drop old results if queue is full
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.output_queue.put_nowait(results)

                last_process_time = current_time

            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.01)  # Brief pause before continuing

    def process_frame(self, frame):
        """Process a single frame with all perception modules"""
        results = {
            'timestamp': time.time(),
            'objects': [],
            'faces': [],
            'processing_time': 0
        }

        start_time = time.time()

        # Object detection
        if self.object_detector:
            results['objects'] = self.object_detector.detect_objects(frame)

        # Face detection
        if self.face_perceptor:
            results['faces'] = self.face_perceptor.detect_faces(frame)

        results['processing_time'] = time.time() - start_time
        return results

    def get_latest_results(self):
        """Get the latest perception results"""
        try:
            # Get the most recent results from queue
            results = None
            while not self.output_queue.empty():
                results = self.output_queue.get_nowait()
            return results
        except queue.Empty:
            return None

    def optimize_processing(self):
        """Optimize processing based on performance"""
        # Implementation would adjust processing based on available time
        pass
```

## Perception for Human-Robot Interaction

### Attention and Gaze Control

Implementing perception-driven attention systems.

```python
class AttentionSystem:
    def __init__(self, camera_system, neck_controller):
        self.camera = camera_system
        self.neck = neck_controller
        self.attention_targets = []
        self.current_focus = None
        self.attention_priority = {}  # Priority for different types of targets

    def set_attention_priorities(self):
        """Set priorities for different attention targets"""
        self.attention_priority = {
            'face': 10,
            'person': 8,
            'moving_object': 6,
            'known_object': 5,
            'unknown_object': 3,
            'sound_source': 7
        }

    def find_attention_targets(self, perception_data):
        """Find targets that should attract attention"""
        targets = []

        # Visual targets
        if 'visual' in perception_data:
            # Faces have highest priority
            for face in perception_data['visual']['faces']:
                targets.append({
                    'type': 'face',
                    'position': face['center'],
                    'priority': self.attention_priority['face'],
                    'data': face
                })

            # Other objects
            for obj in perception_data['visual']['objects']:
                targets.append({
                    'type': obj['label'],
                    'position': obj['center'],
                    'priority': self.attention_priority.get(obj['label'],
                                                         self.attention_priority.get('unknown_object', 1)),
                    'data': obj
                })

        # Sort by priority
        targets.sort(key=lambda x: x['priority'], reverse=True)
        return targets

    def attend_to_target(self, target):
        """Direct attention to a specific target"""
        if target['type'] in ['face', 'person']:
            # Look at the target
            self.look_at_position(target['position'])
        elif target['type'] == 'sound_source':
            # Turn head toward sound
            self.turn_toward_direction(target['position'])

    def look_at_position(self, position_2d, camera_fov=(60, 45)):
        """Control neck to look at 2D image position"""
        # Convert 2D image coordinates to 3D world coordinates
        # This requires depth information or assumptions
        image_width, image_height = 640, 480  # Assuming standard resolution
        center_x, center_y = image_width / 2, image_height / 2

        # Calculate angular offsets from center
        x_offset = (position_2d[0] - center_x) / center_x * (camera_fov[0] / 2)
        y_offset = (position_2d[1] - center_y) / center_y * (camera_fov[1] / 2)

        # Send commands to neck controller
        self.neck.pan(x_offset)
        self.neck.tilt(-y_offset)  # Negative because image coordinates are inverted

    def maintain_attention(self, perception_data, focus_duration=2.0):
        """Maintain attention on important targets"""
        targets = self.find_attention_targets(perception_data)

        if targets:
            # Attend to the highest priority target
            primary_target = targets[0]
            self.attend_to_target(primary_target)
            self.current_focus = primary_target
        else:
            # Look straight ahead if no targets
            self.neck.center()
```

## Safety and Robustness

### Perception Validation

Ensuring perception results are reliable.

```python
class PerceptionValidator:
    def __init__(self):
        self.confidence_thresholds = {
            'object_detection': 0.7,
            'face_recognition': 0.8,
            'speech_recognition': 0.85
        }
        self.temporal_consistency_window = 5  # frames

    def validate_object_detection(self, detection_result):
        """Validate object detection results"""
        valid_detections = []

        for detection in detection_result:
            # Check confidence
            if detection['confidence'] < self.confidence_thresholds['object_detection']:
                continue

            # Check if bounding box is reasonable
            bbox = detection['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Reject extremely small or large detections
            if width < 10 or height < 10 or width > 600 or height > 400:
                continue

            valid_detections.append(detection)

        return valid_detections

    def validate_face_recognition(self, face_result):
        """Validate face recognition results"""
        valid_faces = []

        for face in face_result:
            if face['confidence'] < self.confidence_thresholds['face_recognition']:
                continue

            # Additional validation could include:
            # - Face size consistency
            # - Face position plausibility
            # - Temporal consistency

            valid_faces.append(face)

        return valid_faces

    def check_temporal_consistency(self, current_detections, previous_detections, max_velocity=100):
        """Check if detections are temporally consistent"""
        # Implementation would compare current and previous detections
        # to verify that objects haven't moved unrealistically fast
        pass

    def handle_perception_failures(self, failure_type):
        """Handle different types of perception failures"""
        if failure_type == 'camera_failure':
            # Switch to alternative camera or use other sensors
            pass
        elif failure_type == 'recognition_failure':
            # Use alternative recognition methods or request clarification
            pass
        elif failure_type == 'sensor_noise':
            # Apply additional filtering or request re-sensing
            pass
```

## Performance Optimization

### Efficient Processing Techniques

```python
class EfficientPerception:
    def __init__(self):
        self.roi_enabled = True  # Region of interest processing
        self.pyramid_levels = 3  # Image pyramid for multi-scale processing
        self.temporal_filtering = True

    def process_roi(self, image, roi_bbox):
        """Process only region of interest"""
        x1, y1, x2, y2 = map(int, roi_bbox)
        roi = image[y1:y2, x1:x2]

        # Process only the ROI
        roi_result = self.process_image(roi)

        # Adjust coordinates back to full image space
        adjusted_result = self.adjust_coordinates(roi_result, (x1, y1))

        return adjusted_result

    def build_image_pyramid(self, image, levels=None):
        """Build image pyramid for multi-scale processing"""
        if levels is None:
            levels = self.pyramid_levels

        pyramid = [image]
        for i in range(1, levels):
            pyramid.append(cv2.pyrDown(pyramid[i-1]))

        return pyramid

    def multi_scale_detection(self, image):
        """Perform detection at multiple scales"""
        pyramid = self.build_image_pyramid(image)
        all_detections = []

        for scale, img in enumerate(pyramid):
            detections = self.detect_at_scale(img, scale_factor=2**scale)
            all_detections.extend(detections)

        return self.merge_detections(all_detections)

    def temporal_filter_detections(self, current_detections, previous_detections):
        """Filter detections using temporal information"""
        if not self.temporal_filtering or not previous_detections:
            return current_detections

        # Match detections with previous frame
        filtered_detections = []
        for current_det in current_detections:
            matched = False
            for prev_det in previous_detections:
                # Check if detections are close enough to be the same object
                center_dist = np.linalg.norm(
                    np.array(current_det['center']) - np.array(prev_det['center'])
                )

                if center_dist < 50:  # Threshold for matching
                    # Combine information from both frames
                    combined_det = self.combine_detections(current_det, prev_det)
                    filtered_detections.append(combined_det)
                    matched = True
                    break

            if not matched:
                # New detection
                current_det['age'] = 0
                filtered_detections.append(current_det)

        return filtered_detections

    def combine_detections(self, det1, det2):
        """Combine two detections of the same object"""
        combined = det1.copy()

        # Average positions
        combined['center'] = (np.array(det1['center']) + np.array(det2['center'])) / 2

        # Average confidence (with some temporal weighting)
        combined['confidence'] = 0.7 * det1['confidence'] + 0.3 * det2['confidence']

        return combined
```

## Summary

Perception systems are critical for humanoid robots to interact effectively with their environment. Successful implementation requires:

1. **Multi-sensor Integration**: Combining visual, auditory, and tactile inputs
2. **Real-time Processing**: Meeting timing constraints for interactive applications
3. **Robustness**: Handling sensor failures and noisy environments
4. **Calibration**: Properly calibrating all sensors
5. **Validation**: Ensuring perception results are reliable
6. **Optimization**: Efficient processing for resource-constrained platforms

The next chapter will explore how these perception systems integrate with AI and control systems to enable complex humanoid behaviors.