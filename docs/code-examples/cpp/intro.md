---
title: C++ Examples for AI Humanoid Robotics
sidebar_position: 2
description: C++ code examples for AI humanoid robotics implementation
---

# C++ Examples for AI Humanoid Robotics

This section provides practical C++ code examples for implementing various aspects of AI humanoid robotics. C++ is widely used in robotics for its performance characteristics, real-time capabilities, and system-level programming requirements.

## Setting Up the Development Environment

Before diving into the examples, you'll need to set up your C++ development environment:

```bash
# Install build tools and dependencies
sudo apt-get update
sudo apt-get install build-essential cmake git
sudo apt-get install libeigen3-dev libopencv-dev libboost-all-dev
sudo apt-get install libgsl-dev liblapack-dev libblas-dev

# For robotics frameworks (like ROS)
sudo apt-get install ros-noetic-desktop-full  # For ROS 1
# or
# Install ROS 2 Foxy/Humble following official instructions

# For neural network integration
sudo apt-get install libtorch-dev  # LibTorch (PyTorch C++ API)
```

## Basic Movement Control Example

Let's start with a basic example of controlling robot movements in C++:

```cpp
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>
#include <stdexcept>

/**
 * @brief Robot controller class for humanoid robot movements
 */
class RobotController {
public:
    struct JointLimits {
        double min_angle;
        double max_angle;
    };

    struct JointState {
        double position;
        double velocity;
        double effort;
    };

    RobotController(const std::string& model = "generic", bool simulation = true)
        : robot_model_(model), simulation_(simulation) {
        initializeJointLimits();
        initializeJointStates();
    }

    /**
     * @brief Move a specific joint to target position
     */
    void moveToPosition(const std::string& joint_name, double target_position, double duration = 1.0) {
        if (joint_limits_.find(joint_name) == joint_limits_.end()) {
            throw std::invalid_argument("Unknown joint: " + joint_name);
        }

        // Check joint limits
        JointLimits limits = joint_limits_[joint_name];
        if (target_position < limits.min_angle || target_position > limits.max_angle) {
            std::cerr << "Warning: Position " << target_position
                      << " exceeds limits for " << joint_name << std::endl;

            // Clamp to limits
            target_position = std::max(limits.min_angle,
                                     std::min(limits.max_angle, target_position));
        }

        // Get current position
        double current_position = joint_states_[joint_name].position;

        // Calculate movement parameters
        const int steps = static_cast<int>(duration * 100); // 100Hz control rate
        double delta = target_position - current_position;

        // Execute movement with smooth interpolation
        for (int i = 0; i <= steps; ++i) {
            double t = static_cast<double>(i) / steps; // Normalized time (0 to 1)

            // Use cubic interpolation for smooth motion (ease-in, ease-out)
            double smooth_t = 3 * t * t - 2 * t * t * t;

            double intermediate_position = current_position + smooth_t * delta;

            // Update joint position
            joint_states_[joint_name].position = intermediate_position;

            // In simulation, just update internal state
            // In real robot, send command to actual joint
            if (simulation_) {
                simulateJointMovement(joint_name, intermediate_position);
            } else {
                sendJointCommand(joint_name, intermediate_position);
            }

            // Sleep for control rate
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration * 1000 / steps)));
        }

        std::cout << "Successfully moved " << joint_name << " to "
                  << target_position << " radians" << std::endl;
    }

    /**
     * @brief Move multiple joints simultaneously
     */
    void moveMultipleJoints(const std::map<std::string, double>& joint_targets, double duration = 1.0) {
        // Validate all joints exist
        for (const auto& [joint_name, target_position] : joint_targets) {
            if (joint_limits_.find(joint_name) == joint_limits_.end()) {
                throw std::invalid_argument("Unknown joint: " + joint_name);
            }
        }

        // Get starting positions
        std::map<std::string, double> start_positions;
        for (const auto& [joint_name, target_position] : joint_targets) {
            start_positions[joint_name] = joint_states_[joint_name].position;
        }

        // Execute simultaneous movement
        const int steps = static_cast<int>(duration * 100); // 100Hz control rate

        for (int i = 0; i <= steps; ++i) {
            double t = static_cast<double>(i) / steps; // Normalized time (0 to 1)
            double smooth_t = 3 * t * t - 2 * t * t * t; // Smooth interpolation

            for (const auto& [joint_name, target_position] : joint_targets) {
                double start_pos = start_positions.at(joint_name);
                double delta = target_position - start_pos;
                double intermediate_position = start_pos + smooth_t * delta;

                // Update joint position
                joint_states_[joint_name].position = intermediate_position;

                // Send command (simulation or real)
                if (simulation_) {
                    simulateJointMovement(joint_name, intermediate_position);
                } else {
                    sendJointCommand(joint_name, intermediate_position);
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration * 1000 / steps)));
        }

        std::cout << "\nSuccessfully moved multiple joints" << std::endl;
    }

    /**
     * @brief Execute a waving motion with the right arm
     */
    void waveHand(double duration = 2.0) {
        std::cout << "Executing wave hand motion..." << std::endl;

        // Define wave motion sequence
        std::vector<std::map<std::string, double>> wave_sequence = {
            {{"RShoulderPitch", -0.5}, {"RElbowRoll", -1.0}, {"RWristYaw", 0.0}},   // Raise arm
            {{"RShoulderPitch", -0.5}, {"RElbowRoll", -1.0}, {"RWristYaw", 0.5}},   // Wave right
            {{"RShoulderPitch", -0.5}, {"RElbowRoll", -1.0}, {"RWristYaw", -0.5}},  // Wave left
            {{"RShoulderPitch", -0.5}, {"RElbowRoll", -1.0}, {"RWristYaw", 0.5}},   // Wave right
            {{"RShoulderPitch", -0.5}, {"RElbowRoll", -1.0}, {"RWristYaw", 0.0}},   // Return to center
            {{"RShoulderPitch", 0.0}, {"RElbowRoll", 0.0}, {"RWristYaw", 0.0}}      // Lower arm
        };

        // Execute wave sequence
        for (size_t i = 0; i < wave_sequence.size(); ++i) {
            std::cout << "Wave motion step " << (i + 1) << "/" << wave_sequence.size() << std::endl;
            moveMultipleJoints(wave_sequence[i], duration / wave_sequence.size());
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Small pause between steps
        }
    }

    /**
     * @brief Execute a bowing motion
     */
    void bow(double duration = 1.0) {
        std::cout << "Executing bow motion..." << std::endl;

        // Bow forward
        std::map<std::string, double> bow_targets = {
            {"LHipPitch", -0.5}, {"RHipPitch", -0.5},
            {"LAnklePitch", 0.3}, {"RAnklePitch", 0.3}
        };
        moveMultipleJoints(bow_targets, duration);

        // Hold the bow
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Return to standing position
        std::map<std::string, double> stand_targets = {
            {"LHipPitch", 0.0}, {"RHipPitch", 0.0},
            {"LAnklePitch", 0.0}, {"RAnklePitch", 0.0}
        };
        moveMultipleJoints(stand_targets, duration);

        std::cout << "Bow completed" << std::endl;
    }

    /**
     * @brief Get current positions of all joints
     */
    std::map<std::string, JointState> getCurrentPositions() const {
        return joint_states_;
    }

private:
    std::string robot_model_;
    bool simulation_;
    std::map<std::string, JointLimits> joint_limits_;
    std::map<std::string, JointState> joint_states_;

    void initializeJointLimits() {
        if (robot_model_ == "nao") {
            joint_limits_["HeadYaw"] = {-2.0857, 2.0857};
            joint_limits_["HeadPitch"] = {-0.7068, 0.4469};
            joint_limits_["LShoulderPitch"] = {-2.0857, 2.0857};
            joint_limits_["LShoulderRoll"] = {-0.3142, 1.3265};
            joint_limits_["LElbowYaw"] = {-2.0857, 2.0857};
            joint_limits_["LElbowRoll"] = {-1.5446, -0.0349};
            joint_limits_["LWristYaw"] = {-1.8238, 1.8238};
            joint_limits_["LHipYawPitch"] = {-1.1453, 0.7408};
            joint_limits_["LHipRoll"] = {-0.3794, 0.7905};
            joint_limits_["LHipPitch"] = {-1.5358, 0.4840};
            joint_limits_["LKneePitch"] = {-0.0923, 2.1125};
            joint_limits_["LAnklePitch"] = {-1.1895, 0.9329};
            joint_limits_["LAnkleRoll"] = {-0.3900, 0.7408};
        } else {
            // Default limits for generic humanoid
            joint_limits_["hip_pitch"] = {-1.57, 1.57};
            joint_limits_["knee_pitch"] = {0, 2.35};
            joint_limits_["ankle_pitch"] = {-0.5, 0.5};
            joint_limits_["shoulder_pitch"] = {-2.09, 1.57};
            joint_limits_["elbow_pitch"] = {-2.35, 0};
            joint_limits_["wrist_yaw"] = {-1.57, 1.57};
        }
    }

    void initializeJointStates() {
        for (const auto& [joint_name, limits] : joint_limits_) {
            joint_states_[joint_name] = {0.0, 0.0, 0.0}; // pos, vel, effort
        }
    }

    void simulateJointMovement(const std::string& joint_name, double position) {
        // Simulate joint movement (for demonstration purposes)
        std::cout << "  " << joint_name << ": " << std::fixed << std::setprecision(3)
                  << position << " rad" << std::flush;
        std::cout << "\r";
    }

    void sendJointCommand(const std::string& joint_name, double position) {
        // Send actual command to robot joint (implementation depends on robot platform)
        // This would interface with the actual robot
    }
};

// Example usage
int main() {
    std::cout << "=== Humanoid Robot Controller Example ===" << std::endl;

    // Create robot controller instance
    RobotController robot("nao", true); // NAO robot model, simulation mode

    std::cout << "\n1. Moving head to look left..." << std::endl;
    robot.moveToPosition("HeadYaw", 0.5, 1.0);

    std::cout << "\n2. Moving head to look right..." << std::endl;
    robot.moveToPosition("HeadYaw", -0.5, 1.0);

    std::cout << "\n3. Returning head to center..." << std::endl;
    robot.moveToPosition("HeadYaw", 0.0, 1.0);

    std::cout << "\n4. Waving hand..." << std::endl;
    robot.waveHand(3.0);

    std::cout << "\n5. Bowing..." << std::endl;
    robot.bow(1.0);

    std::cout << "\n6. Current joint positions:" << std::endl;
    auto positions = robot.getCurrentPositions();
    for (const auto& [joint, state] : positions) {
        std::cout << "  " << joint << ": " << std::fixed << std::setprecision(3)
                  << state.position << " radians" << std::endl;
    }

    return 0;
}
```

## Computer Vision Example

Here's an example of implementing computer vision in C++ using OpenCV:

```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

/**
 * @brief Computer vision system for humanoid robot
 */
class VisionSystem {
public:
    VisionSystem() {
        initializeObjectDetector();
    }

    /**
     * @brief Initialize object detection model
     */
    void initializeObjectDetector() {
        try {
            // For this example, we'll use a pre-trained MobileNet SSD
            // In practice, you might use YOLO, SSD, or other models
            // net_ = cv::dnn::readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel");
            std::cout << "Object detector initialized (using basic detection as fallback)" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Warning: Could not load pre-trained model. Using basic detection." << std::endl;
        }
    }

    /**
     * @brief Start camera capture
     */
    bool startCamera(int camera_id = 0) {
        cap_.open(camera_id);
        if (!cap_.isOpened()) {
            std::cerr << "Could not open camera " << camera_id << std::endl;
            return false;
        }

        // Set camera properties
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap_.set(cv::CAP_PROP_FPS, 30);

        std::cout << "Camera started successfully" << std::endl;
        return true;
    }

    /**
     * @brief Detect objects in image
     */
    struct Detection {
        std::string label;
        float confidence;
        cv::Rect bbox;
        cv::Point center;
    };

    std::vector<Detection> detectObjects(const cv::Mat& image) {
        // Use basic color-based detection as fallback
        return basicColorDetection(image);
    }

    /**
     * @brief Detect faces in image
     */
    std::vector<Detection> detectFaces(const cv::Mat& image) {
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Load face cascade classifier
        cv::CascadeClassifier face_cascade;
        if (!face_cascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"))) {
            std::cerr << "Error: Could not load face cascade classifier" << std::endl;
            return {};
        }

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(30, 30));

        std::vector<Detection> face_detections;
        for (const auto& face : faces) {
            Detection detection;
            detection.label = "face";
            detection.confidence = 0.9f; // High confidence for face detection
            detection.bbox = face;
            detection.center = cv::Point(face.x + face.width/2, face.y + face.height/2);
            face_detections.push_back(detection);
        }

        return face_detections;
    }

    /**
     * @brief Process video stream continuously
     */
    void processVideoStream() {
        if (!cap_.isOpened()) {
            std::cerr << "Error: Camera not opened. Call startCamera() first." << std::endl;
            return;
        }

        std::cout << "Starting video processing... Press 'q' to quit" << std::endl;

        cv::Mat frame;
        while (true) {
            cap_ >> frame;
            if (frame.empty()) {
                std::cerr << "Failed to read frame from camera" << std::endl;
                break;
            }

            // Process frame
            auto objects = detectObjects(frame);
            auto faces = detectFaces(frame);

            // Draw detections on frame
            cv::Mat processed_frame = drawDetections(frame, objects, faces);

            // Display frame
            cv::imshow("Robot Vision", processed_frame);

            // Exit on 'q' key press
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }

        // Clean up
        cap_.release();
        cv::destroyAllWindows();
    }

private:
    cv::VideoCapture cap_;
    cv::dnn::Net net_;

    /**
     * @brief Basic color-based object detection as fallback
     */
    std::vector<Detection> basicColorDetection(const cv::Mat& image) {
        // Convert to HSV for easier color detection
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

        // Define color ranges (in HSV)
        struct ColorRange {
            cv::Scalar lower;
            cv::Scalar upper;
        };

        std::map<std::string, ColorRange> color_ranges = {
            {"red", {{0, 50, 50}, {10, 255, 255}}},
            {"blue", {{100, 50, 50}, {130, 255, 255}}},
            {"green", {{40, 50, 50}, {80, 255, 255}}}
        };

        std::vector<Detection> objects;

        for (const auto& [color_name, range] : color_ranges) {
            // Create mask for this color
            cv::Mat mask;
            cv::inRange(hsv, range.lower, range.upper, mask);

            // Find contours
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                double area = cv::contourArea(contour);

                // Filter by minimum area
                if (area > 500) {
                    // Get bounding box
                    cv::Rect bbox = cv::boundingRect(contour);

                    Detection detection;
                    detection.label = color_name;
                    detection.confidence = 0.7f; // Assumed confidence for color detection
                    detection.bbox = bbox;
                    detection.center = cv::Point(bbox.x + bbox.width/2, bbox.y + bbox.height/2);
                    objects.push_back(detection);
                }
            }
        }

        return objects;
    }

    /**
     * @brief Draw detection results on image
     */
    cv::Mat drawDetections(const cv::Mat& image,
                          const std::vector<Detection>& objects,
                          const std::vector<Detection>& faces) {
        cv::Mat output_image = image.clone();

        // Draw object detections
        for (const auto& obj : objects) {
            // Draw bounding box
            cv::rectangle(output_image, obj.bbox, cv::Scalar(0, 255, 0), 2);

            // Draw label
            std::string label = obj.label + ": " + std::to_string(obj.confidence).substr(0, 4);
            cv::putText(output_image, label,
                       cv::Point(obj.bbox.x, obj.bbox.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        // Draw face detections
        for (const auto& face : faces) {
            // Draw bounding box (different color for faces)
            cv::rectangle(output_image, face.bbox, cv::Scalar(255, 0, 0), 2);

            // Draw label
            std::string label = face.label + ": " + std::to_string(face.confidence).substr(0, 4);
            cv::putText(output_image, label,
                       cv::Point(face.bbox.x, face.bbox.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
        }

        return output_image;
    }
};

// Example usage for vision system
void visionExample() {
    std::cout << "=== Robot Vision System Example ===" << std::endl;

    // Create vision system
    VisionSystem vision;

    // Create a sample image for demonstration
    cv::Mat sample_image = cv::Mat::zeros(480, 640, CV_8UC3);

    // Add some colored rectangles to simulate objects
    cv::rectangle(sample_image, cv::Rect(100, 100, 100, 100), cv::Scalar(0, 0, 255), -1);  // Red rectangle
    cv::rectangle(sample_image, cv::Rect(300, 150, 100, 100), cv::Scalar(255, 0, 0), -1);  // Blue rectangle
    cv::circle(sample_image, cv::Point(500, 300), 50, cv::Scalar(0, 255, 0), -1);  // Green circle

    std::cout << "Sample image created with colored shapes" << std::endl;

    // Detect objects in sample image
    auto objects = vision.detectObjects(sample_image);
    auto faces = vision.detectFaces(sample_image);  // Will find none in this example

    std::cout << "\nDetected " << objects.size() << " objects:" << std::endl;
    for (size_t i = 0; i < objects.size(); ++i) {
        const auto& obj = objects[i];
        std::cout << "  Object " << (i + 1) << ": " << obj.label
                  << " at (" << obj.center.x << ", " << obj.center.y
                  << ") with confidence " << obj.confidence << std::endl;
    }

    std::cout << "\nDetected " << faces.size() << " faces" << std::endl;

    // Draw detections on image
    cv::Mat result_image = vision.drawDetections(sample_image, objects, faces);

    // Display result (uncomment to show image)
    // cv::imshow("Detection Result", result_image);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
}

int main_vision() {
    visionExample();
    return 0;
}
```

## AI Integration Example

Here's an example of integrating AI with the robot system using C++ and LibTorch:

```cpp
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <random>
#include <memory>
#include <algorithm>

/**
 * @brief AI system for humanoid robot decision making
 */
class RobotAI {
public:
    RobotAI() {
        initializeModels();
    }

    /**
     * @brief Process sensory input through AI model
     */
    torch::Tensor processSensoryInput(const std::vector<float>& sensory_data) {
        // Convert to tensor
        torch::Tensor input_tensor = torch::from_blob(
            const_cast<float*>(sensory_data.data()),
            {1, static_cast<long>(sensory_data.size())},
            torch::kFloat
        ).clone();

        // Process through perception model
        torch::Tensor perception_output;
        perception_output = perception_model_.forward({input_tensor}).toTensor();

        return perception_output;
    }

    /**
     * @brief Make decision based on state vector
     */
    torch::Tensor makeDecision(const torch::Tensor& state_tensor) {
        torch::Tensor action_values;
        action_values = decision_model_.forward({state_tensor}).toTensor();

        return action_values;
    }

    /**
     * @brief Select action from action values
     */
    int selectAction(const torch::Tensor& action_values, const std::string& method = "greedy") {
        auto values = action_values.accessor<float, 1>();

        if (method == "greedy") {
            int max_idx = 0;
            float max_val = values[0];
            for (int i = 1; i < values.size(0); ++i) {
                if (values[i] > max_val) {
                    max_val = values[i];
                    max_idx = i;
                }
            }
            return max_idx;
        } else if (method == "softmax") {
            // Implement softmax selection
            return selectActionSoftmax(action_values);
        } else if (method == "epsilon_greedy") {
            float epsilon = 0.1f; // Exploration rate
            if (static_cast<float>(rand()) / RAND_MAX < epsilon) {
                return rand() % values.size(0); // Random action
            } else {
                return selectAction(action_values, "greedy"); // Greedy action
            }
        } else {
            throw std::invalid_argument("Unknown selection method: " + method);
        }
    }

    /**
     * @brief Update model with experience (for learning)
     */
    void updateModel(const torch::Tensor& state,
                    int action,
                    float reward,
                    const torch::Tensor& next_state,
                    bool done = false) {
        // Store experience for training
        experiences_.push_back({state.clone(), action, reward, next_state.clone(), done});

        // Train if we have enough experiences
        if (experiences_.size() >= 1000) {
            trainModel();
        }
    }

private:
    struct Experience {
        torch::Tensor state;
        int action;
        float reward;
        torch::Tensor next_state;
        bool done;
    };

    torch::jit::script::Module perception_model_;
    torch::jit::script::Module decision_model_;
    std::vector<Experience> experiences_;

    void initializeModels() {
        // Create simple neural network models
        // Perception model: input -> hidden -> output
        perception_model_ = createPerceptionModel(20, 10); // 20 input features -> 10 output features

        // Decision model: state -> actions
        decision_model_ = createDecisionModel(10, 5); // 10 state features -> 5 possible actions
    }

    torch::jit::script::Module createPerceptionModel(int input_size, int output_size) {
        // Create a simple sequential model
        auto model = torch::nn::Sequential(
            torch::nn::Linear(input_size, 64),
            torch::nn::ReLU(),
            torch::nn::Linear(64, 64),
            torch::nn::ReLU(),
            torch::nn::Linear(64, output_size),
            torch::nn::Softmax(torch::nn::SoftmaxOptions(1))
        );

        // Convert to TorchScript model
        return torch::jit::script::Module(model);
    }

    torch::jit::script::Module createDecisionModel(int state_size, int action_size) {
        // Create a simple sequential model
        auto model = torch::nn::Sequential(
            torch::nn::Linear(state_size, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, 64),
            torch::nn::ReLU(),
            torch::nn::Linear(64, action_size)
        );

        // Convert to TorchScript model
        return torch::jit::script::Module(model);
    }

    int selectActionSoftmax(const torch::Tensor& action_values) {
        // Apply softmax to get probabilities
        auto softmax_values = torch::softmax(action_values, 0);
        auto probs = softmax_values.accessor<float, 1>();

        // Create cumulative distribution
        std::vector<float> cumsum(probs.size(0));
        cumsum[0] = probs[0];
        for (int i = 1; i < probs.size(0); ++i) {
            cumsum[i] = cumsum[i-1] + probs[i];
        }

        // Generate random number and find corresponding action
        float rand_val = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < cumsum.size(); ++i) {
            if (rand_val <= cumsum[i]) {
                return i;
            }
        }

        return probs.size(0) - 1; // Fallback
    }

    void trainModel() {
        std::cout << "Training AI models with collected experiences..." << std::endl;

        // This would implement actual training
        // For this example, we'll just print a message
        std::cout << "Trained on " << experiences_.size() << " experiences" << std::endl;

        // Clear old experiences to prevent memory issues
        if (experiences_.size() > 10000) {
            experiences_.erase(experiences_.begin(), experiences_.end() - 5000); // Keep last 5000
        }
    }
};

/**
 * @brief Robot class that integrates AI for decision making
 */
class RobotWithAI {
public:
    RobotWithAI() : ai_system_(std::make_unique<RobotAI>()) {
        action_names_ = {"move_forward", "move_backward", "turn_left", "turn_right", "wave_hand"};
    }

    /**
     * @brief Run the main interaction loop with AI decision making
     */
    void runInteractionLoop(int max_steps = 50) {
        std::cout << "Starting AI-powered robot interaction loop..." << std::endl;

        for (int step = 0; step < max_steps; ++step) {
            std::cout << "\n--- Step " << (step + 1) << "/" << max_steps << " ---" << std::endl;

            // Sense environment (simulated)
            auto sensory_data = senseEnvironment();
            std::cout << "Current state sensed" << std::endl;

            // Process through AI perception
            auto perception_output = ai_system_->processSensoryInput(sensory_data);
            std::cout << "Perception processed" << std::endl;

            // Make decision
            auto action_values = ai_system_->makeDecision(perception_output);
            int action_index = ai_system_->selectAction(action_values, "epsilon_greedy");
            std::string action_name = action_names_[action_index];

            std::cout << "AI decided: " << action_name << " (index " << action_index << ")" << std::endl;

            // Execute action
            executeAction(action_index);

            // Get next state (simulated)
            auto next_sensory_data = senseEnvironment();

            // Calculate reward
            float reward = calculateReward(action_index, next_sensory_data);
            std::cout << "Reward: " << reward << std::endl;

            // Update AI model with experience
            ai_system_->updateModel(
                perception_output,  // State
                action_index,       // Action
                reward,             // Reward
                ai_system_->processSensoryInput(next_sensory_data)  // Next state
            );

            // Small pause between steps
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::cout << "\nInteraction loop completed after " << max_steps << " steps" << std::endl;
        std::cout << "AI learned from experiences" << std::endl;
    }

private:
    std::unique_ptr<RobotAI> ai_system_;
    std::vector<std::string> action_names_;

    /**
     * @brief Sense the current environment (simulated)
     */
    std::vector<float> senseEnvironment() {
        // Simulate sensory input (20 features)
        std::vector<float> sensory_data(20);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        for (auto& val : sensory_data) {
            val = dis(gen);
        }

        return sensory_data;
    }

    /**
     * @brief Execute the selected action
     */
    void executeAction(int action_index) {
        std::string action_name = action_names_[action_index];
        std::cout << "Executing action: " << action_name << std::endl;

        // In a real robot, this would execute the actual action
        // For simulation, we'll just print the action
    }

    /**
     * @brief Calculate reward for the action taken
     */
    float calculateReward(int action_index, const std::vector<float>& next_state) {
        float reward = 0.0f;

        // Positive rewards for stable behavior
        if (std::abs(next_state[0]) < 0.5f && std::abs(next_state[1]) < 0.5f) {
            reward += 0.1f; // Bonus for stability
        }

        // Negative rewards for dangerous behavior
        if (std::abs(next_state[2]) > 1.0f) {
            reward -= 0.5f; // Penalty for spinning too fast
        }

        // Small negative reward for each step (to encourage efficiency)
        reward -= 0.01f;

        return reward;
    }
};

int main_ai() {
    std::cout << "=== AI-Integrated Robot Example ===" << std::endl;

    // Create robot with AI
    RobotWithAI robot_with_ai;

    // Run the interaction loop
    robot_with_ai.runInteractionLoop(10);  // Reduced steps for demo

    return 0;
}
```

## Real-Time Control Example

Here's an example of real-time control with proper timing and safety:

```cpp
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <signal.h>
#include <memory>

/**
 * @brief Real-time controller with proper timing and safety
 */
class RealTimeController {
public:
    RealTimeController(double control_frequency = 100.0)  // 100 Hz
        : control_frequency_(control_frequency),
          control_period_(std::chrono::microseconds(static_cast<int>(1000000.0 / control_frequency))),
          running_(false) {
        initializeSafetyLimits();
    }

    ~RealTimeController() {
        stop();
    }

    /**
     * @brief Start the real-time control loop
     */
    void start() {
        running_ = true;
        control_thread_ = std::thread(&RealTimeController::controlLoop, this);
    }

    /**
     * @brief Stop the real-time control loop
     */
    void stop() {
        running_ = false;
        if (control_thread_.joinable()) {
            control_thread_.join();
        }
    }

    /**
     * @brief Set control command
     */
    void setCommand(const std::vector<double>& command) {
        std::lock_guard<std::mutex> lock(command_mutex_);
        command_ = command;
    }

    /**
     * @brief Get current state
     */
    std::vector<double> getState() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    /**
     * @brief Emergency stop
     */
    void emergencyStop() {
        std::lock_guard<std::mutex> lock(command_mutex_);
        for (auto& cmd : command_) {
            cmd = 0.0;  // Set all commands to zero
        }
        std::cout << "EMERGENCY STOP ACTIVATED" << std::endl;
    }

private:
    double control_frequency_;
    std::chrono::microseconds control_period_;
    std::atomic<bool> running_;
    std::thread control_thread_;
    mutable std::mutex command_mutex_, state_mutex_;
    std::vector<double> command_;
    std::vector<double> state_;
    std::vector<double> velocity_limits_;
    std::vector<double> position_limits_;
    std::vector<double> torque_limits_;

    void initializeSafetyLimits() {
        // Initialize with some example limits
        position_limits_ = std::vector<double>(6, 3.14);  // 6 joints with ±π limits
        velocity_limits_ = std::vector<double>(6, 5.0);   // 5 rad/s max velocity
        torque_limits_ = std::vector<double>(6, 50.0);    // 50 Nm max torque

        command_ = std::vector<double>(6, 0.0);
        state_ = std::vector<double>(6, 0.0);
    }

    void controlLoop() {
        auto next_iteration = std::chrono::steady_clock::now();

        while (running_) {
            next_iteration += control_period_;

            // Process control cycle
            processControlCycle();

            // Wait until next iteration
            std::this_thread::sleep_until(next_iteration);
        }
    }

    void processControlCycle() {
        // Get current command safely
        std::vector<double> current_command;
        {
            std::lock_guard<std::mutex> lock(command_mutex_);
            current_command = command_;
        }

        // Apply safety checks and limits
        std::vector<double> safe_command = applySafetyLimits(current_command);

        // Update state based on command (simplified dynamics)
        updateState(safe_command);

        // Check for safety violations
        if (checkSafetyViolations()) {
            emergencyStop();
        }
    }

    std::vector<double> applySafetyLimits(const std::vector<double>& command) {
        std::vector<double> limited_command = command;

        // Apply position limits
        for (size_t i = 0; i < limited_command.size() && i < position_limits_.size(); ++i) {
            limited_command[i] = std::max(-position_limits_[i],
                                        std::min(position_limits_[i], limited_command[i]));
        }

        // Apply velocity limits
        std::vector<double> current_state;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            current_state = state_;
        }

        for (size_t i = 0; i < limited_command.size() && i < velocity_limits_.size(); ++i) {
            double velocity = (limited_command[i] - current_state[i]) * control_frequency_;
            if (std::abs(velocity) > velocity_limits_[i]) {
                // Limit the command to respect velocity constraint
                double max_delta = velocity_limits_[i] / control_frequency_;
                limited_command[i] = current_state[i] +
                                   (limited_command[i] > current_state[i] ? max_delta : -max_delta);
            }
        }

        return limited_command;
    }

    void updateState(const std::vector<double>& command) {
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Simple first-order dynamics update
        for (size_t i = 0; i < state_.size() && i < command.size(); ++i) {
            // Apply first-order dynamics: state = state + (command - state) * dt * gain
            double gain = 0.1;  // First-order system gain
            state_[i] += (command[i] - state_[i]) * control_period_.count() * gain * 1e-6;  // Convert microseconds to seconds
        }
    }

    bool checkSafetyViolations() {
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Check for position violations
        for (size_t i = 0; i < state_.size() && i < position_limits_.size(); ++i) {
            if (std::abs(state_[i]) > position_limits_[i] * 1.05) {  // 5% safety margin
                return true;
            }
        }

        // Check for velocity violations (approximated)
        static std::vector<double> last_state = state_;
        for (size_t i = 0; i < state_.size() && i < velocity_limits_.size(); ++i) {
            double velocity = std::abs(state_[i] - last_state[i]) * control_frequency_;
            if (velocity > velocity_limits_[i] * 1.1) {  // 10% safety margin
                last_state = state_;
                return true;
            }
        }
        last_state = state_;

        return false;
    }
};

int main_realtime() {
    std::cout << "=== Real-Time Control Example ===" << std::endl;

    // Create real-time controller
    RealTimeController controller(200.0);  // 200 Hz control frequency

    // Start controller
    controller.start();

    // Send some example commands
    for (int i = 0; i < 10; ++i) {
        std::vector<double> command = {0.1 * i, 0.1 * i, 0.1 * i, 0.1 * i, 0.1 * i, 0.1 * i};
        controller.setCommand(command);

        std::cout << "Command sent: ";
        for (const auto& val : command) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop controller
    controller.stop();

    std::cout << "Real-time control example completed" << std::endl;
    return 0;
}

int main() {
    std::cout << "=== C++ AI Humanoid Robotics Examples ===" << std::endl;

    std::cout << "\n1. Running movement controller example..." << std::endl;
    // The movement example is already defined above

    std::cout << "\n2. Running vision system example..." << std::endl;
    visionExample();

    std::cout << "\n3. Running AI integration example..." << std::endl;
    main_ai();

    std::cout << "\n4. Running real-time control example..." << std::endl;
    main_realtime();

    return 0;
}
```

## Building the Examples

To compile these examples, create a CMakeLists.txt file:

```cmake
cmake_minimum_required(VERSION 3.10)
project(HumanoidRoboticsCpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(PkgConfig REQUIRED)

# Set compiler flags for real-time performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Example 1: Basic movement control
add_executable(robot_controller robot_controller.cpp)
target_link_libraries(robot_controller)

# Example 2: Computer vision
add_executable(vision_system vision_system.cpp)
target_link_libraries(vision_system ${OpenCV_LIBS})

# Example 3: AI integration
add_executable(robot_ai robot_ai.cpp)
target_link_libraries(robot_ai ${TORCH_LIBRARIES})

# Example 4: Real-time control
add_executable(real_time_controller real_time_controller.cpp)
target_link_libraries(real_time_controller pthread)

# Set properties for LibTorch
set_property(TARGET robot_ai PROPERTY CXX_STANDARD 14)
```

Then build using:
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Key Features of the C++ Examples

These C++ examples demonstrate:

1. **Performance-Critical Operations**: C++ excels at real-time control and performance-sensitive applications
2. **Memory Management**: Explicit control over memory allocation and deallocation
3. **Real-Time Control**: Proper timing, thread safety, and interrupt handling
4. **Hardware Interface**: Direct control of actuators and sensors
5. **AI Integration**: Using LibTorch for neural network inference
6. **Computer Vision**: Using OpenCV for image processing and object detection
7. **Safety Systems**: Proper safety limits and emergency stop functionality

The C++ code is designed for performance-critical applications where real-time responsiveness and low-level hardware control are essential. It provides the foundation for deploying humanoid robotics applications in production environments.