---
title: System Integration
sidebar_position: 3
---

import CodeRunner from '@site/src/components/CodeRunner';

# System Integration in Python

This section covers system integration for humanoid robots using Python.

## Component Integration

How to integrate different robot components:

```python
import asyncio
import threading
from typing import Dict, Any

class RobotSystemIntegrator:
    def __init__(self):
        self.components = {}
        self.is_running = False

    def register_component(self, name: str, component):
        """Register a robot component"""
        self.components[name] = component
        print(f"Registered component: {name}")

    def integrate_systems(self):
        """Integrate all registered components"""
        print("Starting system integration...")

        # Initialize all components
        for name, component in self.components.items():
            if hasattr(component, 'initialize'):
                component.initialize()

        print("System integration complete")

    async def run_system(self):
        """Run the integrated robot system"""
        self.is_running = True

        while self.is_running:
            # Update all components
            for name, component in self.components.items():
                if hasattr(component, 'update'):
                    component.update()

            await asyncio.sleep(0.01)  # 10ms update rate
```

## Communication Protocols

Implementing communication between components:

```python
import json
import queue
import time

class ComponentCommunicator:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.subscribers = {}

    def subscribe(self, topic: str, callback):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message to a topic"""
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(message)

    def send_command(self, component: str, command: str, params: Dict[str, Any] = None):
        """Send command to a specific component"""
        message = {
            'component': component,
            'command': command,
            'params': params or {},
            'timestamp': time.time()
        }
        self.publish(f'command/{component}', message)
```

## Data Flow Management

Managing data flow between components:

```python
class DataFlowManager:
    def __init__(self):
        self.data_buffers = {}
        self.processing_pipelines = {}

    def create_pipeline(self, name: str, processors: list):
        """Create a data processing pipeline"""
        self.processing_pipelines[name] = processors

    def process_data(self, pipeline_name: str, data: Any):
        """Process data through a pipeline"""
        if pipeline_name not in self.processing_pipelines:
            return data

        result = data
        for processor in self.processing_pipelines[pipeline_name]:
            result = processor(result)

        return result

    def buffer_data(self, key: str, data: Any, max_size: int = 100):
        """Buffer data with size limit"""
        if key not in self.data_buffers:
            self.data_buffers[key] = []

        self.data_buffers[key].append(data)

        if len(self.data_buffers[key]) > max_size:
            self.data_buffers[key].pop(0)
```

## Integration Example

Complete integration example:

```python
# Example of integrating sensor, control, and actuator systems
class RobotIntegrationExample:
    def __init__(self):
        self.integrator = RobotSystemIntegrator()
        self.communicator = ComponentCommunicator()
        self.data_manager = DataFlowManager()

        # Register components
        self.integrator.register_component('sensors', self.create_sensor_system())
        self.integrator.register_component('controller', self.create_controller())
        self.integrator.register_component('actuators', self.create_actuator_system())

        # Set up communication
        self.communicator.subscribe('sensor_data', self.handle_sensor_data)
        self.communicator.subscribe('control_commands', self.handle_control_commands)

    def create_sensor_system(self):
        """Create sensor system component"""
        class SensorSystem:
            def initialize(self):
                print("Sensor system initialized")

            def update(self):
                # Simulate sensor reading
                pass

        return SensorSystem()

    def create_controller(self):
        """Create controller component"""
        class Controller:
            def initialize(self):
                print("Controller initialized")

            def update(self):
                # Process control logic
                pass

        return Controller()

    def create_actuator_system(self):
        """Create actuator system component"""
        class ActuatorSystem:
            def initialize(self):
                print("Actuator system initialized")

            def update(self):
                # Execute actuator commands
                pass

        return ActuatorSystem()

    def handle_sensor_data(self, message):
        """Handle incoming sensor data"""
        print(f"Received sensor data: {message}")

    def handle_control_commands(self, message):
        """Handle incoming control commands"""
        print(f"Received control command: {message}")
```

## Interactive Integration Demo

<CodeRunner language="python" title="System Integration Demo">

```python
class SimpleRobotIntegrator:
    def __init__(self):
        self.sensors = {'position': [0, 0, 0], 'orientation': [0, 0, 0, 1]}
        self.controller = {'target': [1, 0, 0]}
        self.actuators = {'motors': [0, 0, 0, 0]}

    def update_sensors(self):
        import random
        self.sensors['position'] = [p + random.uniform(-0.01, 0.01) for p in self.sensors['position']]
        return self.sensors

    def run_controller(self):
        current_pos = self.sensors['position']
        target_pos = self.controller['target']
        # Simple proportional control
        error = [t - c for t, c in zip(target_pos, current_pos)]
        motor_commands = [e * 0.1 for e in error[:3]]  # Use first 3 elements
        self.actuators['motors'] = motor_commands
        return motor_commands

    def execute_actuators(self):
        print(f"Actuating with commands: {self.actuators['motors']}")

    def step(self):
        self.update_sensors()
        commands = self.run_controller()
        self.execute_actuators()
        print(f"Current position: {self.sensors['position']}")

# Create and run the integrator
integrator = SimpleRobotIntegrator()
for i in range(5):
    print(f"Step {i+1}:")
    integrator.step()
    print()
```

</CodeRunner>