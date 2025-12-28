import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './RobotSimulator.module.css';

const RobotSimulator = () => {
  const [robotState, setRobotState] = useState({
    position: { x: 50, y: 50 },
    direction: 0, // degrees
    status: 'idle',
    battery: 100,
  });

  const moveRobot = (direction) => {
    setRobotState(prev => {
      let newX = prev.position.x;
      let newY = prev.position.y;

      switch(direction) {
        case 'up':
          newY = Math.max(0, prev.position.y - 10);
          break;
        case 'down':
          newY = Math.min(90, prev.position.y + 10);
          break;
        case 'left':
          newX = Math.max(0, prev.position.x - 10);
          break;
        case 'right':
          newX = Math.min(90, prev.position.x + 10);
          break;
        default:
          break;
      }

      return {
        ...prev,
        position: { x: newX, y: newY },
        status: 'moving'
      };
    });

    setTimeout(() => {
      setRobotState(prev => ({ ...prev, status: 'idle' }));
    }, 500);
  };

  const resetRobot = () => {
    setRobotState({
      position: { x: 50, y: 50 },
      direction: 0,
      status: 'idle',
      battery: 100,
    });
  };

  return (
    <div className={clsx('container', styles.robotSimulator)}>
      <h3>Interactive Robot Simulator</h3>
      <div className={styles.simulationArea}>
        <div
          className={styles.robot}
          style={{
            left: `${robotState.position.x}%`,
            top: `${robotState.position.y}%`,
            transform: `rotate(${robotState.direction}deg)`,
            opacity: robotState.status === 'moving' ? 0.8 : 1
          }}
        >
          🤖
        </div>
        <div className={styles.controls}>
          <div className={styles.directionControls}>
            <button
              className={clsx('button button--primary button--sm', styles.controlButton)}
              onClick={() => moveRobot('up')}
            >
              ↑ Up
            </button>
            <div>
              <button
                className={clsx('button button--primary button--sm', styles.controlButton)}
                onClick={() => moveRobot('left')}
              >
                ← Left
              </button>
              <button
                className={clsx('button button--primary button--sm', styles.controlButton)}
                onClick={() => moveRobot('right')}
              >
                Right →
              </button>
            </div>
            <button
              className={clsx('button button--primary button--sm', styles.controlButton)}
              onClick={() => moveRobot('down')}
            >
              ↓ Down
            </button>
          </div>
          <button
            className={clsx('button button--secondary button--sm', styles.resetButton)}
            onClick={resetRobot}
          >
            Reset Position
          </button>
        </div>
      </div>
      <div className={styles.statusPanel}>
        <div>Position: X: {robotState.position.x}%, Y: {robotState.position.y}%</div>
        <div>Status: {robotState.status}</div>
        <div>Battery: {robotState.battery}%</div>
      </div>
    </div>
  );
};

export default RobotSimulator;