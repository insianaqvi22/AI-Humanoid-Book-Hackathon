import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './CodeRunner.module.css';

const CodeRunner = ({ language = 'python', children, title = 'Interactive Code' }) => {
  const [code, setCode] = useState(children ? (typeof children === 'string' ? children.trim() : (children.props?.children || '')) : '');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  const runCode = () => {
    setIsRunning(true);
    setOutput('Running code...');

    // Simulate code execution (in a real implementation, this would connect to a backend service)
    setTimeout(() => {
      setOutput(`// ${language.toUpperCase()} code executed successfully\n// Output would appear here in a real implementation\n// This is a simulation of code execution`);
      setIsRunning(false);
    }, 1500);
  };

  const copyCode = () => {
    navigator.clipboard.writeText(code);
    alert('Code copied to clipboard!');
  };

  return (
    <div className={clsx('container', styles.codeRunner)}>
      <div className={styles.header}>
        <h3>{title}</h3>
        <div className={styles.actions}>
          <button
            className={clsx('button button--primary button--sm', styles.runButton)}
            onClick={runCode}
            disabled={isRunning}
          >
            {isRunning ? 'Running...' : '▶ Run Code'}
          </button>
          <button
            className={clsx('button button--secondary button--sm', styles.copyButton)}
            onClick={copyCode}
          >
            📋 Copy
          </button>
        </div>
      </div>

      <div className={styles.editor}>
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className={styles.codeInput}
          placeholder={`Enter your ${language} code here...`}
          rows={8}
        />
      </div>

      <div className={styles.output}>
        <h4>Output:</h4>
        <pre className={styles.outputContent}>
          {output || '// Output will appear here after running the code'}
        </pre>
      </div>
    </div>
  );
};

export default CodeRunner;