import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Interactive Learning',
    description: (
      <>
        Engage with interactive code runners and robot simulators to enhance your learning experience.
      </>
    ),
  },
  {
    title: 'Comprehensive Coverage',
    description: (
      <>
        From foundational concepts to advanced implementations in AI-powered humanoid robotics.
      </>
    ),
  },
  {
    title: 'Practical Examples',
    description: (
      <>
        Real-world code examples in Python and C++ with simulation environments.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}