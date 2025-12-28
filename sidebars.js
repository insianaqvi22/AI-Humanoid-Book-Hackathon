// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: ['getting-started/overview', 'getting-started/foundations', 'getting-started/prerequisites'],
    },
    {
      type: 'category',
      label: 'AI Concepts',
      items: ['ai-concepts/machine-learning', 'ai-concepts/neural-networks', 'ai-concepts/computer-vision'],
    },
    {
      type: 'category',
      label: 'Humanoid Design',
      items: ['humanoid-design/mechanical-structure', 'humanoid-design/sensors-actuators', 'humanoid-design/locomotion'],
    },
    {
      type: 'category',
      label: 'Implementation',
      items: ['implementation/basic-movements', 'implementation/perception-systems', 'implementation/ai-integration'],
    },
    {
      type: 'category',
      label: 'Advanced Topics',
      items: ['advanced-topics/human-robot-interaction', 'advanced-topics/safety-considerations', 'advanced-topics/future-directions'],
    },
    {
      type: 'category',
      label: 'Code Examples',
      items: [
        'code-examples/python/intro',
        'code-examples/cpp/intro',
        'code-examples/simulation/intro'
      ],
    },
  ],
};

module.exports = sidebars;