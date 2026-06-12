import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/index',
        'getting-started/installation',
        'getting-started/quickstart',
        'getting-started/configuration',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'architecture/index',
        'architecture/data-flow',
        'architecture/tech-stack',
      ],
    },
    {
      type: 'category',
      label: 'Retrieval Module',
      items: [
        'retrieval/index',
        'retrieval/sparse',
        'retrieval/dense',
        'retrieval/static-embedding',
        'retrieval/instruction',
        'retrieval/colbert',
        'retrieval/hybrid',
        'retrieval/reranker',
      ],
    },
    {
      type: 'category',
      label: 'Generation Module',
      items: [
        'generation/index',
        'generation/generators',
        'generation/prompt-engineering',
      ],
    },
    {
      type: 'category',
      label: 'Agentic Workflow',
      items: [
        'agentic-workflow/index',
        'agentic-workflow/decomposition',
        'agentic-workflow/chain-reasoning',
        'agentic-workflow/verification',
        'agentic-workflow/multi-turn',
      ],
    },
    {
      type: 'category',
      label: 'Evaluation',
      items: [
        'evaluation/index',
        'evaluation/retrieval-metrics',
        'evaluation/qa-metrics',
        'evaluation/running-eval',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api-reference/index',
        'api-reference/endpoints',
        'api-reference/database',
      ],
    },
    {
      type: 'category',
      label: 'Deployment',
      items: [
        'deployment/index',
        'deployment/docker',
      ],
    },
  ],
};

export default sidebars;
