import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const FEATURES = [
  {
    title: 'Hybrid Retrieval',
    icon: '🔍',
    description: 'Combines BM25 keyword search with BGE semantic embeddings using Reciprocal Rank Fusion, plus cross-encoder re-ranking for maximum precision.',
  },
  {
    title: 'Agentic Workflow',
    icon: '🤖',
    description: 'Multi-hop question answering with LLM-driven query decomposition, chain-of-thought reasoning, and answer verification.',
  },
  {
    title: '6 Retrieval Strategies',
    icon: '📊',
    description: 'BM25, BGE, Word2Vec, E5-instruct, ColBERT, and Hybrid — choose the right tool for your use case.',
  },
  {
    title: 'Full Evaluation Suite',
    icon: '📈',
    description: 'nDCG, MAP, Recall for retrieval; EM, F1 for QA; Joint metrics for end-to-end performance measurement.',
  },
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/getting-started">
            Get Started →
          </Link>
        </div>
      </div>
    </header>
  );
}

function Features() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FEATURES.map((f, idx) => (
            <div key={idx} className={clsx('col col--3')}>
              <div className={styles.featureCard}>
                <div className={styles.featureIcon}>{f.icon}</div>
                <Heading as="h3">{f.title}</Heading>
                <p>{f.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Multi-Hop RAG System"
      description="RAG42 - A Retrieval-Augmented Generation system for multi-hop question answering on HotpotQA">
      <HomepageHeader />
      <main>
        <Features />
      </main>
    </Layout>
  );
}
