import React, {useState} from 'react';
import styles from './styles.module.css';

const RETRIEVERS = [
  {
    name: 'BM25 (Sparse)',
    speed: 95, quality: 60, memory: 90,
    desc: 'Keyword-based matching using term frequency. Fast and interpretable, but cannot understand synonyms or paraphrases.',
    best: 'Exact keyword search, technical terms',
    color: '#2563eb',
  },
  {
    name: 'BGE (Dense)',
    speed: 80, quality: 80, memory: 60,
    desc: 'Neural embedding-based semantic search. Captures meaning but may miss exact keyword matches.',
    best: 'Semantic search, general purpose',
    color: '#7c3aed',
  },
  {
    name: 'Word2Vec',
    speed: 85, quality: 50, memory: 85,
    desc: 'Static word embeddings averaged into document vectors. Lightweight baseline approach.',
    best: 'Resource-constrained environments',
    color: '#0891b2',
  },
  {
    name: 'E5 (Instruction)',
    speed: 75, quality: 82, memory: 55,
    desc: 'Instruction-tuned embeddings that understand task context. Better zero-shot performance.',
    best: 'Task-specific retrieval',
    color: '#059669',
  },
  {
    name: 'ColBERT',
    speed: 40, quality: 90, memory: 30,
    desc: 'Multi-vector late interaction. Token-level matching for best quality, but slow and memory-hungry.',
    best: 'Maximum retrieval quality',
    color: '#dc2626',
  },
  {
    name: 'Hybrid + RRF',
    speed: 75, quality: 92, memory: 55,
    desc: 'Combines BM25 + BGE with Reciprocal Rank Fusion and cross-encoder re-ranking. Best overall.',
    best: 'Production systems',
    color: '#ea580c',
  },
];

function Bar({value, color}: {value: number; color: string}) {
  return (
    <div className={styles.barOuter}>
      <div className={styles.barInner} style={{width: `${value}%`, background: color}} />
    </div>
  );
}

export default function RetrieverComparison(): React.JSX.Element {
  const [selected, setSelected] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'quality' | 'speed' | 'memory'>('quality');

  const sorted = [...RETRIEVERS].sort((a, b) => b[sortBy] - a[sortBy]);

  return (
    <div className={styles.container}>
      <div className={styles.sortControls}>
        Sort by:&nbsp;
        {(['quality', 'speed', 'memory'] as const).map(s => (
          <button key={s} className={`${styles.sortBtn} ${sortBy === s ? styles.sortActive : ''}`}
            onClick={() => setSortBy(s)}>
            {s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>

      <div className={styles.grid}>
        {sorted.map(r => (
          <div key={r.name}
            className={`${styles.card} ${selected === r.name ? styles.cardActive : ''}`}
            onClick={() => setSelected(selected === r.name ? null : r.name)}
            style={{borderLeftColor: r.color}}>
            <div className={styles.cardHeader}>
              <strong>{r.name}</strong>
            </div>
            <div className={styles.bars}>
              <div className={styles.barRow}><span>Quality</span><Bar value={r.quality} color={r.color} /><span>{r.quality}%</span></div>
              <div className={styles.barRow}><span>Speed</span><Bar value={r.speed} color={r.color} /><span>{r.speed}%</span></div>
              <div className={styles.barRow}><span>Memory</span><Bar value={r.memory} color={r.color} /><span>{r.memory}%</span></div>
            </div>
            {selected === r.name && (
              <div className={styles.detail}>
                <p>{r.desc}</p>
                <p><strong>Best for:</strong> {r.best}</p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
