import React, {useState, useMemo} from 'react';
import styles from './styles.module.css';

const DEFAULT_DOCS = [
  {id: 'Doc A', bm25Rank: 1, bgeRank: 5},
  {id: 'Doc B', bm25Rank: 3, bgeRank: 2},
  {id: 'Doc C', bm25Rank: 2, bgeRank: 4},
  {id: 'Doc D', bm25Rank: 5, bgeRank: 1},
  {id: 'Doc E', bm25Rank: 4, bgeRank: 3},
];

export default function RRFcalculator(): React.JSX.Element {
  const [k, setK] = useState(60);
  const [docs, setDocs] = useState(DEFAULT_DOCS);

  const results = useMemo(() => {
    return docs
      .map(d => ({
        ...d,
        rrfScore: 1 / (k + d.bm25Rank) + 1 / (k + d.bgeRank),
      }))
      .sort((a, b) => b.rrfScore - a.rrfScore);
  }, [docs, k]);

  const updateRank = (idx: number, field: 'bm25Rank' | 'bgeRank', value: number) => {
    const newDocs = [...docs];
    newDocs[idx] = {...newDocs[idx], [field]: Math.max(1, Math.min(10, value))};
    setDocs(newDocs);
  };

  const maxScore = Math.max(...results.map(r => r.rrfScore));

  return (
    <div className={styles.container}>
      <div className={styles.controls}>
        <label>
          RRF Constant k:&nbsp;
          <input type="range" min={1} max={120} value={k} onChange={e => setK(Number(e.target.value))} />
          &nbsp;<strong>{k}</strong>
        </label>
      </div>

      <table className={styles.table}>
        <thead>
          <tr>
            <th>Document</th>
            <th>BM25 Rank</th>
            <th>BGE Rank</th>
            <th>RRF Score</th>
            <th>Final Rank</th>
          </tr>
        </thead>
        <tbody>
          {results.map((doc, i) => (
            <tr key={doc.id} className={i === 0 ? styles.winner : ''}>
              <td><strong>{doc.id}</strong></td>
              <td>
                <input type="number" min={1} max={10} value={docs.find(d => d.id === doc.id)?.bm25Rank}
                  onChange={e => { const idx = docs.findIndex(d => d.id === doc.id); updateRank(idx, 'bm25Rank', Number(e.target.value)); }}
                  className={styles.rankInput} />
              </td>
              <td>
                <input type="number" min={1} max={10} value={docs.find(d => d.id === doc.id)?.bgeRank}
                  onChange={e => { const idx = docs.findIndex(d => d.id === doc.id); updateRank(idx, 'bgeRank', Number(e.target.value)); }}
                  className={styles.rankInput} />
              </td>
              <td>
                <div className={styles.scoreBar}>
                  <div className={styles.scoreFill} style={{width: `${(doc.rrfScore / maxScore) * 100}%`}} />
                  <span>{doc.rrfScore.toFixed(4)}</span>
                </div>
              </td>
              <td className={styles.finalRank}>#{i + 1}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className={styles.formula}>
        <code>RRF_score(d) = Σ 1/(k + rank_i(d))</code>
        <p>Try adjusting the ranks above or the k constant to see how the fusion changes!</p>
      </div>
    </div>
  );
}
