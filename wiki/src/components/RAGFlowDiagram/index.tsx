import React, {useState, useEffect} from 'react';
import styles from './styles.module.css';

const STEPS = [
  {id: 'query', label: 'User Query', icon: '💬', desc: 'User asks a question via the web interface'},
  {id: 'reformulate', label: 'Query Reformulation', icon: '🔄', desc: 'For multi-turn conversations, resolve coreferences like "he", "it" using chat history'},
  {id: 'decompose', label: 'Query Decomposition', icon: '🧩', desc: 'LLM breaks complex multi-hop questions into simpler sub-questions'},
  {id: 'retrieve', label: 'Hybrid Retrieval', icon: '🔍', desc: 'BM25 (keyword) + BGE (semantic) retrieve candidates, fused with RRF'},
  {id: 'rerank', label: 'Cross-Encoder Re-ranking', icon: '📊', desc: 'A cross-encoder model re-scores top candidates for better precision'},
  {id: 'generate', label: 'LLM Generation', icon: '🤖', desc: 'LLM generates answers using retrieved evidence as context'},
  {id: 'synthesize', label: 'Answer Synthesis', icon: '✨', desc: 'Sub-answers are combined into a final concise answer'},
  {id: 'answer', label: 'Final Answer', icon: '✅', desc: 'Clean, post-processed answer returned to the user'},
];

export default function RAGFlowDiagram(): React.JSX.Element {
  const [activeStep, setActiveStep] = useState<number | null>(null);
  const [animating, setAnimating] = useState(false);
  const [currentAnim, setCurrentAnim] = useState(-1);

  const startAnimation = () => {
    setAnimating(true);
    setCurrentAnim(0);
  };

  useEffect(() => {
    if (animating && currentAnim < STEPS.length) {
      const timer = setTimeout(() => setCurrentAnim(currentAnim + 1), 800);
      return () => clearTimeout(timer);
    }
    if (currentAnim >= STEPS.length) {
      setTimeout(() => { setAnimating(false); setCurrentAnim(-1); }, 2000);
    }
  }, [animating, currentAnim]);

  return (
    <div className={styles.container}>
      <div className={styles.controls}>
        <button className={styles.playBtn} onClick={startAnimation} disabled={animating}>
          {animating ? '⏳ Running...' : '▶ Run Pipeline'}
        </button>
      </div>
      <div className={styles.flowGrid}>
        {STEPS.map((step, i) => (
          <React.Fragment key={step.id}>
            <div
              className={`${styles.node} ${activeStep === i ? styles.active : ''} ${
                animating && currentAnim >= i ? styles.animActive : ''
              } ${animating && currentAnim === i ? styles.animCurrent : ''}`}
              onMouseEnter={() => setActiveStep(i)}
              onMouseLeave={() => setActiveStep(null)}
            >
              <span className={styles.icon}>{step.icon}</span>
              <span className={styles.label}>{step.label}</span>
              <span className={styles.stepNum}>{i + 1}</span>
            </div>
            {i < STEPS.length - 1 && (
              <div className={`${styles.arrow} ${
                animating && currentAnim > i ? styles.arrowActive : ''
              }`}>
                →
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
      {activeStep !== null && (
        <div className={styles.description}>
          <strong>Step {activeStep + 1}:</strong> {STEPS[activeStep].desc}
        </div>
      )}
      {animating && currentAnim < STEPS.length && (
        <div className={styles.description}>
          <strong>Processing Step {currentAnim + 1}:</strong> {STEPS[currentAnim].desc}
        </div>
      )}
    </div>
  );
}
