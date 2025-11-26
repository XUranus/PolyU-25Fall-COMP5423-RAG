import React from 'react';
import './ShimmerEffect.css'; // import the CSS above

const ThinkingGlow = ({ isThinking, text, ...props }) => {
  if (!isThinking) return null;

  return (
    <div className="thinking-text">
      {text}
    </div>
  );
};


export default ThinkingGlow;
