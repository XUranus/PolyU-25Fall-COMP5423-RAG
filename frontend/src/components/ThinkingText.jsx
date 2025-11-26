import React, { useState, useEffect } from 'react';
import './ShimmerEffect.css'; // import the CSS above

const ThinkingGlow = ({ isThinking, text }) => {
  if (!isThinking) return null;

  return (
    <div className="thinking-text">
      {text}
    </div>
  );
};


export default ThinkingGlow;
