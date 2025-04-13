import React from 'react';
import './RecommendButton.css'
const RecommendButton = ({ onRecommend, isLoading, disabled }) => {
  return (
    <button 
      className="recommend-button"
      onClick={onRecommend}
      disabled={isLoading || disabled}
    >
      {isLoading ? (
        <>
          <span className="button-spinner"></span>
          Finding Recommendations...
        </>
      ) : (
        <>
          Get Recommendations
          <span className="button-icon">→</span>
        </>
      )}
    </button>
  );
};

export default RecommendButton;