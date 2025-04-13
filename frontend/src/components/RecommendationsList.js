import React from 'react';
import './RecommendationsList.css';

const RecommendationsList = ({ recommendations, selectedMovie, showExplanations = false }) => {
  if (!recommendations || recommendations.length === 0) {
    return null;
  }

  return (
    <div className="recommendations-container">
      <h3>Recommended Movies {selectedMovie && `Based on "${selectedMovie}"`}</h3>
      
      <div className="recommendations-list">
        {recommendations.map((recommendation, index) => {
          // Handle different data formats (object or string)
          const title = typeof recommendation === 'object' ? recommendation.title : recommendation;
          const score = typeof recommendation === 'object' ? recommendation.score : null;
          const explanation = typeof recommendation === 'object' ? recommendation.explanation : null;
          
          return (
            <div className="recommendation-item" key={index}>
              <div className="recommendation-header">
                <h4 className="movie-title">{title}</h4>
                {score && (
                  <span className="match-score">
                    {Math.round(score * 100)}% Match
                  </span>
                )}
              </div>
              
              {showExplanations && explanation && (
                <p className="recommendation-explanation">{explanation}</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default RecommendationsList;