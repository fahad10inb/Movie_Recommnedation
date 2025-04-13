import React from 'react';
import './LikedMoviesList.css'

const LikedMoviesList = ({ likedMovies, onRemoveMovie }) => {
  if (!likedMovies || likedMovies.length === 0) {
    return null;
  }

  return (
    <div className="liked-movies-container">
      <h3>Movies You Like</h3>
      <div className="liked-movies-list">
        {likedMovies.map((movie, index) => (
          <div className="liked-movie-chip" key={index}>
            <span className="liked-movie-title">{movie}</span>
            <button 
              className="remove-movie-btn"
              onClick={() => onRemoveMovie(index)}
              aria-label="Remove movie"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LikedMoviesList;