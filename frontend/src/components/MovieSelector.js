import React from 'react';
import './MovieSelector.css'
const MovieSelector = ({ 
  movies, 
  selectedMovie, 
  onSelectMovie, 
  searchResults = [],
  isSearching = false,
  onSearchChange,
  searchQuery = '',
  disabled = false 
}) => {
  return (
    <div className="movie-selector-container">
      {/* Search functionality */}
      <div className="movie-search-container">
        <label htmlFor="movie-search">Search for a movie:</label>
        <div className="search-input-wrapper">
          <input
            type="text"
            id="movie-search"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Type at least 3 characters..."
            disabled={disabled}
            className="search-input"
          />
          {isSearching && <div className="mini-spinner"></div>}
        </div>
      </div>
      
      {/* Search results */}
      {searchResults.length > 0 && (
        <div className="search-results">
          <h4>Search Results:</h4>
          <ul>
            {searchResults.map((movie) => (
              <li 
                key={typeof movie === 'object' ? movie.movieId : movie} 
                onClick={() => onSelectMovie(typeof movie === 'object' ? movie.title : movie)}
              >
                {typeof movie === 'object' ? movie.title : movie}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Dropdown selector (fallback or alternative) */}
      <div className="dropdown-selector">
        <label htmlFor="movie-select">Or select from popular movies:</label>
        <select
          id="movie-select"
          value={selectedMovie}
          onChange={(e) => onSelectMovie(e.target.value)}
          disabled={disabled}
          className="form-select"
        >
          <option value="">-- Select a movie --</option>
          {Array.isArray(movies) && movies.map((movie, index) => (
            <option key={index} value={typeof movie === 'object' ? movie.title : movie}>
              {typeof movie === 'object' ? movie.title : movie}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

export default MovieSelector;