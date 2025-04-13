import React from 'react';
import './GenreSelector.css'
const GenreSelector = ({ genres, selectedGenre, onSelectGenre }) => {
  return (
    <div className="mb-3">
      <label htmlFor="genre" className="form-label">Preferred Genre (optional):</label>
      <select className="form-select" id="genre" value={selectedGenre} onChange={(e) => onSelectGenre(e.target.value)}>
        <option value="">Any</option>
        {genres.map((genre) => (
          <option key={genre} value={genre}>{genre}</option>
        ))}
      </select>
    </div>
  );
};

export default GenreSelector;