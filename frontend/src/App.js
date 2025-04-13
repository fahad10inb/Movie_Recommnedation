// App.js - Updated to work with Gemini API-based AgentChat
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import './App.css';

// Import components
import MovieSelector from './components/MovieSelector';
import GenreSelector from './components/GenreSelector';
import RecommendButton from './components/RecommendButton';
import RecommendationsList from './components/RecommendationsList';
import LikedMoviesList from './components/LikedMoviesList';
import AgentChat from './components/AgentChat';

// Configure axios base URL for backend
axios.defaults.baseURL = 'http://localhost:5000';

function App() {
  // Data state
  const [movies, setMovies] = useState([]);
  const [genres, setGenres] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  
  // UI state
  const [isLoading, setIsLoading] = useState(true);
  const [isRecommending, setIsRecommending] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [activeView, setActiveView] = useState('standard'); // 'standard' or 'agent'
  
  // User selections
  const [selectedMovie, setSelectedMovie] = useState('');
  const [selectedGenre, setSelectedGenre] = useState('');
  const [likedMovies, setLikedMovies] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Session ID for agent conversations
  const [sessionId] = useState(uuidv4());
  const [agentReplies, setAgentReplies] = useState([]);

  // Check backend connection and fetch initial data
  useEffect(() => {
    const checkConnection = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Test API connection with a health check
        const healthResponse = await axios.get('/api/health');
        console.log('Health check response:', healthResponse.data);
        setConnectionStatus('connected');
        
        // Fetch movies and genres
        const response = await axios.get('/api/movies');
        console.log('Movies data received:', response.data);
        
        if (response.data?.movies) {
          setMovies(response.data.movies);
        } else {
          console.error('Movies data not in expected format:', response.data);
          setError('Received invalid movies data from server');
        }
        
        if (response.data?.genres) {
          setGenres(response.data.genres);
        } else {
          console.error('Genres data not in expected format:', response.data);
          setError('Received invalid genres data from server');
        }
      } catch (error) {
        console.error('Connection error:', error);
        setConnectionStatus('failed');
        setError(
          error.response?.data?.error ||
          'Failed to connect to the movie recommendation service. Check if the backend server is running.'
        );
      } finally {
        setIsLoading(false);
      }
    };

    checkConnection();
  }, []);

  // Handle movie search
  const handleSearch = async (query) => {
    if (!query || query.length < 3) {
      setSearchResults([]);
      return;
    }

    try {
      setIsSearching(true);
      setError(null);
      
      const response = await axios.get(`/api/search?q=${encodeURIComponent(query)}`);
      console.log('Search results:', response.data);
      
      if (response.data?.results) {
        setSearchResults(response.data.results);
      } else {
        setSearchResults([]);
      }
    } catch (error) {
      console.error('Error searching movies:', error);
      setSearchResults([]);
      setError('Failed to search movies. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  // Debounce search
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchQuery.length >= 3) {
        handleSearch(searchQuery);
      } else {
        setSearchResults([]);
      }
    }, 300);
    
    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  // Handle standard recommendation
  const handleRecommend = async () => {
    if (!selectedMovie) {
      setError('Please select a movie');
      return;
    }

    try {
      setIsRecommending(true);
      setError(null);
      
      console.log('Requesting recommendations for:', selectedMovie, 'Genre:', selectedGenre);
      
      const response = await axios.post('/api/recommend', {
        movie: selectedMovie,
        genre: selectedGenre || '',
        num: 10,
      });
      
      console.log('Recommendation response:', response.data);
      
      if (response.data?.recommendations) {
        if (response.data.recommendations.length > 0) {
          setRecommendations(response.data.recommendations);
          
          if (!response.data.exact_match) {
            console.info(`Requested "${selectedMovie}" but matched to "${response.data.selected_movie}"`);
          }
        } else {
          setError('No recommendations found for this movie. Try another one or a different genre.');
        }
      } else {
        setError('Received invalid recommendation data');
        console.error('Invalid recommendation response:', response.data);
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      const errorMessage = error.response?.data?.error || 'Failed to get recommendations. Please try again.';
      setError(errorMessage);
    } finally {
      setIsRecommending(false);
    }
  };

  // Handle AI agent recommendation with multiple movies
  const handleAgentRecommend = async () => {
    if (likedMovies.length === 0) {
      setError('Please add at least one movie you like');
      return;
    }

    try {
      setIsRecommending(true);
      setError(null);
      
      console.log('Requesting agent recommendations for:', likedMovies, 'Genre:', selectedGenre);
      
      const response = await axios.post('/api/agent', {
        liked_movies: likedMovies,
        genre: selectedGenre || '',
        session_id: sessionId,
      });
      
      console.log('Agent recommendation response:', response.data);
      
      if (response.data?.recommendations) {
        if (response.data.recommendations.length > 0) {
          setRecommendations(response.data.recommendations);
          
          if (response.data.agent_reply) {
            setAgentReplies((prev) => [...prev, response.data.agent_reply]);
          }
          
          if (response.data.matched_titles && response.data.original_titles) {
            const mismatches = response.data.matched_titles.filter(
              (title, i) => title.toLowerCase() !== response.data.original_titles[i].toLowerCase()
            );
            if (mismatches.length > 0) {
              console.info('Some movie titles were matched to different titles:', mismatches);
            }
          }
        } else {
          setError('No recommendations found based on your preferences. Try different movies or genres.');
        }
      } else {
        setError('Received invalid recommendation data from agent');
      }
    } catch (error) {
      console.error('Error fetching agent recommendations:', error);
      const errorMessage = error.response?.data?.error || error.message;
      setError(`Failed to get recommendations: ${errorMessage}`);
    } finally {
      setIsRecommending(false);
    }
  };

  // Add movie to liked list
  const handleAddLikedMovie = () => {
    if (!selectedMovie) {
      setError('Please select a movie first');
      return;
    }
    
    if (!likedMovies.includes(selectedMovie)) {
      setLikedMovies([...likedMovies, selectedMovie]);
      setSelectedMovie('');
      setSearchQuery('');
      setSearchResults([]);
    } else {
      setError('This movie is already in your liked list');
    }
  };

  // Remove movie from liked list
  const handleRemoveLikedMovie = (index) => {
    const updatedLikes = [...likedMovies];
    updatedLikes.splice(index, 1);
    setLikedMovies(updatedLikes);
  };

  // Handle recommendations from chat agent
  const handleAgentRecommendations = (recs, agentReply) => {
    if (recs?.length > 0) {
      // Transform the recommendations to match the expected format
      const formattedRecs = recs.map((rec) => ({
        title: rec.title,
        genres: [rec.genre], // Convert single genre to array format
        overview: rec.description,
        similarity: 'Recommended by AI',
        vote_average: 0,
      }));
      
      setRecommendations(formattedRecs);
      console.log('Recommendations from agent chat:', formattedRecs);
      
      if (agentReply) {
        setAgentReplies((prev) => [...prev, agentReply]);
      }
    } else {
      console.log('No recommendations received from agent chat');
      setError('No recommendations found. Try providing more details in the chat.');
    }
  };

  // Select movie from search results
  const handleSelectFromSearch = (movieTitle) => {
    setSelectedMovie(movieTitle);
    setSearchQuery('');
    setSearchResults([]);
  };

  // Loading and error states
  if (connectionStatus === 'checking' || isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-card">
          <div className="spinner"></div>
          <h2>Connecting to movie database...</h2>
        </div>
      </div>
    );
  }

  if (connectionStatus === 'failed') {
    return (
      <div className="loading-container">
        <div className="error-card">
          <div className="error-icon">⚠️</div>
          <h2>Connection Failed</h2>
          <p>{error || 'Unable to connect to the movie recommendation service.'}</p>
          <p className="error-details">Make sure your backend server is running at {axios.defaults.baseURL}</p>
          <button 
            className="retry-button"
            onClick={() => window.location.reload()}
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <div className="content-container">
        {/* Header */}
        <div className="app-header">
          <div className="title-container">
            <span className="movie-icon">🎬</span>
            <h1>Movie Recommender AI</h1>
          </div>
          <p className="subtitle">Find your next favorite movie with our intelligent recommendation engine</p>
          
          {/* View Toggle */}
          <div className="view-toggle">
            <button 
              className={`toggle-btn ${activeView === 'standard' ? 'active' : ''}`}
              onClick={() => setActiveView('standard')}
            >
              Standard Mode
            </button>
            <button 
              className={`toggle-btn ${activeView === 'agent' ? 'active' : ''}`}
              onClick={() => setActiveView('agent')}
            >
              AI Assistant Mode
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-content">
          {/* Show components in standard mode or if agent mode is selected */}
          <div className={activeView === 'agent' ? 'hidden' : ''}>
            {/* Unified Movie Search */}
            <div className="app-card search-card">
              <div className="card-body">
                {error && (
                  <div className="error-alert">
                    <span className="alert-icon">⚠️</span>
                    <p>{error}</p>
                  </div>
                )}
                
                <div className="movie-search-container">
                  <label htmlFor="movie-search">Search for a movie:</label>
                  <input
                    type="text"
                    id="movie-search"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Type at least 3 characters..."
                    disabled={isRecommending}
                    className="search-input"
                  />
                  {isSearching && <div className="mini-spinner"></div>}
                </div>
                
                {searchResults.length > 0 && (
                  <div className="search-results">
                    <h4>Search Results:</h4>
                    <ul>
                      {searchResults.map((movie) => (
                        <li key={movie.movieId} onClick={() => handleSelectFromSearch(movie.title)}>
                          {movie.title}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            {/* Standard Components */}
            <div className="app-card">
              <div className="card-body">
                <div className="movie-selection-display">
                  <h3>Selected Movie:</h3>
                  <div className="selected-movie">
                    {selectedMovie ? (
                      <p>{selectedMovie}</p>
                    ) : (
                      <p className="text-muted">Search and select a movie above</p>
                    )}
                  </div>
                </div>

                <GenreSelector 
                  genres={genres} 
                  selectedGenre={selectedGenre} 
                  onSelectGenre={setSelectedGenre}
                  disabled={isRecommending}
                />

                <RecommendButton 
                  onRecommend={handleRecommend}
                  isLoading={isRecommending}
                  disabled={!selectedMovie}
                />

                {recommendations.length > 0 && (
                  <RecommendationsList 
                    recommendations={recommendations}
                    selectedMovie={selectedMovie}
                    showExplanations={true}
                  />
                )}
              </div>
            </div>
          </div>

          {/* AI Assistant Mode - Only show the chatbot */}
          {activeView === 'agent' && (
            <div className="app-card">
              <div className="card-body">
                <AgentChat 
                  sessionId={sessionId}
                  onRecommendationsReceived={handleAgentRecommendations}
                  initialRecommendations={recommendations.length > 0 ? recommendations : []}
                  userName="" // Optional: Add user name if available
                  showHeader={true}
                />
                
                {/* Only show recommendations from AI chat if available */}
                {recommendations.length > 0 && (
                  <RecommendationsList 
                    recommendations={recommendations}
                    showExplanations={true}
                  />
                )}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="app-footer">
          <div className="connection-status">
            <p>Status: 
              <span className={connectionStatus === 'connected' ? 'status-connected' : 'status-disconnected'}>
                {connectionStatus === 'connected' ? ' Connected to AI recommendation engine' : ' Disconnected'}
              </span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;