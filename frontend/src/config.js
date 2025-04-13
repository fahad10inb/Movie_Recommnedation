export const API_BASE_URL = 'http://localhost:5000';

// API endpoints
export const API_ENDPOINTS = {
  HEALTH: '/api/health',
  MOVIES: '/api/movies',
  RECOMMEND: '/api/recommend',
  AGENT: '/api/agent',
  AGENT_SESSION: '/api/agent/session'
};

// Default settings
export const DEFAULT_SETTINGS = {
  MAX_RECOMMENDATIONS: 10,
  SIMILARITY_THRESHOLD: 0.5
};