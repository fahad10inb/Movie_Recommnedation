import React, { useState, useRef, useEffect } from 'react';
import './AgentChat.css';

// Direct Gemini API endpoint
const API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent';
// Access API key from environment variables, with fallback for development
const GEMINI_API_KEY = process.env.REACT_APP_GEMINI_API_KEY || 'AIzaSyCWSvLErweMxSetPBqRDBPZExh2etpi170';

/**
 * AgentChat component for movie recommendation website
 * @param {string} sessionId - Unique identifier for the chat session
 * @param {function} onRecommendationsReceived - Callback when new recommendations are received
 * @param {array} initialRecommendations - Initial movie recommendations to display (optional)
 * @param {string} userName - User's name if available (optional)
 * @param {boolean} showHeader - Whether to show the chat header (default: true)
 */
const AgentChat = ({
  sessionId,
  onRecommendationsReceived,
  initialRecommendations = [],
  userName = '',
  showHeader = true,
}) => {
  // Sanitize input to prevent XSS
  const sanitizeInput = (text) => {
    if (!text) return '';
    return String(text).replace(/[<>]/g, '');
  };

  // Personalized greeting
  const baseGreeting = "I'm here to recommend movies you'll love. Tell me about the kinds of movies you enjoy!";
  const initialGreeting = userName
    ? `Hello ${sanitizeInput(userName)}! ${baseGreeting}`
    : `Hello! ${baseGreeting}`;
  const [messages, setMessages] = useState([
    { role: 'agent', content: initialGreeting },
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [apiError, setApiError] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // Check API key on component mount
  useEffect(() => {
    if (!GEMINI_API_KEY) {
      console.error('Gemini API key is missing');
      setApiError(true);
    } else {
      console.log('Gemini API key is configured');
      setApiError(false);
    }
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus input field on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Display initial recommendations if provided
  useEffect(() => {
    if (initialRecommendations.length > 0) {
      onRecommendationsReceived(initialRecommendations, messages[0].content);
    }
  }, [initialRecommendations, onRecommendationsReceived, messages]);

  // Scroll function
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Simulate typing delay
  const simulateTyping = (responseLength) => {
    setIsTyping(true);
    const typingTime = Math.min(Math.max(responseLength * 20, 500), 2000);
    return new Promise((resolve) => setTimeout(resolve, typingTime));
  };

  // Format chat history for Gemini API
  const formatChatHistory = (chatHistory) => {
    return chatHistory.map(msg => ({
      role: msg.role === 'user' ? 'user' : 'model',
      parts: [{ text: msg.content }]
    }));
  };

  // Create prompt for Gemini API with improved formatting instructions
  const createPrompt = (userInput, chatHistory) => {
    // Convert previous messages to Gemini format
    const formattedHistory = formatChatHistory(chatHistory);
    
    // Add the new user message with enhanced formatting instructions
    const contents = [
      ...formattedHistory,
      {
        role: "user",
        parts: [
          {
            text: `You are a movie recommendation expert. Keep your responses brief and well-formatted. 
                  Your response should follow this structure exactly:
                  
                  1. A very brief acknowledgment of the user's input (1 sentence max)
                  2. Exactly 3 movie recommendations, formatted as individual sentences using this EXACT template for each movie: 
                     "[Title] ([Year]) is a [Genre] film about [one-line description]." 
                  3. A short follow-up question (1 sentence)
                  
                  CRITICAL FORMATTING RULES:
                  - NEVER use special characters like ".", "*", or any other symbols in the title or anywhere in your response
                  - ALWAYS format each title correctly - no periods, asterisks or other symbols inside the title
                  - ALWAYS include the year in parentheses immediately after the title
                  - Each movie recommendation MUST be a complete, standalone sentence
                  - Use only plain text - no formatting, no bullet points
                  - Verify each title is correctly formatted before sending
                  
                  Example of CORRECT formatting:
                  "Thanks for sharing your interest in sci-fi! Star Wars (1977) is a Sci-Fi/Adventure film about a farm boy who joins rebels to save a princess from an evil empire. The Matrix (1999) is a Sci-Fi/Action film about a computer hacker who discovers reality is a simulation created by machines. Blade Runner (1982) is a Sci-Fi/Noir film about a detective hunting rogue androids in a dystopian future. Which of these themes appeals to you most?"
                  
                  User input: ${userInput}
                  
                  Always include recommendations in a JSON object at the end of the response, enclosed in triple backticks, like:
                  \`\`\`json
                  {
                    "recommendations": [
                      {"title": "Movie Name", "year": "Year", "genre": "Genre", "description": "Very short description"},
                      ...
                    ]
                  }
                  \`\`\``
          }
        ]
      }
    ];
    
    return {
      contents,
      generationConfig: {
        temperature: 0.7,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 400,
      }
    };
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isSending || apiError) return;

    const userMessage = sanitizeInput(input.trim());
    setInput('');
    setIsSending(true);

    // Add user message to chat
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);

    try {
      // Show typing indicator
      setIsTyping(true);

      // Create prompt with chat history
      const prompt = createPrompt(userMessage, messages);

      // Fetch response from Gemini API with retry logic
      const retryFetch = async (url, options, retries = 3, delay = 1000) => {
        for (let i = 0; i < retries; i++) {
          try {
            // Add API key as query parameter
            const apiUrl = `${url}?key=${GEMINI_API_KEY}`;
            const response = await fetch(apiUrl, options);
            if (response.status === 429) {
              throw new Error('Rate limit exceeded');
            }
            if (!response.ok) {
              throw new Error(`Gemini API request failed: ${response.statusText}`);
            }
            return response;
          } catch (error) {
            console.error(`Attempt ${i + 1} failed:`, error);
            if (i === retries - 1 || error.message !== 'Rate limit exceeded') {
              throw error;
            }
            await new Promise((resolve) => setTimeout(resolve, delay * (i + 1)));
          }
        }
      };

      const response = await retryFetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(prompt)
      });

      const data = await response.json();
      if (!data.candidates?.[0]?.content?.parts?.[0]?.text) {
        throw new Error('Unexpected Gemini API response format');
      }
      const agentReply = data.candidates[0].content.parts[0].text;

      // Format and clean response
      const formattedReply = formatGeminiResponse(agentReply);

      // Simulate typing delay
      await simulateTyping(formattedReply.length);

      // Parse recommendations from the response
      let recommendations = [];
      try {
        const jsonMatch = agentReply.match(/```json\n([\s\S]*?)\n```/);
        if (jsonMatch?.[1]) {
          const parsed = JSON.parse(jsonMatch[1]);
          recommendations = parsed.recommendations || [];
        }
      } catch (parseError) {
        console.warn('Could not parse recommendations:', parseError);
      }

      // Add agent response to chat
      setMessages((prev) => [
        ...prev,
        { role: 'agent', content: formattedReply },
      ]);

      // Trigger callback with recommendations or show fallback
      if (recommendations.length > 0) {
        onRecommendationsReceived(recommendations, formattedReply);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: 'agent',
            content: "I couldn't find specific movies. Could you share a favorite film or genre?",
          },
        ]);
      }
    } catch (error) {
      console.error('Error with Gemini API:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'agent',
          content: "Sorry, something went wrong. What kinds of movies do you enjoy?",
        },
      ]);
    } finally {
      setIsTyping(false);
      setIsSending(false);
      inputRef.current?.focus();
    }
  };

  // Enhanced formatting function with post-processing to fix specific issues
  const formatGeminiResponse = (rawResponse) => {
    // Remove the JSON block
    let cleanResponse = rawResponse.replace(/```json\n[\s\S]*?\n```/, '').trim();
    
    // Step 1: Basic cleanup
    cleanResponse = cleanResponse
      // Remove all markdown formatting
      .replace(/\*+([^*]+)\*+/g, '$1')
      .replace(/_+([^_]+)_+/g, '$1')
      .replace(/\#+\s/g, '')
      .replace(/\~\~([^~]+)\~\~/g, '$1')
      .replace(/\`([^`]+)\`/g, '$1')
      // Remove bullet points and list markers
      .replace(/^[\s-]*[-•*+][\s]*/gm, '')
      .replace(/^\s*\d+\.\s*/gm, '')
      // Standardize whitespace
      .replace(/\s+/g, ' ')
      .replace(/\n+/g, ' ')
      .trim();
    
    // Step 2: Fix specific formatting issues with periods in movie titles
    cleanResponse = cleanResponse
      // Remove erroneous periods in titles (like "The . Force Awakens")
      .replace(/([A-Z][a-z]+)\s+\.\s+([A-Z][a-z]+)/g, '$1 $2')
      // Fix periods appearing at the end of titles before year
      .replace(/\.\s+\((\d{4})\)/g, ' ($1)')
      // Fix periods appearing inside titles
      .replace(/([A-Z][a-z]+)\s+\.\s+([A-Z][a-z]+)/g, '$1 $2')
      // Remove any stray dots that shouldn't be there
      .replace(/\s\.\s/g, ' ')
      // Fix periods where they should be sentence endings
      .replace(/\.\s+([A-Z])/g, '. $1')
      .trim();
    
    // Step 3: Make sure each movie recommendation is properly formatted
    // Using regex to identify movie patterns and ensure proper formatting
    const moviePattern = /([A-Za-z0-9\s&:;,'!?-]+)\s\((\d{4})\)\s/g;
    cleanResponse = cleanResponse.replace(moviePattern, (match) => {
      // This ensures the movie title and year are properly formatted
      return match.replace(/\s{2,}/g, ' ');
    });
    
    // Step 4: Final cleanup
    cleanResponse = cleanResponse
      // Fix any remaining double spaces
      .replace(/\s{2,}/g, ' ')
      // Fix any remaining double periods
      .replace(/\.{2,}/g, '.')
      // Ensure proper spacing after periods
      .replace(/\.([A-Z])/g, '. $1')
      // Remove unnecessary prefixes
      .replace(/^(Here are|I recommend|Check out|You might enjoy)\s+/i, '')
      .trim();
    
    // Step 5: Post-processing validation for movie recommendations
    // Look for patterns like "Title (Year) is a Genre"
    const movieTitlePattern = /([A-Za-z0-9\s&:;,'!?-]+)\s\((\d{4})\)\sis\sa/g;
    let matches = [...cleanResponse.matchAll(movieTitlePattern)];
    
    // If we have recommendations, ensure they're formatted properly
    if (matches.length > 0) {
      matches.forEach(match => {
        const fullMatch = match[0];
        const title = match[1].trim();
        
        // Check if the title contains any periods that shouldn't be there
        if (title.includes('.') && !title.match(/[A-Z]\./)) { // Exclude legitimate abbreviations
          const fixedTitle = title.replace(/\./g, '');
          cleanResponse = cleanResponse.replace(fullMatch, fullMatch.replace(title, fixedTitle));
        }
      });
    }
    
    return cleanResponse;
  };

  // Handle input changes
  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  // Handle keypress events
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  // Render error message if API key is missing
  if (apiError) {
    return (
      <div className="agent-chat-container">
        <div className="error-message">
          <p>Sorry, the movie recommendation service is currently unavailable.</p>
          <p>Please check your API configuration and try again later.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="agent-chat-container">
      {showHeader && (
        <div className="agent-header">
          <h3>Movie AI Assistant</h3>
          <p className="agent-subtitle">Chat with me about movies you like, and I'll find recommendations!</p>
        </div>
      )}

      <div className="messages-container" ref={messagesContainerRef}>
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-content">{msg.content}</div>
          </div>
        ))}
        {isTyping && (
          <div className="message agent">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={input}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder="Tell me what movies you like..."
          className="chat-input"
          disabled={isSending || apiError}
          ref={inputRef}
          aria-label="Chat message input"
        />
        <button
          type="submit"
          className="send-button"
          disabled={isSending || !input.trim() || apiError}
          aria-label="Send message"
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default AgentChat;