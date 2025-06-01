import axios from 'axios';

const API_URL = '/api';

// Create axios instance with base URL
const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// API service functions
const ApiService = {
  // Server status check
  getStatus: async () => {
    try {
      const response = await apiClient.get('/status');
      return response.data;
    } catch (error) {
      console.error('Error checking server status:', error);
      throw error;
    }
  },

  // Prediction API
  makePrediction: async (data) => {
    try {
      const response = await apiClient.post('/predict', data);
      return response.data;
    } catch (error) {
      console.error('Error making prediction:', error);
      throw error;
    }
  },

  // Model management APIs (admin only)
  reloadModels: async () => {
    try {
      const response = await apiClient.post('/reload-models');
      return response.data;
    } catch (error) {
      console.error('Error reloading models:', error);
      throw error;
    }
  },

  analyzeModels: async () => {
    try {
      const response = await apiClient.get('/analyze-models');
      return response.data;
    } catch (error) {
      console.error('Error analyzing models:', error);
      throw error;
    }
  },

  trainElongationModel: async () => {
    try {
      const response = await apiClient.post('/train-elongation-model');
      return response.data;
    } catch (error) {
      console.error('Error training elongation model:', error);
      throw error;
    }
  }
};

export default ApiService; 