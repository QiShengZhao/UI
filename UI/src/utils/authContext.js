import React, { createContext, useState, useContext, useEffect } from 'react';

// Create the authentication context
const AuthContext = createContext();

// Hook to easily use the auth context
export const useAuth = () => useContext(AuthContext);

// Provider component that wraps the app and makes auth object available to any child component
export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userRole, setUserRole] = useState(null);
  const [userName, setUserName] = useState('');
  const [loading, setLoading] = useState(true);

  // Check if user is already logged in on component mount
  useEffect(() => {
    const checkAuth = () => {
      const storedUser = localStorage.getItem('user');
      if (storedUser) {
        const user = JSON.parse(storedUser);
        setIsAuthenticated(true);
        setUserRole(user.role);
        setUserName(user.name);
      }
      setLoading(false);
    };

    checkAuth();
  }, []);

  // Login function
  const login = (userData) => {
    // In a real app, validate credentials from backend
    return new Promise((resolve, reject) => {
      try {
        // Simple validation for demo
        if (userData.username && userData.password) {
          const role = userData.username.includes('admin') ? 'admin' : 'user';
          const user = {
            name: userData.username,
            role: role
          };
          
          // Store in localStorage
          localStorage.setItem('user', JSON.stringify(user));
          
          // Update state
          setIsAuthenticated(true);
          setUserRole(role);
          setUserName(userData.username);
          
          resolve(user);
        } else {
          reject(new Error('Invalid credentials'));
        }
      } catch (error) {
        reject(error);
      }
    });
  };

  // Logout function
  const logout = () => {
    localStorage.removeItem('user');
    setIsAuthenticated(false);
    setUserRole(null);
    setUserName('');
  };

  // Value to be provided to consumers
  const value = {
    isAuthenticated,
    userRole,
    userName,
    loading,
    login,
    logout
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
}; 