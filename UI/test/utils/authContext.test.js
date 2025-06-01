const React = require('react');
const { render, act, waitFor } = require('@testing-library/react');
const { AuthProvider, useAuth } = require('../../src/utils/authContext');

// Test component that uses the auth context
const TestComponent = () => {
  const { isAuthenticated, userRole, userName, login, logout } = useAuth();
  
  return (
    <div>
      <div data-testid="auth-status">{isAuthenticated ? 'logged-in' : 'logged-out'}</div>
      <div data-testid="user-role">{userRole || 'no-role'}</div>
      <div data-testid="user-name">{userName || 'no-name'}</div>
      <button 
        data-testid="login-button" 
        onClick={() => login({ username: 'testuser', password: 'password' })}
      >
        Login
      </button>
      <button data-testid="logout-button" onClick={logout}>Logout</button>
    </div>
  );
};

describe('AuthContext', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();
  });

  test('provides initial unauthenticated state', async () => {
    const { getByTestId } = render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );
    
    await waitFor(() => {
      expect(getByTestId('auth-status')).toHaveTextContent('logged-out');
      expect(getByTestId('user-role')).toHaveTextContent('no-role');
      expect(getByTestId('user-name')).toHaveTextContent('no-name');
    });
  });

  test('authenticates user on login', async () => {
    const { getByTestId } = render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );
    
    // Click login button to trigger login
    await act(async () => {
      getByTestId('login-button').click();
    });
    
    // Check that state is updated
    await waitFor(() => {
      expect(getByTestId('auth-status')).toHaveTextContent('logged-in');
      expect(getByTestId('user-role')).toHaveTextContent('user');
      expect(getByTestId('user-name')).toHaveTextContent('testuser');
    });
    
    // Check localStorage
    const storedUser = JSON.parse(localStorage.getItem('user'));
    expect(storedUser).toEqual({ name: 'testuser', role: 'user' });
  });

  test('logs out user', async () => {
    const { getByTestId } = render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );
    
    // Login first
    await act(async () => {
      getByTestId('login-button').click();
    });
    
    // Then logout
    await act(async () => {
      getByTestId('logout-button').click();
    });
    
    // Check that state is updated
    await waitFor(() => {
      expect(getByTestId('auth-status')).toHaveTextContent('logged-out');
      expect(getByTestId('user-role')).toHaveTextContent('no-role');
      expect(getByTestId('user-name')).toHaveTextContent('no-name');
    });
    
    // Check localStorage
    expect(localStorage.getItem('user')).toBeNull();
  });

  test('loads user from localStorage on mount', async () => {
    // Set user in localStorage before rendering
    localStorage.setItem('user', JSON.stringify({ name: 'saveduser', role: 'admin' }));
    
    const { getByTestId } = render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    );
    
    // Check that state is loaded from localStorage
    await waitFor(() => {
      expect(getByTestId('auth-status')).toHaveTextContent('logged-in');
      expect(getByTestId('user-role')).toHaveTextContent('admin');
      expect(getByTestId('user-name')).toHaveTextContent('saveduser');
    });
  });
}); 