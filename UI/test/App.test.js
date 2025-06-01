const React = require('react');
const { render, screen } = require('@testing-library/react');
const { BrowserRouter } = require('react-router-dom');
const App = require('../src/App').default;
const { AuthProvider } = require('../src/utils/authContext');

// Mock required components
jest.mock('../src/pages/Login', () => () => <div data-testid="login-page">Login Page</div>);
jest.mock('../src/components/layouts/UserLayout', () => ({ children }) => <div data-testid="user-layout">User Layout {children}</div>);
jest.mock('../src/components/layouts/AdminLayout', () => ({ children }) => <div data-testid="admin-layout">Admin Layout {children}</div>);
jest.mock('../src/pages/Dashboard', () => () => <div data-testid="dashboard-page">Dashboard Page</div>);
jest.mock('../src/pages/BigScreen', () => () => <div data-testid="big-screen-page">Big Screen Page</div>);

// Mock useAuth hook
jest.mock('../src/utils/authContext', () => {
  const originalModule = jest.requireActual('../src/utils/authContext');
  return {
    ...originalModule,
    useAuth: jest.fn(),
  };
});

describe('App Component', () => {
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
  });

  test('renders login page for unauthenticated users', () => {
    // Setup useAuth mock
    const { useAuth } = require('../src/utils/authContext');
    useAuth.mockReturnValue({
      isAuthenticated: false,
      userRole: null,
    });

    // Render app at login route
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );

    // Expect login page to be rendered
    expect(screen.getByTestId('login-page')).toBeInTheDocument();
  });

  test('redirects to login for protected routes when not authenticated', () => {
    // Setup useAuth mock for unauthenticated user
    const { useAuth } = require('../src/utils/authContext');
    useAuth.mockReturnValue({
      isAuthenticated: false,
      userRole: null,
    });

    // Create a memory router with initial entry at protected route
    const { container } = render(
      <BrowserRouter initialEntries={['/']}>
        <App />
      </BrowserRouter>
    );

    // Should not render dashboard
    expect(screen.queryByTestId('dashboard-page')).not.toBeInTheDocument();
  });

  test('renders big screen without authentication', () => {
    // Setup useAuth mock
    const { useAuth } = require('../src/utils/authContext');
    useAuth.mockReturnValue({
      isAuthenticated: false,
      userRole: null,
    });

    // Render app at big screen route
    const { container } = render(
      <BrowserRouter initialEntries={['/big-screen']}>
        <App />
      </BrowserRouter>
    );

    // Mock navigation to big screen
    window.history.pushState({}, '', '/big-screen');

    // Re-render with new location
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );

    // Should render big screen page
    expect(screen.getByTestId('big-screen-page')).toBeInTheDocument();
  });
}); 