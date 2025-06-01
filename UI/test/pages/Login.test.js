const React = require('react');
const { render, screen, fireEvent, waitFor } = require('@testing-library/react');
const { BrowserRouter } = require('react-router-dom');
const Login = require('../../src/pages/Login').default;
const { AuthProvider } = require('../../src/utils/authContext');

// Mock navigate function
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => jest.fn(),
  useLocation: () => ({ state: { from: { pathname: '/' } } }),
}));

// Mock message
jest.mock('antd', () => {
  const antd = jest.requireActual('antd');
  return {
    ...antd,
    message: {
      success: jest.fn(),
      error: jest.fn(),
    },
  };
});

const renderLoginWithRouter = () => {
  return render(
    <BrowserRouter>
      <AuthProvider>
        <Login />
      </AuthProvider>
    </BrowserRouter>
  );
};

describe('Login Component', () => {
  test('renders login form with title', () => {
    renderLoginWithRouter();
    
    expect(screen.getByText('钢铁工业分析系统')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('用户名')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('密码')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '登 录' })).toBeInTheDocument();
  });

  test('shows error when submitting empty form', async () => {
    renderLoginWithRouter();
    
    const loginButton = screen.getByRole('button', { name: '登 录' });
    fireEvent.click(loginButton);
    
    await waitFor(() => {
      expect(screen.getByText('请输入用户名')).toBeInTheDocument();
      expect(screen.getByText('请输入密码')).toBeInTheDocument();
    });
  });

  test('submits form with valid credentials', async () => {
    renderLoginWithRouter();
    
    const usernameInput = screen.getByPlaceholderText('用户名');
    const passwordInput = screen.getByPlaceholderText('密码');
    const loginButton = screen.getByRole('button', { name: '登 录' });
    
    fireEvent.change(usernameInput, { target: { value: 'user' } });
    fireEvent.change(passwordInput, { target: { value: 'password' } });
    fireEvent.click(loginButton);
    
    // Wait for the login process to complete
    await waitFor(() => {
      // Local storage should be updated
      const storedUser = JSON.parse(localStorage.getItem('user'));
      expect(storedUser).toEqual({ name: 'user', role: 'user' });
    });
  });
}); 