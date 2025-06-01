import React, { useState } from 'react';
import { Form, Input, Button, Typography, Card, message, Spin } from 'antd';
import { UserOutlined, LockOutlined, SafetyOutlined } from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../utils/authContext';

const { Title, Text } = Typography;

// 自定义样式
const styles = {
  loginContainer: {
    position: 'relative',
    height: '100vh',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundImage: 'url("/assets/images/steel_factory_bg.jpg")',
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    backgroundRepeat: 'no-repeat',
    overflow: 'hidden',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    zIndex: 1,
  },
  loginForm: {
    width: 360,
    padding: 24,
    background: 'white',
    borderRadius: 8,
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
    zIndex: 2,
  },
  logoContainer: {
    textAlign: 'center', 
    marginBottom: 24,
  },
  logo: {
    height: 80, 
    width: 80,
  },
  title: {
    marginTop: 16,
    color: '#1f3a60',
  },
  tip: {
    textAlign: 'center',
    marginTop: 24,
  }
};

const Login = () => {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();

  const from = location.state?.from?.pathname || '/';

  const handleSubmit = async (values) => {
    setLoading(true);
    try {
      const user = await login(values);
      message.success(`欢迎回来，${user.name}！`);
      navigate(user.role === 'admin' ? '/admin' : '/', { replace: true });
    } catch (error) {
      message.error('登录失败，请检查用户名和密码');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.loginContainer}>
      <div style={styles.overlay} />
      <Card style={styles.loginForm} variant="outlined">
        <div style={styles.logoContainer}>
          <img 
            src="/assets/logo.svg" 
            alt="钢铁工业分析系统" 
            style={styles.logo} 
          />
          <Title level={2} style={styles.title}>钢铁工业分析系统</Title>
        </div>

        <Spin spinning={loading}>
          <Form
            name="login"
            initialValues={{ remember: true }}
            onFinish={handleSubmit}
            size="large"
          >
            <Form.Item
              name="username"
              rules={[{ required: true, message: '请输入用户名' }]}
            >
              <Input 
                prefix={<UserOutlined />} 
                placeholder="用户名" 
              />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[{ required: true, message: '请输入密码' }]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="密码"
              />
            </Form.Item>

            <Form.Item>
              <Button type="primary" htmlType="submit" block>
                登录
              </Button>
            </Form.Item>
          </Form>
        </Spin>

        <div style={styles.tip}>
          <Text type="secondary">用户提示：普通用户使用 user/password 登录</Text>
          <br />
          <Text type="secondary">管理员使用 admin/password 登录</Text>
        </div>
      </Card>
    </div>
  );
};

export default Login; 