import React, { useState, useEffect } from 'react';
import { Layout, Menu, theme, Typography, Avatar, Dropdown, Button, Badge } from 'antd';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import {
  DashboardOutlined,
  ExperimentOutlined,
  AreaChartOutlined,
  UserOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  BellOutlined,
  FullscreenOutlined
} from '@ant-design/icons';
import { useAuth } from '../../utils/authContext';
import ApiService from '../../services/api';

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

const UserLayout = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [serverStatus, setServerStatus] = useState({ running: false, models_loaded: {} });
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const { userName, logout } = useAuth();
  const { token } = theme.useToken();

  // Check server status
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const status = await ApiService.getStatus();
        setServerStatus(status);
      } catch (error) {
        console.error('Unable to connect to server');
      }
    };

    checkStatus();
    // Set up periodic status check
    const intervalId = setInterval(checkStatus, 60000);
    
    return () => clearInterval(intervalId);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: <Link to="/">数据仪表板</Link>,
    },
    {
      key: '/prediction',
      icon: <ExperimentOutlined />,
      label: <Link to="/prediction">性能预测</Link>,
    },
    {
      key: '/analysis',
      icon: <AreaChartOutlined />,
      label: <Link to="/analysis">数据分析</Link>,
    },
    {
      key: '/big-screen',
      icon: <FullscreenOutlined />,
      label: <a href="/big-screen" target="_blank" rel="noopener noreferrer">大屏展示</a>,
    },
  ];

  const userMenuItems = [
    {
      key: '1',
      icon: <UserOutlined />,
      label: '个人设置',
    },
    {
      key: '2',
      icon: <LogoutOutlined />,
      label: '退出登录',
      onClick: handleLogout,
    },
  ];

  // Calculate the number of offline models
  const offlineModelsCount = Object.values(serverStatus.models_loaded).filter(status => !status).length;

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider 
        collapsible 
        collapsed={collapsed} 
        onCollapse={setCollapsed}
        style={{
          overflow: 'auto',
          height: '100vh',
          position: 'sticky',
          top: 0,
          left: 0,
        }}
        theme="dark"
      >
        <div style={{ 
          height: 64, 
          margin: 16, 
          display: 'flex', 
          alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'flex-start'
        }}>
          <img 
            src="/assets/logo.svg" 
            alt="钢铁工业分析系统" 
            style={{ height: 40, width: 40 }} 
          />
          {!collapsed && (
            <Title level={4} style={{ color: token.colorBgContainer, margin: '0 0 0 12px' }}>
              钢铁工业分析
            </Title>
          )}
        </div>
        <Menu 
          theme="dark" 
          mode="inline" 
          selectedKeys={[pathname]} 
          items={menuItems} 
        />
      </Sider>
      <Layout>
        <Header 
          style={{ 
            padding: 0, 
            background: token.colorBgContainer,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            boxShadow: '0 1px 2px rgba(0, 0, 0, 0.03)',
            position: 'sticky',
            top: 0,
            zIndex: 1,
            width: '100%',
          }}
        >
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{ fontSize: '16px', width: 64, height: 64 }}
          />
          <div style={{ display: 'flex', alignItems: 'center', marginRight: 24 }}>
            <Badge count={offlineModelsCount} size="small">
              <Button 
                icon={<BellOutlined />} 
                type="text" 
                style={{ marginRight: 12 }}
                title={serverStatus.running ? '服务器在线' : '服务器离线'}
              />
            </Badge>
            <Dropdown menu={{ items: userMenuItems }} trigger={['click']}>
              <div style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                <Avatar icon={<UserOutlined />} style={{ marginRight: 8 }} />
                <span>{userName}</span>
              </div>
            </Dropdown>
          </div>
        </Header>
        <Content className="content-container">
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
};

export default UserLayout; 