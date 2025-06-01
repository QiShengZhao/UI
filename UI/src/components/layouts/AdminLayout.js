import React, { useState, useEffect } from 'react';
import { Layout, Menu, theme, Typography, Avatar, Dropdown, Button, Badge, Tag } from 'antd';
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom';
import {
  DashboardOutlined,
  ExperimentOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  BellOutlined,
  TeamOutlined,
  ApiOutlined
} from '@ant-design/icons';
import { useAuth } from '../../utils/authContext';
import ApiService from '../../services/api';

const { Header, Content, Sider } = Layout;
const { Title } = Typography;

const AdminLayout = () => {
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
    const intervalId = setInterval(checkStatus, 30000); // More frequent for admin
    
    return () => clearInterval(intervalId);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const menuItems = [
    {
      key: '/admin',
      icon: <DashboardOutlined />,
      label: <Link to="/admin">管理仪表板</Link>,
    },
    {
      key: '/admin/models',
      icon: <ApiOutlined />,
      label: <Link to="/admin/models">模型管理</Link>,
    },
    {
      key: '/admin/users',
      icon: <TeamOutlined />,
      label: <Link to="/admin/users">用户管理</Link>,
    },
    {
      type: 'divider',
    },
    {
      key: '/',
      icon: <ExperimentOutlined />,
      label: <Link to="/">用户界面</Link>,
    },
  ];

  const userMenuItems = [
    {
      key: '1',
      icon: <UserOutlined />,
      label: '管理员设置',
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
              管理员控制台
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
            <Tag color="red" style={{ marginRight: 16 }}>管理员模式</Tag>
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
                <Avatar icon={<UserOutlined />} style={{ marginRight: 8, backgroundColor: '#f56a00' }} />
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

export default AdminLayout; 