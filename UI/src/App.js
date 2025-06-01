import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/lib/locale/zh_CN';

// Pages
import Login from './pages/Login';
import UserLayout from './components/layouts/UserLayout';
import AdminLayout from './components/layouts/AdminLayout';
import Dashboard from './pages/Dashboard';
import PredictionForm from './pages/PredictionForm';
import DataAnalysis from './pages/DataAnalysis';
import ModelManagement from './pages/admin/ModelManagement';
import UserManagement from './pages/admin/UserManagement';
import BigScreen from './pages/BigScreen';

// Auth context for role-based access
import { AuthProvider, useAuth } from './utils/authContext';

// Protect routes based on authentication and role
const ProtectedRoute = ({ children, requiredRole }) => {
  const { isAuthenticated, userRole } = useAuth();
  const location = useLocation();

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (requiredRole && userRole !== requiredRole) {
    return <Navigate to="/" />;
  }

  return children;
};

const App = () => {
  return (
    <ConfigProvider locale={zhCN}>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          
          {/* 大屏展示路由 - 不需要验证登录状态 */}
          <Route path="/big-screen" element={<BigScreen />} />
          
          {/* User Routes */}
          <Route path="/" element={
            <ProtectedRoute>
              <UserLayout />
            </ProtectedRoute>
          }>
            <Route index element={<Dashboard />} />
            <Route path="prediction" element={<PredictionForm />} />
            <Route path="analysis" element={<DataAnalysis />} />
          </Route>
          
          {/* Admin Routes */}
          <Route path="/admin" element={
            <ProtectedRoute requiredRole="admin">
              <AdminLayout />
            </ProtectedRoute>
          }>
            <Route index element={<Dashboard />} />
            <Route path="models" element={<ModelManagement />} />
            <Route path="users" element={<UserManagement />} />
          </Route>
          
          {/* Fallback route */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </ConfigProvider>
  );
};

export default App; 