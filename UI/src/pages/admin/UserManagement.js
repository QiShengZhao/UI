import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Typography,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  message,
  Popconfirm,
  Divider,
  Avatar,
  Row,
  Col,
  Statistic,
  Badge,
  Drawer,
  Descriptions
} from 'antd';
import {
  UserOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SearchOutlined,
  LockOutlined,
  UserSwitchOutlined,
  SolutionOutlined,
  TeamOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;

// 模拟用户数据 - 在实际应用中应该从API获取
const mockUsers = [
  { id: 1, username: 'admin', name: '管理员', role: 'admin', department: '技术部', status: 'active', lastLogin: '2023-06-01 08:30:00' },
  { id: 2, username: 'user1', name: '张工程师', role: 'user', department: '生产部', status: 'active', lastLogin: '2023-06-01 09:15:00' },
  { id: 3, username: 'user2', name: '李质检', role: 'user', department: '质检部', status: 'active', lastLogin: '2023-06-01 08:45:00' },
  { id: 4, username: 'user3', name: '王分析师', role: 'user', department: '研发部', status: 'inactive', lastLogin: '2023-05-28 14:20:00' },
  { id: 5, username: 'user4', name: '赵工程师', role: 'user', department: '生产部', status: 'active', lastLogin: '2023-06-01 10:05:00' },
];

const UserManagement = () => {
  const [loading, setLoading] = useState(true);
  const [users, setUsers] = useState([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [modalTitle, setModalTitle] = useState('');
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [form] = Form.useForm();
  const [editingId, setEditingId] = useState(null);
  const [searchText, setSearchText] = useState('');
  const [confirmLoading, setConfirmLoading] = useState(false);

  // 获取用户数据
  useEffect(() => {
    // 模拟API请求
    setTimeout(() => {
      setUsers(mockUsers);
      setLoading(false);
    }, 1000);
  }, []);

  // 打开创建用户模态框
  const showCreateModal = () => {
    setModalTitle('创建新用户');
    setEditingId(null);
    form.resetFields();
    setModalVisible(true);
  };

  // 打开编辑用户模态框
  const showEditModal = (user) => {
    setModalTitle('编辑用户');
    setEditingId(user.id);
    form.setFieldsValue({
      username: user.username,
      name: user.name,
      role: user.role,
      department: user.department,
      status: user.status,
      password: '',
      confirmPassword: '',
    });
    setModalVisible(true);
  };

  // 处理模态框提交
  const handleModalSubmit = () => {
    form.validateFields().then(values => {
      // 在实际应用中，这里应该调用API
      if (editingId) {
        // 更新现有用户
        const updatedUsers = users.map(user => 
          user.id === editingId ? { ...user, ...values } : user
        );
        setUsers(updatedUsers);
        message.success('用户已更新');
      } else {
        // 创建新用户
        const newUser = {
          id: Math.max(...users.map(u => u.id)) + 1,
          ...values,
          lastLogin: '-'
        };
        setUsers([...users, newUser]);
        message.success('用户已创建');
      }
      setModalVisible(false);
    });
  };

  // 删除用户
  const handleDelete = (userId) => {
    // 在实际应用中，这里应该调用API
    setUsers(users.filter(user => user.id !== userId));
    message.success('用户已删除');
  };

  // 显示用户详情抽屉
  const showUserDetail = (user) => {
    setCurrentUser(user);
    setDrawerVisible(true);
  };

  // 过滤用户
  const filteredUsers = users.filter(user => 
    user.username.toLowerCase().includes(searchText.toLowerCase()) ||
    user.name.toLowerCase().includes(searchText.toLowerCase()) ||
    user.department.toLowerCase().includes(searchText.toLowerCase())
  );

  // 表格列定义
  const columns = [
    {
      title: '用户名',
      dataIndex: 'username',
      key: 'username',
      render: (text, record) => (
        <Space>
          <Avatar 
            style={{ backgroundColor: record.role === 'admin' ? '#f56a00' : '#1890ff' }}
            icon={<UserOutlined />} 
          />
          <a onClick={() => showUserDetail(record)}>{text}</a>
        </Space>
      ),
    },
    {
      title: '姓名',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '角色',
      dataIndex: 'role',
      key: 'role',
      render: role => (
        <Tag color={role === 'admin' ? 'red' : 'blue'}>
          {role === 'admin' ? '管理员' : '普通用户'}
        </Tag>
      ),
    },
    {
      title: '部门',
      dataIndex: 'department',
      key: 'department',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => (
        <Badge 
          status={status === 'active' ? 'success' : 'default'} 
          text={status === 'active' ? '启用' : '禁用'} 
        />
      ),
    },
    {
      title: '上次登录',
      dataIndex: 'lastLogin',
      key: 'lastLogin',
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="middle">
          <Button 
            icon={<EditOutlined />} 
            size="small" 
            onClick={() => showEditModal(record)}
          >
            编辑
          </Button>
          <Popconfirm
            title="确定要删除此用户吗？"
            onConfirm={() => handleDelete(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Button 
              icon={<DeleteOutlined />} 
              size="small" 
              danger
              disabled={record.username === 'admin'} // 禁止删除管理员
            >
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Title level={2}>用户管理</Title>
      <Text>管理系统用户账号、权限和状态</Text>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} sm={8} md={6} lg={6}>
          <Card>
            <Statistic 
              title="总用户数" 
              value={users.length} 
              prefix={<TeamOutlined />} 
            />
          </Card>
        </Col>
        <Col xs={24} sm={8} md={6} lg={6}>
          <Card>
            <Statistic 
              title="活跃用户" 
              value={users.filter(u => u.status === 'active').length} 
              prefix={<UserOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8} md={6} lg={6}>
          <Card>
            <Statistic 
              title="管理员数量" 
              value={users.filter(u => u.role === 'admin').length} 
              prefix={<UserSwitchOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={24} md={6} lg={6}>
          <Card bodyStyle={{ padding: '20px 24px' }}>
            <Button 
              type="primary" 
              icon={<PlusOutlined />} 
              onClick={showCreateModal}
              block
            >
              创建新用户
            </Button>
          </Card>
        </Col>
      </Row>

      <Card style={{ marginTop: 16 }}>
        <Space style={{ marginBottom: 16 }}>
          <Input
            placeholder="搜索用户"
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
            prefix={<SearchOutlined />}
            style={{ width: 200 }}
            allowClear
          />
        </Space>

        <Table
          loading={loading}
          columns={columns}
          dataSource={filteredUsers}
          rowKey="id"
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* 创建/编辑用户模态框 */}
      <Modal
        title={modalTitle}
        open={modalVisible}
        onOk={handleModalSubmit}
        onCancel={() => setModalVisible(false)}
        confirmLoading={confirmLoading}
      >
        <Form
          form={form}
          layout="vertical"
        >
          <Form.Item
            name="username"
            label="用户名"
            rules={[{ required: true, message: '请输入用户名' }]}
          >
            <Input prefix={<UserOutlined />} disabled={!!editingId} />
          </Form.Item>

          <Form.Item
            name="name"
            label="姓名"
            rules={[{ required: true, message: '请输入姓名' }]}
          >
            <Input />
          </Form.Item>

          <Form.Item
            name="role"
            label="角色"
            rules={[{ required: true, message: '请选择角色' }]}
          >
            <Select>
              <Option value="admin">管理员</Option>
              <Option value="user">普通用户</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="department"
            label="部门"
            rules={[{ required: true, message: '请选择部门' }]}
          >
            <Select>
              <Option value="技术部">技术部</Option>
              <Option value="生产部">生产部</Option>
              <Option value="质检部">质检部</Option>
              <Option value="研发部">研发部</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="status"
            label="状态"
            rules={[{ required: true, message: '请选择状态' }]}
          >
            <Select>
              <Option value="active">启用</Option>
              <Option value="inactive">禁用</Option>
            </Select>
          </Form.Item>

          <Divider dashed />
          
          <Form.Item
            name="password"
            label="密码"
            rules={[
              { required: !editingId, message: '请输入密码' },
              { min: 6, message: '密码长度不能少于6个字符' }
            ]}
          >
            <Input.Password prefix={<LockOutlined />} />
          </Form.Item>

          <Form.Item
            name="confirmPassword"
            label="确认密码"
            dependencies={['password']}
            rules={[
              { required: !editingId, message: '请确认密码' },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue('password') === value) {
                    return Promise.resolve();
                  }
                  return Promise.reject(new Error('两次输入的密码不匹配'));
                },
              }),
            ]}
          >
            <Input.Password prefix={<LockOutlined />} />
          </Form.Item>
        </Form>
      </Modal>

      {/* 用户详情抽屉 */}
      <Drawer
        title="用户详情"
        width={500}
        open={drawerVisible}
        onClose={() => setDrawerVisible(false)}
        maskClosable={true}
      >
        {currentUser && (
          <div>
            <div style={{ textAlign: 'center', marginBottom: 24 }}>
              <Avatar 
                size={80} 
                icon={<UserOutlined />} 
                style={{ backgroundColor: currentUser.role === 'admin' ? '#f56a00' : '#1890ff' }}
              />
              <Title level={4} style={{ marginTop: 12, marginBottom: 4 }}>
                {currentUser.name}
              </Title>
              <Text type="secondary">{currentUser.username}</Text>
              <br />
              <Tag color={currentUser.role === 'admin' ? 'red' : 'blue'} style={{ marginTop: 8 }}>
                {currentUser.role === 'admin' ? '管理员' : '普通用户'}
              </Tag>
            </div>

            <Divider />

            <Descriptions title="基本信息" bordered column={1}>
              <Descriptions.Item label="部门">{currentUser.department}</Descriptions.Item>
              <Descriptions.Item label="状态">
                <Badge 
                  status={currentUser.status === 'active' ? 'success' : 'default'} 
                  text={currentUser.status === 'active' ? '启用' : '禁用'} 
                />
              </Descriptions.Item>
              <Descriptions.Item label="上次登录">{currentUser.lastLogin}</Descriptions.Item>
              <Descriptions.Item label="创建时间">2023-01-01</Descriptions.Item>
            </Descriptions>

            <Divider />

            <Space>
              <Button 
                type="primary" 
                icon={<EditOutlined />} 
                onClick={() => {
                  setDrawerVisible(false);
                  showEditModal(currentUser);
                }}
              >
                编辑用户
              </Button>
              <Button 
                icon={<LockOutlined />}
                onClick={() => message.info('重置密码功能待实现')}
              >
                重置密码
              </Button>
            </Space>
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default UserManagement; 