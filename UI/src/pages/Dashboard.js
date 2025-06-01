import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Typography, Table, Spin, Alert, Button } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, ExperimentOutlined, LineChartOutlined, AreaChartOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { useAuth } from '../utils/authContext';
import ApiService from '../services/api';
import { Link } from 'react-router-dom';

const { Title, Text, Paragraph } = Typography;

// Banner样式
const bannerStyles = {
  container: {
    position: 'relative',
    height: '220px',
    borderRadius: '8px',
    overflow: 'hidden',
    marginBottom: '24px',
    backgroundImage: 'url("/assets/images/steel_banner.jpg")',
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    color: '#fff',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(31, 58, 96, 0.8)',
    padding: '24px',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
  },
  title: {
    color: '#fff',
    marginBottom: '16px',
  },
  description: {
    color: 'rgba(255, 255, 255, 0.85)',
    maxWidth: '60%',
    marginBottom: '24px',
  },
  buttonsContainer: {
    display: 'flex',
    gap: '12px',
  },
};

// 样本数据 - 在真实应用中应该从API获取
const sampleData = {
  modelStatus: {
    toughness: { status: true, accuracy: 0.89 },
    yield_strength: { status: true, accuracy: 0.92 },
    tensile_strength: { status: true, accuracy: 0.91 },
    elongation: { status: true, accuracy: 0.87 }
  },
  recentPredictions: [
    { id: 1, date: '2023-06-01', c: 0.12, si: 1.2, mn: 0.8, temp: 950, thickness: 15, coolTemp: 600, yield_strength: 450, tensile_strength: 650, elongation: 22, toughness: 45 },
    { id: 2, date: '2023-06-02', c: 0.15, si: 1.1, mn: 0.9, temp: 930, thickness: 12, coolTemp: 580, yield_strength: 480, tensile_strength: 670, elongation: 20, toughness: 42 },
    { id: 3, date: '2023-06-03', c: 0.11, si: 1.3, mn: 0.7, temp: 940, thickness: 18, coolTemp: 610, yield_strength: 430, tensile_strength: 640, elongation: 24, toughness: 48 },
  ],
  performanceTrend: {
    dates: ['1月', '2月', '3月', '4月', '5月', '6月'],
    yieldStrength: [420, 435, 445, 460, 455, 470],
    tensileStrength: [620, 635, 640, 650, 645, 660],
    elongation: [21, 22, 21.5, 23, 22.5, 24],
    toughness: [40, 42, 44, 45, 44, 46]
  }
};

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [serverStatus, setServerStatus] = useState(null);
  const [dashboardData, setDashboardData] = useState(sampleData);
  const { userRole, userName } = useAuth();
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        // In a real app, we would fetch data from API
        const status = await ApiService.getStatus();
        setServerStatus(status);
        
        // In a real app, you'd fetch real dashboard data here
        // For now, we'll use the sample data with a slight delay to simulate loading
        setTimeout(() => {
          setDashboardData(sampleData);
          setLoading(false);
        }, 1000);
      } catch (err) {
        setError('无法连接到服务器，请稍后再试');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const getPerformanceOption = () => {
    return {
      title: {
        text: '钢材性能趋势',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: ['屈服强度', '抗拉强度', '伸长率', '韧性'],
        bottom: 10
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: dashboardData.performanceTrend.dates
      },
      yAxis: [
        {
          type: 'value',
          name: '强度 (MPa)',
          position: 'left'
        },
        {
          type: 'value',
          name: '伸长率/韧性 (%)',
          position: 'right'
        }
      ],
      series: [
        {
          name: '屈服强度',
          type: 'line',
          yAxisIndex: 0,
          data: dashboardData.performanceTrend.yieldStrength,
          smooth: true
        },
        {
          name: '抗拉强度',
          type: 'line',
          yAxisIndex: 0,
          data: dashboardData.performanceTrend.tensileStrength,
          smooth: true
        },
        {
          name: '伸长率',
          type: 'line',
          yAxisIndex: 1,
          data: dashboardData.performanceTrend.elongation,
          smooth: true
        },
        {
          name: '韧性',
          type: 'line',
          yAxisIndex: 1,
          data: dashboardData.performanceTrend.toughness,
          smooth: true
        }
      ]
    };
  };

  const getModelAccuracyOption = () => {
    const modelNames = Object.keys(dashboardData.modelStatus);
    const accuracies = modelNames.map(key => dashboardData.modelStatus[key].accuracy * 100);
    
    return {
      title: {
        text: '模型精度',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: '{b}: {c}%'
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: ['韧性模型', '屈服强度模型', '抗拉强度模型', '伸长率模型']
      },
      yAxis: {
        type: 'value',
        min: 80,
        max: 100,
        axisLabel: {
          formatter: '{value}%'
        }
      },
      series: [
        {
          type: 'bar',
          data: accuracies,
          itemStyle: {
            color: function(params) {
              // Colorize bars based on accuracy
              const value = params.value;
              if (value >= 90) return '#52c41a';
              if (value >= 85) return '#faad14';
              return '#f5222d';
            }
          },
          label: {
            show: true,
            position: 'top',
            formatter: '{c}%'
          }
        }
      ]
    };
  };

  const columns = [
    { title: '日期', dataIndex: 'date', key: 'date' },
    { title: 'C', dataIndex: 'c', key: 'c' },
    { title: 'Si', dataIndex: 'si', key: 'si' },
    { title: 'Mn', dataIndex: 'mn', key: 'mn' },
    { title: '出炉温度', dataIndex: 'temp', key: 'temp' },
    { title: '厚度', dataIndex: 'thickness', key: 'thickness' },
    { title: '水冷温度', dataIndex: 'coolTemp', key: 'coolTemp' },
    { title: '屈服强度', dataIndex: 'yield_strength', key: 'yield_strength' },
    { title: '抗拉强度', dataIndex: 'tensile_strength', key: 'tensile_strength' },
    { title: '伸长率', dataIndex: 'elongation', key: 'elongation' },
    { title: '韧性', dataIndex: 'toughness', key: 'toughness' }
  ];

  const renderBanner = () => {
    return (
      <div style={bannerStyles.container}>
        <div style={bannerStyles.overlay}>
          <Title level={2} style={bannerStyles.title}>
            欢迎回来，{userName || '用户'}
          </Title>
          <Paragraph style={bannerStyles.description}>
            钢铁工业分析系统为您提供全面的材料性能预测和数据分析服务。选择以下功能快速开始：
          </Paragraph>
          <div style={bannerStyles.buttonsContainer}>
            <Button type="primary" size="large" icon={<ExperimentOutlined />}>
              <Link to="/prediction">开始预测</Link>
            </Button>
            <Button type="default" size="large" ghost icon={<AreaChartOutlined />}>
              <Link to="/analysis">数据分析</Link>
            </Button>
            {userRole === 'admin' && (
              <Button type="default" size="large" ghost icon={<LineChartOutlined />}>
                <Link to="/admin/models">模型管理</Link>
              </Button>
            )}
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <Spin size="large">
          <div style={{padding: '50px', textAlign: 'center'}}>
            加载数据中...
          </div>
        </Spin>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ margin: '24px 0' }}>
        <Alert message="错误" description={error} type="error" showIcon />
      </div>
    );
  }

  return (
    <div>
      {renderBanner()}
      
      <Title level={2}>
        {userRole === 'admin' ? '管理员仪表板' : '性能预测仪表板'}
      </Title>

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="屈服强度预测精度"
              value={dashboardData.modelStatus.yield_strength.accuracy * 100}
              precision={1}
              valueStyle={{ color: '#3f8600' }}
              suffix="%"
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="抗拉强度预测精度"
              value={dashboardData.modelStatus.tensile_strength.accuracy * 100}
              precision={1}
              valueStyle={{ color: '#3f8600' }}
              suffix="%"
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="伸长率预测精度"
              value={dashboardData.modelStatus.elongation.accuracy * 100}
              precision={1}
              valueStyle={{ color: '#cf1322' }}
              suffix="%"
              prefix={<ArrowDownOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="韧性预测精度"
              value={dashboardData.modelStatus.toughness.accuracy * 100}
              precision={1}
              valueStyle={{ color: '#3f8600' }}
              suffix="%"
              prefix={<ArrowUpOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={12}>
          <Card className="dashboard-card">
            <ReactECharts 
              option={getPerformanceOption()} 
              style={{ height: '350px' }} 
              opts={{ renderer: 'svg' }}
            />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card className="dashboard-card">
            <ReactECharts 
              option={getModelAccuracyOption()} 
              style={{ height: '350px' }} 
              opts={{ renderer: 'svg' }}
            />
          </Card>
        </Col>
      </Row>

      <Card style={{ marginTop: 16 }}>
        <Title level={4}>最近预测记录</Title>
        <Table 
          dataSource={dashboardData.recentPredictions} 
          columns={columns} 
          rowKey="id"
          scroll={{ x: 'max-content' }}
        />
      </Card>
    </div>
  );
};

export default Dashboard; 