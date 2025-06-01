import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Typography, Spin, Divider } from 'antd';
import { ExperimentOutlined, RiseOutlined, LineChartOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';

const { Title, Text } = Typography;

// 大屏样式
const screenStyles = {
  container: {
    padding: '20px',
    backgroundColor: '#001529',
    minHeight: '100vh',
    color: '#fff',
  },
  header: {
    textAlign: 'center',
    marginBottom: '20px',
    borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
    paddingBottom: '10px',
    color: '#fff',
  },
  card: {
    backgroundColor: '#0f2749',
    borderRadius: '4px',
    border: 'none',
  },
  cardTitle: {
    color: '#fff',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
    paddingBottom: '10px',
    marginBottom: '15px',
  },
  statistic: {
    color: '#fff',
  }
};

// 随机生成数据
const generateRandomData = () => {
  return {
    productions: {
      total: Math.floor(Math.random() * 5000) + 15000,
      today: Math.floor(Math.random() * 200) + 800,
      qualified: Math.floor(Math.random() * 5) + 95,
    },
    performances: {
      yieldStrength: [
        Math.floor(Math.random() * 20) + 440,
        Math.floor(Math.random() * 20) + 445,
        Math.floor(Math.random() * 20) + 450,
        Math.floor(Math.random() * 20) + 455,
        Math.floor(Math.random() * 20) + 460,
      ],
      tensileStrength: [
        Math.floor(Math.random() * 30) + 640,
        Math.floor(Math.random() * 30) + 645,
        Math.floor(Math.random() * 30) + 650,
        Math.floor(Math.random() * 30) + 655,
        Math.floor(Math.random() * 30) + 660,
      ],
      timeLabels: ['08:00', '10:00', '12:00', '14:00', '16:00'],
    },
    process: {
      temperature: Math.floor(Math.random() * 100) + 900,
      thickness: Math.floor(Math.random() * 5) + 13,
      coolingTemp: Math.floor(Math.random() * 50) + 550,
    }
  };
};

const BigScreen = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);

  useEffect(() => {
    // 模拟加载数据
    setTimeout(() => {
      setData(generateRandomData());
      setLoading(false);
    }, 1000);

    // 设置定期刷新数据
    const timer = setInterval(() => {
      setData(generateRandomData());
    }, 30000);

    return () => clearInterval(timer);
  }, []);

  const getStrengthChartOption = () => {
    if (!data) return {};

    return {
      title: {
        text: '钢材强度实时监测',
        textStyle: {
          color: '#fff',
        }
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      legend: {
        data: ['屈服强度', '抗拉强度'],
        textStyle: {
          color: '#fff'
        },
        top: '10%'
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '25%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: data.performances.timeLabels,
        axisLine: {
          lineStyle: {
            color: '#fff'
          }
        },
        axisLabel: {
          color: '#fff'
        }
      },
      yAxis: {
        type: 'value',
        name: '强度 (MPa)',
        nameTextStyle: {
          color: '#fff'
        },
        axisLine: {
          lineStyle: {
            color: '#fff'
          }
        },
        axisLabel: {
          color: '#fff'
        },
        splitLine: {
          lineStyle: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        }
      },
      series: [
        {
          name: '屈服强度',
          type: 'line',
          data: data.performances.yieldStrength,
          smooth: true,
          lineStyle: {
            width: 3,
          },
          itemStyle: {
            color: '#1890ff'
          }
        },
        {
          name: '抗拉强度',
          type: 'line',
          data: data.performances.tensileStrength,
          smooth: true,
          lineStyle: {
            width: 3,
          },
          itemStyle: {
            color: '#52c41a'
          }
        }
      ]
    };
  };

  const getQualifiedChartOption = () => {
    if (!data) return {};

    return {
      title: {
        text: '产品合格率',
        textStyle: {
          color: '#fff',
        }
      },
      tooltip: {
        formatter: '{a} <br/>{b} : {c}%'
      },
      series: [
        {
          name: '合格率',
          type: 'gauge',
          radius: '85%',
          detail: { 
            formatter: '{value}%',
            fontSize: 24,
            color: '#fff'
          },
          data: [{ value: data.productions.qualified, name: '合格率' }],
          title: {
            color: '#fff'
          },
          axisLine: {
            lineStyle: {
              width: 20,
              color: [
                [0.7, '#f44336'],
                [0.9, '#faad14'],
                [1, '#52c41a']
              ]
            }
          }
        }
      ]
    };
  };

  const getProcessChartOption = () => {
    if (!data) return {};

    // 创建循环的光点图表
    const hours = [
      '12a', '1a', '2a', '3a', '4a', '5a', '6a',
      '7a', '8a', '9a', '10a', '11a',
      '12p', '1p', '2p', '3p', '4p', '5p',
      '6p', '7p', '8p', '9p', '10p', '11p'
    ];
    
    const days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
    
    const processData = [];
    for (let i = 0; i < 7; i++) {
      for (let j = 0; j < 24; j++) {
        processData.push([j, i, Math.random() * 70 + 880]);
      }
    }

    return {
      title: {
        text: '本周工艺温度热图',
        textStyle: {
          color: '#fff'
        }
      },
      tooltip: {
        position: 'top',
        formatter: function (params) {
          return `${days[params.value[1]]}, ${hours[params.value[0]]}: ${params.value[2].toFixed(0)}°C`;
        }
      },
      grid: {
        left: '2%',
        right: '8%',
        top: '17%',
        bottom: '5%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: hours,
        splitArea: {
          show: true
        },
        axisLine: {
          lineStyle: {
            color: '#fff'
          }
        },
        axisLabel: {
          color: '#fff',
          rotate: 45,
          interval: 3
        }
      },
      yAxis: {
        type: 'category',
        data: days,
        splitArea: {
          show: true
        },
        axisLine: {
          lineStyle: {
            color: '#fff'
          }
        },
        axisLabel: {
          color: '#fff'
        }
      },
      visualMap: {
        min: 880,
        max: 950,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0',
        inRange: {
          color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        },
        textStyle: {
          color: '#fff'
        }
      },
      series: [{
        name: '温度热图',
        type: 'heatmap',
        data: processData,
        label: {
          show: false
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }]
    };
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', backgroundColor: '#001529' }}>
        <Spin size="large" tip="加载大屏数据..." />
      </div>
    );
  }

  return (
    <div style={screenStyles.container}>
      <div style={screenStyles.header}>
        <Title level={1} style={{ color: '#fff', margin: '20px 0' }}>钢铁工业生产监控中心</Title>
        <Text style={{ color: 'rgba(255, 255, 255, 0.85)' }}>
          实时监控和分析 · 更新时间：{new Date().toLocaleString()}
        </Text>
      </div>

      <Row gutter={[16, 16]}>
        <Col xs={24} md={8}>
          <Card style={screenStyles.card} variant="outlined">
            <Title level={4} style={screenStyles.cardTitle}>产量总览</Title>
            <Row gutter={16}>
              <Col span={8}>
                <Statistic 
                  title={<span style={{ color: '#aaa' }}>总产量(吨)</span>}
                  value={data.productions.total}
                  valueStyle={{ color: '#3f8600', fontSize: '24px' }}
                  prefix={<RiseOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title={<span style={{ color: '#aaa' }}>今日产量</span>}
                  value={data.productions.today} 
                  valueStyle={{ color: '#1890ff', fontSize: '24px' }}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title={<span style={{ color: '#aaa' }}>合格率</span>}
                  value={data.productions.qualified} 
                  suffix="%" 
                  valueStyle={{ color: '#52c41a', fontSize: '24px' }}
                />
              </Col>
            </Row>
          </Card>

          <Card style={{...screenStyles.card, marginTop: '16px'}} variant="outlined">
            <Title level={4} style={screenStyles.cardTitle}>工艺参数</Title>
            <Row gutter={16}>
              <Col span={8}>
                <Statistic 
                  title={<span style={{ color: '#aaa' }}>出炉温度</span>}
                  value={data.process.temperature} 
                  suffix="°C"
                  valueStyle={{ color: '#ff7a45', fontSize: '24px' }}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title={<span style={{ color: '#aaa' }}>厚度</span>}
                  value={data.process.thickness} 
                  suffix="mm"
                  valueStyle={{ color: '#1890ff', fontSize: '24px' }}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title={<span style={{ color: '#aaa' }}>水冷温度</span>}
                  value={data.process.coolingTemp} 
                  suffix="°C"
                  valueStyle={{ color: '#36cfc9', fontSize: '24px' }}
                />
              </Col>
            </Row>
          </Card>

          <Card style={{...screenStyles.card, marginTop: '16px', height: '350px'}} variant="outlined">
            <ReactECharts 
              option={getQualifiedChartOption()} 
              style={{ height: '300px' }} 
            />
          </Card>
        </Col>

        <Col xs={24} md={16}>
          <Card style={{...screenStyles.card, height: '400px'}} variant="outlined">
            <ReactECharts 
              option={getStrengthChartOption()} 
              style={{ height: '360px' }} 
            />
          </Card>

          <Card style={{...screenStyles.card, marginTop: '16px', height: '400px'}} variant="outlined">
            <ReactECharts 
              option={getProcessChartOption()} 
              style={{ height: '360px' }} 
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default BigScreen; 