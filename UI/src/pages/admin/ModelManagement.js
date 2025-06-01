import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Typography,
  Space,
  Tag,
  Modal,
  Spin,
  message,
  Tabs,
  Descriptions,
  Collapse,
  Tooltip,
  Progress,
  Empty
} from 'antd';
import {
  ReloadOutlined,
  ExperimentOutlined,
  SearchOutlined,
  InfoCircleOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ToolOutlined
} from '@ant-design/icons';
import ApiService from '../../services/api';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Panel } = Collapse;

const ModelManagement = () => {
  const [loading, setLoading] = useState(true);
  const [modelsData, setModelsData] = useState({});
  const [modelFeatures, setModelFeatures] = useState({});
  const [activeTab, setActiveTab] = useState('overview');
  const [reloadingModel, setReloadingModel] = useState(null);
  const [analyzingModels, setAnalyzingModels] = useState(false);
  const [trainingModel, setTrainingModel] = useState(false);
  const [showTrainModal, setShowTrainModal] = useState(false);

  // 获取模型状态
  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      setLoading(true);
      const status = await ApiService.getStatus();
      setModelsData(status.models_loaded || {});
      setLoading(false);
    } catch (error) {
      message.error('无法获取模型状态');
      setLoading(false);
    }
  };

  // 分析模型
  const handleAnalyzeModels = async () => {
    try {
      setAnalyzingModels(true);
      const result = await ApiService.analyzeModels();
      if (result.success) {
        setModelFeatures(result.results);
        message.success('模型分析完成');
      } else {
        message.error('模型分析失败');
      }
    } catch (error) {
      message.error('模型分析请求失败');
    } finally {
      setAnalyzingModels(false);
    }
  };

  // 重新加载模型
  const handleReloadModel = async (modelType) => {
    try {
      setReloadingModel(modelType);
      const result = await ApiService.reloadModels();
      if (result.success) {
        message.success(`${getModelNameChinese(modelType)}模型重新加载成功`);
        fetchModelStatus();
      } else {
        message.error(`${getModelNameChinese(modelType)}模型重新加载失败`);
      }
    } catch (error) {
      message.error('模型重新加载请求失败');
    } finally {
      setReloadingModel(null);
    }
  };

  // 训练新的伸长率模型
  const handleTrainElongationModel = async () => {
    try {
      setTrainingModel(true);
      const result = await ApiService.trainElongationModel();
      if (result.success) {
        message.success('伸长率模型训练成功');
        fetchModelStatus();
        setShowTrainModal(false);
      } else {
        message.error('伸长率模型训练失败: ' + result.error);
      }
    } catch (error) {
      message.error('模型训练请求失败');
    } finally {
      setTrainingModel(false);
    }
  };

  // 获取模型中文名称
  const getModelNameChinese = (modelType) => {
    const modelNames = {
      'toughness': '韧性',
      'yield_strength': '屈服强度',
      'tensile_strength': '抗拉强度',
      'elongation': '伸长率'
    };
    return modelNames[modelType] || modelType;
  };

  // 模型状态表格列定义
  const columns = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space>
          <ExperimentOutlined />
          <span>{getModelNameChinese(record.key)}</span>
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status ? 'success' : 'error'}>
          {status ? '已加载' : '未加载'}
        </Tag>
      ),
    },
    {
      title: '特征数量',
      dataIndex: 'featureCount',
      key: 'featureCount',
      render: (text, record) => {
        const features = modelFeatures[record.key];
        return features ? features.expected_feature_count || '未知' : '未分析';
      }
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="middle">
          <Button 
            type="primary" 
            size="small"
            icon={<ReloadOutlined />} 
            loading={reloadingModel === record.key}
            onClick={() => handleReloadModel(record.key)}
          >
            重新加载
          </Button>
          <Button 
            type="default" 
            size="small"
            icon={<InfoCircleOutlined />} 
            onClick={() => setActiveTab('details')}
          >
            详情
          </Button>
        </Space>
      ),
    },
  ];

  // 准备表格数据
  const getTableData = () => {
    return Object.keys(modelsData).map(key => ({
      key,
      name: getModelNameChinese(key),
      status: modelsData[key],
      featureCount: modelFeatures[key]?.expected_feature_count || '未知'
    }));
  };

  return (
    <div>
      <Title level={2}>模型管理</Title>
      <Text>管理和监控预测模型的状态和性能</Text>

      <Tabs activeKey={activeTab} onChange={setActiveTab} style={{ marginTop: 16 }}>
        <TabPane tab="模型概览" key="overview">
          <Card>
            <Space style={{ marginBottom: 16 }}>
              <Button 
                type="primary"
                icon={<ReloadOutlined />}
                onClick={() => fetchModelStatus()}
                loading={loading}
              >
                刷新状态
              </Button>
              <Button 
                type="default"
                icon={<SearchOutlined />}
                onClick={handleAnalyzeModels}
                loading={analyzingModels}
              >
                分析模型特征
              </Button>
              <Button 
                type="default"
                icon={<SettingOutlined />}
                onClick={() => setShowTrainModal(true)}
              >
                训练新模型
              </Button>
            </Space>

            <Spin spinning={loading}>
              <Table 
                columns={columns} 
                dataSource={getTableData()} 
                rowKey="key"
                pagination={false}
              />
            </Spin>
          </Card>
        </TabPane>

        <TabPane tab="模型详情" key="details">
          <Card>
            <Collapse defaultActiveKey={['1']} expandIconPosition="end" items={[
              {
                key: '1',
                label: '韧性模型',
                children: (
                  <div className="model-details">
                <Descriptions bordered column={2}>
                  <Descriptions.Item label="模型类型">随机森林回归</Descriptions.Item>
                  <Descriptions.Item label="模型精度">89.2%</Descriptions.Item>
                  <Descriptions.Item label="特征数量">
                    {modelFeatures.toughness?.expected_feature_count || '未知'}
                  </Descriptions.Item>
                  <Descriptions.Item label="训练数据量">892条</Descriptions.Item>
                  <Descriptions.Item label="模型路径">韧性/best_rf_model_chinese.pkl</Descriptions.Item>
                  <Descriptions.Item label="缩放器路径">韧性/rf_feature_scaler_chinese.pkl</Descriptions.Item>
                </Descriptions>
                
                <Title level={5} style={{ marginTop: 16 }}>主要特征</Title>
                <Paragraph>
                  碳含量(C)、硅含量(Si)、锰含量(Mn)、出炉温度、终扎厚度、水冷温度
                </Paragraph>
                  </div>
                )
              },
              {
                key: '2',
                label: '屈服强度模型',
                children: (
                  <div className="model-details">
                <Descriptions bordered column={2}>
                  <Descriptions.Item label="模型类型">随机森林回归</Descriptions.Item>
                  <Descriptions.Item label="模型精度">92.5%</Descriptions.Item>
                  <Descriptions.Item label="特征数量">
                    {modelFeatures.yield_strength?.expected_feature_count || '未知'}
                  </Descriptions.Item>
                  <Descriptions.Item label="训练数据量">1024条</Descriptions.Item>
                  <Descriptions.Item label="模型路径">屈服强度/rf_model.pkl</Descriptions.Item>
                  <Descriptions.Item label="缩放器路径">屈服强度/scaler.pkl</Descriptions.Item>
                </Descriptions>
                
                <Title level={5} style={{ marginTop: 16 }}>主要特征</Title>
                <Paragraph>
                  碳含量(C)、硅含量(Si)、锰含量(Mn)、出炉温度、终扎厚度、水冷温度
                </Paragraph>
                  </div>
                )
              },
              {
                key: '3',
                label: '抗拉强度模型',
                children: (
                  <div className="model-details">
                <Descriptions bordered column={2}>
                  <Descriptions.Item label="模型类型">XGBoost回归</Descriptions.Item>
                  <Descriptions.Item label="模型精度">91.8%</Descriptions.Item>
                  <Descriptions.Item label="特征数量">
                    {modelFeatures.tensile_strength?.expected_feature_count || '未知'}
                  </Descriptions.Item>
                  <Descriptions.Item label="训练数据量">1024条</Descriptions.Item>
                  <Descriptions.Item label="模型路径">抗拉强度/xgb_model.pkl</Descriptions.Item>
                  <Descriptions.Item label="缩放器路径">抗拉强度/scaler.pkl</Descriptions.Item>
                </Descriptions>
                
                <Title level={5} style={{ marginTop: 16 }}>主要特征</Title>
                <Paragraph>
                  碳含量(C)、硅含量(Si)、锰含量(Mn)、出炉温度、终扎厚度、水冷温度
                </Paragraph>
                  </div>
                )
              },
              {
                key: '4',
                label: '伸长率模型',
                children: (
                  <div className="model-details">
                <Descriptions bordered column={2}>
                  <Descriptions.Item label="模型类型">随机森林回归</Descriptions.Item>
                  <Descriptions.Item label="模型精度">87.6%</Descriptions.Item>
                  <Descriptions.Item label="特征数量">
                    {modelFeatures.elongation?.expected_feature_count || '未知'}
                  </Descriptions.Item>
                  <Descriptions.Item label="训练数据量">768条</Descriptions.Item>
                  <Descriptions.Item label="模型路径">伸长率/模型/rf_elongation_model.pkl</Descriptions.Item>
                  <Descriptions.Item label="缩放器路径">伸长率/模型/rf_elongation_scaler.pkl</Descriptions.Item>
                </Descriptions>
                
                <Title level={5} style={{ marginTop: 16 }}>主要特征</Title>
                <Paragraph>
                  碳含量(C)、硅含量(Si)、锰含量(Mn)、出炉温度、终扎厚度、水冷温度、屈服强度、抗拉强度
                </Paragraph>
                
                <Button 
                  type="primary" 
                  icon={<ToolOutlined />}
                  onClick={() => setShowTrainModal(true)}
                  style={{ marginTop: 16 }}
                >
                  训练新模型
                </Button>
                  </div>
                )
              }
            ]} />
          </Card>
        </TabPane>
      </Tabs>

      {/* 训练模型对话框 */}
      <Modal
        title="训练新模型"
        open={showTrainModal}
        onOk={handleTrainElongationModel}
        onCancel={() => setShowTrainModal(false)}
        confirmLoading={trainingModel}
      >
        <Spin spinning={trainingModel}>
          <Paragraph>
            系统将使用最新的数据集训练新的伸长率预测模型。该过程可能需要几分钟时间完成。
          </Paragraph>
          
          <Title level={5}>训练配置</Title>
          <Descriptions bordered column={1} size="small">
            <Descriptions.Item label="算法">随机森林回归</Descriptions.Item>
            <Descriptions.Item label="训练集/测试集比例">80% / 20%</Descriptions.Item>
            <Descriptions.Item label="特征标准化">是</Descriptions.Item>
            <Descriptions.Item label="随机森林树数量">100</Descriptions.Item>
            <Descriptions.Item label="数据源">伸长率/combined_data.csv</Descriptions.Item>
          </Descriptions>
          
          {trainingModel && (
            <div style={{ marginTop: 24, textAlign: 'center' }}>
              <Progress percent={50} status="active" />
              <Text>正在训练模型，请耐心等待...</Text>
            </div>
          )}
        </Spin>
      </Modal>
    </div>
  );
};

export default ModelManagement; 