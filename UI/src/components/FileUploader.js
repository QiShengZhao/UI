import React, { useState } from 'react';
import { Upload, Button, message, Typography, Space } from 'antd';
import { UploadOutlined, FileExcelOutlined } from '@ant-design/icons';
import Papa from 'papaparse';

const { Title, Text } = Typography;

/**
 * 通用文件上传组件，支持CSV文件上传
 * @param {function} onFileUploaded - 文件解析后的回调函数，参数为解析后的数据
 * @param {string} title - 上传区域的标题
 * @param {string} description - 上传区域的描述文字
 */
const FileUploader = ({ onFileUploaded, title = '上传数据', description = '请上传CSV格式的数据文件' }) => {
  const [fileList, setFileList] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [fileData, setFileData] = useState(null);

  const handleUpload = (file) => {
    setUploading(true);
    
    // 判断文件类型
    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
      message.error('只支持上传CSV文件');
      setUploading(false);
      return false;
    }

    // 使用PapaParse解析CSV文件，增强解析配置
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true, // 跳过空行
      delimiter: '', // 自动检测分隔符
      dynamicTyping: true, // 自动类型转换
      encoding: 'UTF-8', // 指定编码
      comments: '#', // 忽略以#开头的行
      transformHeader: (header) => header.trim(), // 修剪标题空白
      complete: (results) => {
        // 解析成功后的处理
        if (results.errors.length > 0) {
          console.error('解析错误:', results.errors);
          message.error('文件解析出错: ' + results.errors[0].message + '@' + file.name);
          setUploading(false);
          return;
        }

        // 过滤空记录
        const validData = results.data.filter(row => {
          // 检查记录是否全为空或只有空白字符
          return Object.values(row).some(val => val !== null && val !== undefined && String(val).trim() !== '');
        });

        if (validData.length === 0) {
          message.error('解析后无有效数据记录');
          setUploading(false);
          return;
        }

        setFileData(validData);
        message.success(`${file.name} 解析成功，共 ${validData.length} 条记录`);
        
        // 调用回调函数传递数据
        if (onFileUploaded) {
          onFileUploaded(validData);
        }
        
        setUploading(false);
      },
      error: (error) => {
        console.error('解析错误:', error);
        message.error('文件解析错误: ' + error.message);
        setUploading(false);
      }
    });

    // 保存文件到列表，用于显示
    setFileList([file]);
    return false; // 阻止默认上传行为
  };

  const uploadProps = {
    accept: '.csv',
    beforeUpload: handleUpload,
    fileList,
    onRemove: () => {
      setFileList([]);
      setFileData(null);
      if (onFileUploaded) {
        onFileUploaded(null);
      }
    }
  };

  const summarizeData = () => {
    if (!fileData || fileData.length === 0) return null;
    
    // 获取数据列名
    const columns = Object.keys(fileData[0]);
    
    return (
      <div style={{ marginTop: 16 }}>
        <Title level={5}>数据概览</Title>
        <Text>列数: {columns.length}</Text>
        <br />
        <Text>行数: {fileData.length}</Text>
        <br />
        <Text>字段名: {columns.join(', ')}</Text>
      </div>
    );
  };

  return (
    <div style={{ marginBottom: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Title level={4}>{title}</Title>
        <Text>{description}</Text>
        <Upload {...uploadProps} className="upload-list-inline">
          <Button icon={<UploadOutlined />} loading={uploading}>
            选择文件
          </Button>
          <Text type="secondary" style={{ marginLeft: 8 }}>
            支持CSV格式
          </Text>
        </Upload>
        {fileData && summarizeData()}
      </Space>
    </div>
  );
};

export default FileUploader; 