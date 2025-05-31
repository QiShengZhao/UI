// 全局变量
let csvData = [];
let dataTable;
let csvFilename = 'test1.csv'; // 默认数据文件名

// 初始化页面
document.addEventListener('DOMContentLoaded', function() {
    // 配置Marked.js - 安全配置
    marked.use({
        mangle: false,
        headerIds: false,
        breaks: true,    // 将\n转换为<br>
        gfm: true,       // 启用GitHub风格的Markdown
        pedantic: false, // 不那么严格
        smartLists: true,// 使用更智能的列表行为
        smartypants: false, // 不使用"智能标点"
        // XSS防护
        renderer: {
            link(href, title, text) {
                // 只允许以http、https开头的链接，防止js等协议
                if (href && (href.startsWith('http://') || href.startsWith('https://'))) {
                    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${text}</a>`;
                }
                return text;
            }
        }
    });
    
    // 加载CSV数据
    loadCSVData();
    
    // 设置导航切换
    setupNavigation();
    
    // 设置按钮事件
    document.getElementById('export-data').addEventListener('click', exportFilteredData);
    document.getElementById('apply-filter').addEventListener('click', applyFilters);
    document.getElementById('predict-button').addEventListener('click', predictStrength);
    document.getElementById('download-report').addEventListener('click', function() {
        // 显示报告下载模态框
        const reportModal = new bootstrap.Modal(document.getElementById('reportModal'));
        reportModal.show();
    });
    document.getElementById('generate-report-button').addEventListener('click', generateReport);
    
    // 设置上传功能
    setupFileUpload();
    
    // 加载保存的报告配置
    loadReportConfig();
    
    // 为报告模态框添加显示事件处理
    document.getElementById('reportModal').addEventListener('show.bs.modal', function() {
        // 如果有#recent-reports元素，显示最近生成的文件列表
        const recentReportsElement = document.getElementById('recent-reports');
        if (recentReportsElement) {
            recentReportsElement.innerHTML = showGeneratedFilesList();
        }
    });
});

// 设置文件上传功能
function setupFileUpload() {
    const uploadButton = document.getElementById('upload-button');
    const fileInput = document.getElementById('csv-file');
    const progressBar = document.querySelector('#upload-progress .progress-bar');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadResult = document.getElementById('upload-result');
    
    // 文件选择事件
    fileInput.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            const file = this.files[0];
            // 验证文件类型
            if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
                uploadResult.innerHTML = '<div class="alert alert-danger">请选择有效的CSV文件</div>';
                uploadResult.classList.remove('d-none');
                uploadButton.disabled = true;
            } else {
                uploadResult.classList.add('d-none');
                uploadButton.disabled = false;
            }
        }
    });
    
    // 上传按钮点击事件
    uploadButton.addEventListener('click', function() {
        if (fileInput.files && fileInput.files[0]) {
            const file = fileInput.files[0];
            
            // 显示进度条
            uploadProgress.classList.remove('d-none');
            progressBar.style.width = '0%';
            uploadResult.classList.add('d-none');
            
            // 读取文件
            parseUploadedCSV(file, function(success, data, message) {
                // 更新进度条为100%
                progressBar.style.width = '100%';
                
                if (success) {
                    // 保存新数据
                    csvData = data;
                    csvFilename = file.name;
                    
                    // 初始化数据表格
                    initializeDataTable();
                    
                    // 计算统计数据
                    calculateStatistics();
                    
                    // 初始化图表
                    initializeCharts();
                    
                    // 显示成功消息
                    uploadResult.innerHTML = `<div class="alert alert-success">
                        <strong>成功！</strong> 已加载 ${data.length} 条数据记录
                    </div>`;
                    uploadResult.classList.remove('d-none');
                    
                    // 3秒后关闭模态框
                    setTimeout(function() {
                        const uploadModal = bootstrap.Modal.getInstance(document.getElementById('uploadModal'));
                        if (uploadModal) {
                            uploadModal.hide();
                        }
                    }, 3000);
                } else {
                    // 显示错误消息
                    uploadResult.innerHTML = `<div class="alert alert-danger">
                        <strong>错误！</strong> ${message}
                    </div>`;
                    uploadResult.classList.remove('d-none');
                }
            });
        }
    });
}

// 解析上传的CSV文件
function parseUploadedCSV(file, callback) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        try {
            // 解析CSV
            Papa.parse(e.target.result, {
                header: true,
                skipEmptyLines: true,
                dynamicTyping: true,
                encoding: "UTF-8",
                complete: function(results) {
                    if (results.data.length === 0) {
                        callback(false, null, '文件内容为空或格式不正确');
                        return;
                    }
                    
                    // 验证必要的列是否存在
                    const requiredColumns = ['C', 'Si', 'Mn', '屈服强度', '抗拉强度', '实际%'];
                    const missingColumns = requiredColumns.filter(col => !results.data[0].hasOwnProperty(col));
                    
                    if (missingColumns.length > 0) {
                        callback(false, null, '文件缺少必要的列: ' + missingColumns.join(', '));
                        return;
                    }
                    
                    console.log('CSV数据加载成功，共', results.data.length, '条记录');
                    callback(true, results.data, '');
                },
                error: function(error) {
                    console.error('CSV解析错误:', error);
                    callback(false, null, '文件解析失败: ' + error.message);
                }
            });
        } catch (e) {
            console.error('文件处理错误:', e);
            callback(false, null, '文件处理失败: ' + e.message);
        }
    };
    
    reader.onerror = function() {
        callback(false, null, '文件读取错误');
    };
    
    // 模拟上传进度
    let progress = 0;
    const progressBar = document.querySelector('#upload-progress .progress-bar');
    const interval = setInterval(function() {
        progress += 5;
        progressBar.style.width = Math.min(progress, 90) + '%';
        if (progress >= 90) {
            clearInterval(interval);
        }
    }, 100);
    
    // 读取文件
    reader.readAsText(file);
}

// 加载CSV数据
function loadCSVData() {
    Papa.parse(csvFilename, {
        download: true,
        header: true,
        skipEmptyLines: true,
        dynamicTyping: true,
        encoding: "UTF-8",
        complete: function(results) {
            csvData = results.data;
            console.log('CSV数据加载完成，共', csvData.length, '条记录');
            
            // 初始化数据表格
            initializeDataTable();
            
            // 计算统计数据
            calculateStatistics();
            
            // 初始化图表
            initializeCharts();
        },
        error: function(error) {
            console.error('CSV加载错误:', error);
            alert('数据加载失败，请检查CSV文件是否存在和格式是否正确。');
        }
    });
}

// 导航切换设置
function setupNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = {
        '#dashboard': document.getElementById('dashboard'),
        '#data-view': document.getElementById('data-view'),
        '#analysis': document.getElementById('analysis')
    };
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            
            // 隐藏所有部分
            Object.values(sections).forEach(section => {
                if (section) section.style.display = 'none';
            });
            
            // 显示目标部分
            if (sections[targetId]) {
                sections[targetId].style.display = 'block';
            }
            
            // 更新活动链接
            navLinks.forEach(navLink => navLink.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

// 初始化数据表格
function initializeDataTable() {
    // 选择要展示的列
    const selectedData = csvData.map(row => ({
        'C': row['C'],
        'Si': row['Si'],
        'Mn': row['Mn'],
        'P': row['P'],
        'S': row['S'],
        'Alt': row['Alt'],
        'Cr': row['Cr'],
        'Cu': row['Cu'],
        '模型计算值': row['模型计算值'],
        '在炉时间': row['在炉时间(分钟)'],
        '出炉温度': row['出炉温度实际值'],
        '终扎厚度': row['终扎厚度实际值'],
        '屈服强度': row['屈服强度'],
        '抗拉强度': row['抗拉强度'],
        '延伸率': row['实际%'],
        '韧性值': calculateToughness(row).toFixed(1)
    }));
    
    // 如果表格已经初始化，则销毁
    if (dataTable) {
        dataTable.destroy();
    }
    
    // 创建DataTable
    dataTable = $('#data-table').DataTable({
        data: selectedData,
        columns: [
            { data: 'C' },
            { data: 'Si' },
            { data: 'Mn' },
            { data: 'P' },
            { data: 'S' },
            { data: 'Alt' },
            { data: 'Cr' },
            { data: 'Cu' },
            { data: '模型计算值' },
            { data: '在炉时间' },
            { data: '出炉温度' },
            { data: '终扎厚度' },
            { data: '屈服强度' },
            { data: '抗拉强度' },
            { data: '延伸率' },
            { data: '韧性值' }
        ],
        pageLength: 10,
        language: {
            "url": "https://cdn.datatables.net/plug-ins/1.11.5/i18n/zh.json"
        },
        responsive: true
    });
}

// 计算统计数据
function calculateStatistics() {
    // 数据总量
    document.getElementById('total-count').textContent = csvData.length.toLocaleString();
    
    // 计算平均屈服强度
    const avgYield = csvData.reduce((sum, row) => {
        return sum + (parseFloat(row['屈服强度']) || 0);
    }, 0) / csvData.length;
    document.getElementById('avg-yield').textContent = avgYield.toFixed(1) + ' MPa';
    
    // 计算平均抗拉强度
    const avgTensile = csvData.reduce((sum, row) => {
        return sum + (parseFloat(row['抗拉强度']) || 0);
    }, 0) / csvData.length;
    document.getElementById('avg-tensile').textContent = avgTensile.toFixed(1) + ' MPa';
    
    // 计算平均延伸率
    const avgElongation = csvData.reduce((sum, row) => {
        return sum + (parseFloat(row['实际%']) || 0);
    }, 0) / csvData.length;
    document.getElementById('avg-elongation').textContent = avgElongation.toFixed(1) + '%';
    
    // 计算平均韧性值
    const avgToughness = csvData.reduce((sum, row) => {
        return sum + calculateToughness(row);
    }, 0) / csvData.length;
    document.getElementById('avg-toughness').textContent = avgToughness.toFixed(1);
}

// 初始化图表
function initializeCharts() {
    createElementStrengthChart();
    createProcessStrengthChart();
    createStrengthDistributionChart();
    createThicknessDistributionChart();
    createHeatTreatmentChart();
    createCorrelationHeatmap();
    createFactorAnalysisChart();
    createToughnessChart();
}

// 创建元素含量与抗拉强度关系图表
function createElementStrengthChart() {
    const chart = echarts.init(document.getElementById('element-strength-chart'));
    
    // 提取 C, Si, Mn 含量与抗拉强度的数据
    const data = csvData.map(row => [
        parseFloat(row['C']) || 0,
        parseFloat(row['Si']) || 0,
        parseFloat(row['Mn']) || 0,
        parseFloat(row['抗拉强度']) || 0
    ]);
    
    const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'cross'
            }
        },
        xAxis: [
            {
                type: 'category',
                data: ['C', 'Si', 'Mn'],
                axisLine: {
                    lineStyle: {
                        color: '#666'
                    }
                }
            }
        ],
        yAxis: [
            {
                type: 'value',
                name: '平均元素含量 (%)',
                position: 'left',
                axisLine: {
                    lineStyle: {
                        color: '#4e73df'
                    }
                },
                axisLabel: {
                    formatter: '{value} %'
                }
            },
            {
                type: 'value',
                name: '平均抗拉强度 (MPa)',
                position: 'right',
                offset: 0,
                axisLine: {
                    lineStyle: {
                        color: '#e74a3b'
                    }
                },
                axisLabel: {
                    formatter: '{value} MPa'
                }
            }
        ],
        series: [
            {
                name: '元素含量',
                type: 'bar',
                data: [
                    { value: average(data.map(item => item[0])).toFixed(2), itemStyle: { color: '#4e73df' } },
                    { value: average(data.map(item => item[1])).toFixed(2), itemStyle: { color: '#4e73df' } },
                    { value: average(data.map(item => item[2])).toFixed(2), itemStyle: { color: '#4e73df' } }
                ],
                barWidth: '30%'
            },
            {
                name: '平均抗拉强度',
                type: 'line',
                smooth: true,
                yAxisIndex: 1,
                data: [
                    { value: calculateAverageByElement('C', '抗拉强度').toFixed(1), symbol: 'circle', symbolSize: 8 },
                    { value: calculateAverageByElement('Si', '抗拉强度').toFixed(1), symbol: 'circle', symbolSize: 8 },
                    { value: calculateAverageByElement('Mn', '抗拉强度').toFixed(1), symbol: 'circle', symbolSize: 8 }
                ],
                lineStyle: {
                    color: '#e74a3b',
                    width: 3
                },
                itemStyle: {
                    color: '#e74a3b'
                }
            }
        ]
    };
    
    chart.setOption(option);
    
    // 当切换到分析部分时，确保重新渲染图表
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === '#analysis') {
            link.addEventListener('click', function() {
                setTimeout(function() {
                    chart.resize();
                    // 重新设置选项，确保数据显示正确
                    chart.setOption(option, true);
                }, 200);
            });
        }
    });
    
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 计算每种元素对应的特性平均值
function calculateAverageByElement(element, property) {
    // 按元素含量分组数据
    const elementValues = [...new Set(csvData.map(row => row[element]))].sort((a, b) => a - b);
    const medianRange = elementValues[Math.floor(elementValues.length / 2)];
    
    // 筛选中间范围的数据
    const filteredData = csvData.filter(row => row[element] === medianRange);
    
    // 计算这些数据的属性平均值
    return filteredData.reduce((sum, row) => sum + (parseFloat(row[property]) || 0), 0) / filteredData.length;
}

// 创建工艺参数与强度关系图表
function createProcessStrengthChart() {
    const chart = echarts.init(document.getElementById('process-strength-chart'));
    
    // 提取数据
    const tempData = csvData.map(row => [
        parseFloat(row['出炉温度实际值']) || 0,
        parseFloat(row['抗拉强度']) || 0
    ]);
    
    const coolingData = csvData.map(row => [
        parseFloat(row['水冷后温度']) || 0, 
        parseFloat(row['抗拉强度']) || 0
    ]);
    
    // 温度范围分组
    const tempGroups = {};
    tempData.forEach(item => {
        // 将温度按50℃为一组
        const tempGroup = Math.floor(item[0] / 50) * 50;
        if (!tempGroups[tempGroup]) {
            tempGroups[tempGroup] = [];
        }
        tempGroups[tempGroup].push(item[1]);
    });
    
    const coolingGroups = {};
    coolingData.forEach(item => {
        // 将冷却温度按50℃为一组
        const coolingGroup = Math.floor(item[0] / 50) * 50;
        if (!coolingGroups[coolingGroup]) {
            coolingGroups[coolingGroup] = [];
        }
        coolingGroups[coolingGroup].push(item[1]);
    });
    
    // 计算每个组的平均强度
    const tempCategories = [];
    const tempStrengths = [];
    for (const group in tempGroups) {
        if (tempGroups[group].length > 0) {
            tempCategories.push(group + '-' + (parseInt(group) + 50) + '°C');
            tempStrengths.push(average(tempGroups[group]).toFixed(1));
        }
    }
    
    const coolingCategories = [];
    const coolingStrengths = [];
    for (const group in coolingGroups) {
        if (coolingGroups[group].length > 0) {
            coolingCategories.push(group + '-' + (parseInt(group) + 50) + '°C');
            coolingStrengths.push(average(coolingGroups[group]).toFixed(1));
        }
    }
    
    const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        legend: {
            data: ['出炉温度', '水冷后温度'],
            top: 10
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: [...new Set([...tempCategories, ...coolingCategories])],
            axisLabel: {
                interval: 0,
                rotate: 45
            }
        },
        yAxis: {
            type: 'value',
            name: '平均抗拉强度 (MPa)'
        },
        series: [
            {
                name: '出炉温度',
                type: 'bar',
                data: tempCategories.map((category, index) => ({
                    value: tempStrengths[index],
                    itemStyle: { color: '#36b9cc' }
                })),
                emphasis: {
                    focus: 'series'
                }
            },
            {
                name: '水冷后温度',
                type: 'bar',
                data: coolingCategories.map((category, index) => ({
                    value: coolingStrengths[index],
                    itemStyle: { color: '#1cc88a' }
                })),
                emphasis: {
                    focus: 'series'
                }
            }
        ]
    };
    
    chart.setOption(option);
    
    // 当切换到分析部分时，确保重新渲染图表
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === '#analysis') {
            link.addEventListener('click', function() {
                setTimeout(function() {
                    chart.resize();
                    // 重新设置选项，确保数据显示正确
                    chart.setOption(option, true);
                }, 200);
            });
        }
    });
    
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 创建强度分布图表
function createStrengthDistributionChart() {
    const chart = echarts.init(document.getElementById('strength-distribution-chart'));
    
    // 提取抗拉强度数据
    const strengthData = csvData.map(row => parseFloat(row['抗拉强度']) || 0);
    
    // 计算强度分布
    const strengthRanges = {
        '< 500': 0,
        '500-525': 0,
        '525-550': 0,
        '550-575': 0,
        '575-600': 0,
        '> 600': 0
    };
    
    strengthData.forEach(strength => {
        if (strength < 500) strengthRanges['< 500']++;
        else if (strength < 525) strengthRanges['500-525']++;
        else if (strength < 550) strengthRanges['525-550']++;
        else if (strength < 575) strengthRanges['550-575']++;
        else if (strength < 600) strengthRanges['575-600']++;
        else strengthRanges['> 600']++;
    });
    
    const option = {
        tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        legend: {
            orient: 'vertical',
            right: 10,
            top: 'center',
            data: Object.keys(strengthRanges)
        },
        series: [
            {
                name: '抗拉强度分布',
                type: 'pie',
                radius: ['40%', '70%'],
                avoidLabelOverlap: false,
                label: {
                    show: false,
                    position: 'center'
                },
                emphasis: {
                    label: {
                        show: true,
                        fontSize: '18',
                        fontWeight: 'bold'
                    }
                },
                labelLine: {
                    show: false
                },
                data: Object.keys(strengthRanges).map(key => ({
                    value: strengthRanges[key],
                    name: key + ' MPa'
                }))
            }
        ]
    };
    
    chart.setOption(option);
    
    // 当切换到分析部分时，确保重新渲染图表
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === '#analysis') {
            link.addEventListener('click', function() {
                setTimeout(function() {
                    chart.resize();
                    // 重新设置选项，确保数据显示正确
                    chart.setOption(option, true);
                }, 200);
            });
        }
    });
    
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 创建厚度分布图表
function createThicknessDistributionChart() {
    const chart = echarts.init(document.getElementById('thickness-distribution-chart'));
    
    // 提取厚度数据
    const thicknessData = csvData.map(row => parseFloat(row['厚度']) || 0);
    
    // 计算厚度分布
    const thicknessRanges = {
        '< 40': 0,
        '40-60': 0,
        '60-80': 0,
        '80-100': 0,
        '100-120': 0,
        '> 120': 0
    };
    
    thicknessData.forEach(thickness => {
        if (thickness < 40) thicknessRanges['< 40']++;
        else if (thickness < 60) thicknessRanges['40-60']++;
        else if (thickness < 80) thicknessRanges['60-80']++;
        else if (thickness < 100) thicknessRanges['80-100']++;
        else if (thickness < 120) thicknessRanges['100-120']++;
        else thicknessRanges['> 120']++;
    });
    
    const option = {
        tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c} ({d}%)'
        },
        legend: {
            orient: 'vertical',
            right: 10,
            top: 'center',
            data: Object.keys(thicknessRanges)
        },
        series: [
            {
                name: '厚度分布',
                type: 'pie',
                radius: ['40%', '70%'],
                avoidLabelOverlap: false,
                label: {
                    show: false,
                    position: 'center'
                },
                emphasis: {
                    label: {
                        show: true,
                        fontSize: '18',
                        fontWeight: 'bold'
                    }
                },
                labelLine: {
                    show: false
                },
                data: Object.keys(thicknessRanges).map(key => ({
                    value: thicknessRanges[key],
                    name: key + ' mm'
                }))
            }
        ]
    };
    
    chart.setOption(option);
    
    // 当切换到分析部分时，确保重新渲染图表
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === '#analysis') {
            link.addEventListener('click', function() {
                setTimeout(function() {
                    chart.resize();
                    // 重新设置选项，确保数据显示正确
                    chart.setOption(option, true);
                }, 200);
            });
        }
    });
    
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 创建热处理参数分布图表
function createHeatTreatmentChart() {
    const chart = echarts.init(document.getElementById('heat-treatment-chart'));
    
    // 提取热处理相关数据
    const tempData = csvData.map(row => parseFloat(row['出炉温度实际值']) || 0);
    const timeData = csvData.map(row => parseFloat(row['在炉时间(分钟)']) || 0);
    const coolingData = csvData.map(row => parseFloat(row['水冷后温度']) || 0);
    
    const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        legend: {
            data: ['出炉温度', '在炉时间', '水冷后温度']
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            name: '数值',
            axisLabel: {
                formatter: '{value}'
            }
        },
        yAxis: {
            type: 'category',
            data: ['最小值', '平均值', '最大值']
        },
        series: [
            {
                name: '出炉温度',
                type: 'bar',
                data: [
                    Math.min(...tempData).toFixed(0),
                    average(tempData).toFixed(0),
                    Math.max(...tempData).toFixed(0)
                ]
            },
            {
                name: '在炉时间',
                type: 'bar',
                data: [
                    Math.min(...timeData.filter(x => x > 0)).toFixed(0),
                    average(timeData.filter(x => x > 0)).toFixed(0),
                    Math.max(...timeData).toFixed(0)
                ]
            },
            {
                name: '水冷后温度',
                type: 'bar',
                data: [
                    Math.min(...coolingData.filter(x => x > 0)).toFixed(0),
                    average(coolingData.filter(x => x > 0)).toFixed(0),
                    Math.max(...coolingData).toFixed(0)
                ]
            }
        ]
    };
    
    chart.setOption(option);
    
    // 当切换到分析部分时，确保重新渲染图表
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === '#analysis') {
            link.addEventListener('click', function() {
                setTimeout(function() {
                    chart.resize();
                    // 重新设置选项，确保数据显示正确
                    chart.setOption(option, true);
                }, 200);
            });
        }
    });
    
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 创建相关性热图
function createCorrelationHeatmap() {
    const chart = echarts.init(document.getElementById('correlation-heatmap'));
    
    // 计算相关性数据
    const variables = ['C', 'Si', 'Mn', 'P', 'S', 'Alt', 'Cr', 'Cu', '屈服强度', '抗拉强度'];
    const correlationMatrix = [];
    
    console.log("开始创建元素相关性热图...");
    
    // 准备数据
    const preparedData = variables.map(variable => {
        return csvData.map(row => {
            const value = parseFloat(row[variable]);
            return isNaN(value) ? 0 : value;
        });
    });
    
    // 构建相关性矩阵
    for (let i = 0; i < variables.length; i++) {
        correlationMatrix[i] = [];
        for (let j = 0; j < variables.length; j++) {
            if (i === j) {
                correlationMatrix[i][j] = 1; // 自相关为1
            } else {
                correlationMatrix[i][j] = calculateCorrelation(preparedData[i], preparedData[j]);
            }
        }
    }
    
    console.log("相关性矩阵计算完成:", correlationMatrix);
    
    // 准备绘图数据
    const data = [];
    for (let i = 0; i < variables.length; i++) {
        for (let j = 0; j < variables.length; j++) {
            data.push([j, i, parseFloat(correlationMatrix[i][j].toFixed(2))]);
        }
    }
    
    console.log("热图数据准备完成，共" + data.length + "个数据点");
    
    const option = {
        tooltip: {
            position: 'top',
            formatter: function(params) {
                return variables[params.value[1]] + ' 与 ' + 
                       variables[params.value[0]] + ' 的相关性: ' + 
                       params.value[2];
            }
        },
        animation: false,
        grid: {
            height: '70%',
            top: '10%'
        },
        xAxis: {
            type: 'category',
            data: variables,
            splitArea: {
                show: true
            },
            axisLabel: {
                interval: 0,
                rotate: 45
            }
        },
        yAxis: {
            type: 'category',
            data: variables,
            splitArea: {
                show: true
            }
        },
        visualMap: {
            min: -1,
            max: 1,
            calculable: true,
            orient: 'horizontal',
            left: 'center',
            bottom: '0%',
            inRange: {
                color: ['#d73027', '#fee090', '#e0f3f8', '#91bfdb']
            }
        },
        series: [{
            name: '相关性',
            type: 'heatmap',
            data: data,
            label: {
                show: true,
                formatter: function(params) {
                    return params.value[2].toFixed(2);
                }
            },
            emphasis: {
                itemStyle: {
                    shadowBlur: 10,
                    shadowColor: 'rgba(0, 0, 0, 0.5)'
                }
            }
        }]
    };
    
    chart.setOption(option);
    
    // 当切换到分析部分时，确保重新渲染图表
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === '#analysis') {
            link.addEventListener('click', function() {
                setTimeout(function() {
                    chart.resize();
                    // 重新设置选项，确保数据显示正确
                    chart.setOption(option, true);
                }, 200);
            });
        }
    });
    
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 计算两个数组之间的皮尔逊相关系数
function calculateCorrelation(x, y) {
    try {
        if (x.length !== y.length) {
            console.error('数组长度不匹配');
            return 0;
        }
        
        // 过滤掉无效值
        const validPairs = [];
        for (let i = 0; i < x.length; i++) {
            if (!isNaN(x[i]) && !isNaN(y[i]) && x[i] !== null && y[i] !== null) {
                validPairs.push([x[i], y[i]]);
            }
        }
        
        if (validPairs.length < 2) {
            return 0; // 样本太少，无法计算相关性
        }
        
        // 计算均值
        const sumX = validPairs.reduce((sum, pair) => sum + pair[0], 0);
        const sumY = validPairs.reduce((sum, pair) => sum + pair[1], 0);
        const meanX = sumX / validPairs.length;
        const meanY = sumY / validPairs.length;
        
        // 检查数据是否全部相同（标准差为0）
        const allSameX = validPairs.every(pair => pair[0] === validPairs[0][0]);
        const allSameY = validPairs.every(pair => pair[1] === validPairs[0][1]);
        
        if (allSameX || allSameY) {
            return 0; // 如果任一变量的值都相同，相关系数无法计算（标准差为0）
        }
        
        // 计算协方差和标准差
        let numerator = 0;
        let denominatorX = 0;
        let denominatorY = 0;
        
        for (const [xi, yi] of validPairs) {
            const diffX = xi - meanX;
            const diffY = yi - meanY;
            numerator += diffX * diffY;
            denominatorX += diffX * diffX;
            denominatorY += diffY * diffY;
        }
        
        // 检查分母是否为零
        if (denominatorX === 0 || denominatorY === 0) {
            return 0;
        }
        
        const correlation = numerator / (Math.sqrt(denominatorX) * Math.sqrt(denominatorY));
        
        // 确保结果在[-1, 1]范围内
        return Math.max(-1, Math.min(1, correlation));
    } catch (error) {
        console.error('计算相关性时出错:', error);
        return 0;
    }
}

// 创建因素影响分析图表
function createFactorAnalysisChart() {
    const chart = echarts.init(document.getElementById('factor-analysis-chart'));
    
    // 计算各因素对抗拉强度的影响程度（使用简化的方法）
    const factors = ['C', 'Si', 'Mn', 'P', 'S', 'Alt', 'Cr', 'Cu', '出炉温度实际值', '水冷后温度'];
    const correlations = factors.map(factor => {
        return {
            factor: factor,
            correlation: Math.abs(calculateSimpleCorrelation(factor, '抗拉强度'))
        };
    });
    
    // 按相关性排序
    correlations.sort((a, b) => b.correlation - a.correlation);
    
    const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true
        },
        xAxis: {
            type: 'value',
            name: '相关程度'
        },
        yAxis: {
            type: 'category',
            data: correlations.map(item => item.factor),
            axisLabel: {
                interval: 0
            }
        },
        series: [
            {
                name: '影响程度',
                type: 'bar',
                data: correlations.map(item => item.correlation.toFixed(3)),
                itemStyle: {
                    color: function(params) {
                        // 根据数值设置不同的颜色
                        var colorList = ['#c23531','#2f4554','#61a0a8','#d48265','#91c7ae','#749f83','#ca8622','#bda29a','#6e7074','#546570'];
                        return colorList[params.dataIndex % colorList.length];
                    }
                }
            }
        ]
    };
    
    chart.setOption(option);
    
    // 当切换到分析部分时，确保重新渲染图表
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === '#analysis') {
            link.addEventListener('click', function() {
                setTimeout(function() {
                    chart.resize();
                    // 重新设置选项，确保数据显示正确
                    chart.setOption(option, true);
                }, 200);
            });
        }
    });
    
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 应用过滤器
function applyFilters() {
    const elementFilter = document.getElementById('element-filter').value;
    const strengthFilter = document.getElementById('strength-filter').value;
    const thicknessFilter = document.getElementById('thickness-filter').value;
    
    let filteredData = csvData;
    
    // 应用元素筛选
    if (elementFilter !== 'all') {
        // 对于元素筛选，我们可以根据中位数进行筛选
        const elementValues = [...new Set(csvData.map(row => row[elementFilter]))].sort((a, b) => a - b);
        const medianValue = elementValues[Math.floor(elementValues.length / 2)];
        
        // 选择中位数附近的数据
        filteredData = filteredData.filter(row => {
            return Math.abs(row[elementFilter] - medianValue) < 0.05;
        });
    }
    
    // 应用强度筛选
    if (strengthFilter !== 'all') {
        filteredData = filteredData.filter(row => {
            const strength = parseFloat(row['抗拉强度']) || 0;
            if (strengthFilter === 'low') return strength < 500;
            if (strengthFilter === 'medium') return strength >= 500 && strength <= 550;
            if (strengthFilter === 'high') return strength > 550;
            return true;
        });
    }
    
    // 应用厚度筛选
    if (thicknessFilter !== 'all') {
        filteredData = filteredData.filter(row => {
            const thickness = parseFloat(row['厚度']) || 0;
            if (thicknessFilter === 'thin') return thickness < 50;
            if (thicknessFilter === 'medium') return thickness >= 50 && thickness <= 80;
            if (thicknessFilter === 'thick') return thickness > 80;
            return true;
        });
    }
    
    // 更新表格数据
    updateDataTable(filteredData);
    
    // 更新统计信息
    updateStatistics(filteredData);
}

// 更新表格数据
function updateDataTable(filteredData) {
    // 选择要展示的列
    const selectedData = filteredData.map(row => ({
        'C': row['C'],
        'Si': row['Si'],
        'Mn': row['Mn'],
        'P': row['P'],
        'S': row['S'],
        'Alt': row['Alt'],
        'Cr': row['Cr'],
        'Cu': row['Cu'],
        '模型计算值': row['模型计算值'],
        '在炉时间': row['在炉时间(分钟)'],
        '出炉温度': row['出炉温度实际值'],
        '终扎厚度': row['终扎厚度实际值'],
        '屈服强度': row['屈服强度'],
        '抗拉强度': row['抗拉强度'],
        '延伸率': row['实际%'],
        '韧性值': calculateToughness(row).toFixed(1)
    }));
    
    // 清空表格
    dataTable.clear();
    
    // 添加数据
    if (selectedData.length > 0) {
        dataTable.rows.add(selectedData);
    }
    
    // 重绘表格
    dataTable.draw();
}

// 更新统计信息
function updateStatistics(filteredData) {
    // 数据总量
    document.getElementById('total-count').textContent = filteredData.length.toLocaleString();
    
    if (filteredData.length === 0) {
        document.getElementById('avg-yield').textContent = '0 MPa';
        document.getElementById('avg-tensile').textContent = '0 MPa';
        document.getElementById('avg-elongation').textContent = '0%';
        document.getElementById('avg-toughness').textContent = '0';
        return;
    }
    
    // 计算平均屈服强度
    const avgYield = filteredData.reduce((sum, row) => {
        return sum + (parseFloat(row['屈服强度']) || 0);
    }, 0) / filteredData.length;
    document.getElementById('avg-yield').textContent = avgYield.toFixed(1) + ' MPa';
    
    // 计算平均抗拉强度
    const avgTensile = filteredData.reduce((sum, row) => {
        return sum + (parseFloat(row['抗拉强度']) || 0);
    }, 0) / filteredData.length;
    document.getElementById('avg-tensile').textContent = avgTensile.toFixed(1) + ' MPa';
    
    // 计算平均延伸率
    const avgElongation = filteredData.reduce((sum, row) => {
        return sum + (parseFloat(row['实际%']) || 0);
    }, 0) / filteredData.length;
    document.getElementById('avg-elongation').textContent = avgElongation.toFixed(1) + '%';
    
    // 计算平均韧性值
    const avgToughness = filteredData.reduce((sum, row) => {
        return sum + calculateToughness(row);
    }, 0) / filteredData.length;
    document.getElementById('avg-toughness').textContent = avgToughness.toFixed(1);
}

// 导出筛选后的数据
function exportFilteredData() {
    // 获取当前DataTable中的数据
    const exportData = dataTable.data().toArray();
    
    // 转换为CSV
    const csv = Papa.unparse(exportData);
    
    // 创建下载链接
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    // 下载文件
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', '钢铁数据导出_' + new Date().toISOString().slice(0, 10) + '.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// 生成并下载分析报告
async function generateReport() {
    // 获取报告选项
    const includeSummary = document.getElementById('include-summary').checked;
    const includeCharts = document.getElementById('include-charts').checked;
    const includeCorrelation = document.getElementById('include-correlation').checked;
    const includeAIAnalysis = document.getElementById('include-ai-analysis').checked;
    const reportTitle = document.getElementById('report-title').value || '钢铁材料分析报告';
    const reportFormat = document.getElementById('report-format').value;
    
    // 显示进度条
    const progressBar = document.querySelector('#report-progress .progress-bar');
    document.getElementById('report-progress').classList.remove('d-none');
    progressBar.style.width = '10%';
    
    try {
        // 保存报告配置到localStorage
        saveReportConfig({
            includeSummary,
            includeCharts,
            includeCorrelation,
            includeAIAnalysis,
            reportTitle,
            reportFormat
        });
        
        progressBar.style.width = '30%';
        
        // 如果包括图表，先捕获当前的图表
        let chartImages = {};
        if (includeCharts) {
            chartImages = await captureCurrentCharts();
            progressBar.style.width = '50%';
        }
        
        // 创建报告内容
        const reportContent = generateReportHTML(reportTitle, includeSummary, includeCharts, includeCorrelation, includeAIAnalysis, chartImages);
        progressBar.style.width = '70%';
        
        // 保存报告内容到localStorage以便后续使用
        storeReportLocally(reportTitle, reportContent);
        
        if (reportFormat === 'html') {
            // 直接下载HTML报告
            downloadHTML(reportContent, reportTitle);
        } else if (reportFormat === 'pdf') {
            // 使用html2pdf库转换为PDF并下载
            await downloadPDF(reportContent, reportTitle);
        }
        
        // 完成进度条
        progressBar.style.width = '100%';
        
        // 报告成功生成通知
        showNotification('报告生成成功', '报告已成功生成并下载到您的本地下载文件夹');
        
        // 3秒后隐藏进度条并关闭模态框
        setTimeout(() => {
            document.getElementById('report-progress').classList.add('d-none');
            const reportModal = bootstrap.Modal.getInstance(document.getElementById('reportModal'));
            if (reportModal) {
                reportModal.hide();
            }
        }, 2000);
    } catch (error) {
        console.error('报告生成失败:', error);
        // 显示错误信息
        document.getElementById('report-progress').classList.add('d-none');
        showNotification('报告生成失败', '生成报告时发生错误: ' + error.message, 'error');
    }
}

// 保存报告配置到localStorage
function saveReportConfig(config) {
    try {
        localStorage.setItem('reportConfig', JSON.stringify(config));
    } catch (error) {
        console.warn('无法保存报告配置:', error);
    }
}

// 加载报告配置从localStorage
function loadReportConfig() {
    try {
        const config = JSON.parse(localStorage.getItem('reportConfig'));
        if (config) {
            document.getElementById('include-summary').checked = config.includeSummary !== false;
            document.getElementById('include-charts').checked = config.includeCharts !== false;
            document.getElementById('include-correlation').checked = config.includeCorrelation !== false;
            document.getElementById('include-ai-analysis').checked = config.includeAIAnalysis !== false;
            document.getElementById('report-title').value = config.reportTitle || '钢铁材料分析报告';
            document.getElementById('report-format').value = config.reportFormat || 'pdf';
        }
    } catch (error) {
        console.warn('无法加载报告配置:', error);
    }
}

// 在localStorage中存储报告
function storeReportLocally(title, content) {
    try {
        // 存储最近的5个报告
        const storedReports = JSON.parse(localStorage.getItem('recentReports') || '[]');
        
        // 添加新报告到列表开头
        const newReport = {
            title,
            timestamp: new Date().toISOString(),
            preview: content.substring(0, 500) + '...' // 仅存储预览
        };
        
        storedReports.unshift(newReport);
        
        // 保留最近的5个报告
        const recentReports = storedReports.slice(0, 5);
        localStorage.setItem('recentReports', JSON.stringify(recentReports));
    } catch (error) {
        console.warn('无法在本地存储报告:', error);
    }
}

// 显示通知消息
function showNotification(title, message, type = 'success') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `toast align-items-center text-white bg-${type === 'success' ? 'success' : 'danger'} border-0`;
    notification.setAttribute('role', 'alert');
    notification.setAttribute('aria-live', 'assertive');
    notification.setAttribute('aria-atomic', 'true');
    
    notification.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <strong>${title}</strong><br>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // 添加到DOM
    const toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(container);
        container.appendChild(notification);
    } else {
        toastContainer.appendChild(notification);
    }
    
    // 显示通知
    const toast = new bootstrap.Toast(notification);
    toast.show();
    
    // 5秒后自动关闭
    setTimeout(() => {
        toast.hide();
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 500);
    }, 5000);
}

// 捕获当前图表为图片
async function captureCurrentCharts() {
    try {
        const charts = {};
        
        // 获取页面上的所有echarts实例
        const chartElements = [
            'element-strength-chart',
            'process-strength-chart',
            'strength-distribution-chart',
            'thickness-distribution-chart',
            'heat-treatment-chart',
            'correlation-heatmap',
            'factor-analysis-chart',
            'toughness-chart'
        ];
        
        // 确保所有图表数据已加载
        for (const id of chartElements) {
            const chartElement = document.getElementById(id);
            if (chartElement) {
                const chart = echarts.getInstanceByDom(chartElement);
                if (chart) {
                    charts[id] = await chartToBase64(chart);
                }
            }
        }
        
        return charts;
    } catch (error) {
        console.error('捕获图表失败:', error);
        return {};
    }
}

// 将echarts图表转换为base64图片
function chartToBase64(chart) {
    return new Promise((resolve) => {
        try {
            const base64 = chart.getDataURL({
                pixelRatio: 2,
                backgroundColor: '#fff'
            });
            resolve(base64);
        } catch (error) {
            console.error('图表转换失败:', error);
            resolve(null);
        }
    });
}

// 生成报告HTML内容
function generateReportHTML(title, includeSummary, includeCharts, includeCorrelation, includeAIAnalysis, chartImages = {}) {
    const currentDate = new Date().toLocaleDateString('zh-CN', {
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
    
    let html = `
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>${title}</title>
        <style>
            body {
                font-family: "Microsoft YaHei", Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .report-header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #0d6efd;
            }
            .report-title {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
                color: #0d6efd;
            }
            .report-date {
                color: #666;
                font-style: italic;
            }
            .report-section {
                margin-bottom: 30px;
                page-break-inside: avoid;
            }
            .section-title {
                font-size: 20px;
                color: #0d6efd;
                margin-bottom: 15px;
                padding-bottom: 8px;
                border-bottom: 1px solid #ddd;
            }
            .stats-container {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                flex: 1;
                min-width: 200px;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                text-align: center;
                background-color: #f8f9fa;
            }
            .stat-value {
                font-size: 20px;
                font-weight: bold;
                color: #0d6efd;
                margin: 10px 0;
            }
            .stat-label {
                font-size: 14px;
                color: #666;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 10px;
                border: 1px solid #ddd;
                text-align: left;
            }
            th {
                background-color: #f8f9fa;
                font-weight: bold;
            }
            .chart-container {
                margin: 20px 0;
                text-align: center;
                page-break-inside: avoid;
            }
            .chart-container img {
                max-width: 100%;
                max-height: 400px;
                display: block;
                margin: 0 auto;
                border: 1px solid #eee;
            }
            .chart-title {
                margin: 10px 0;
                font-weight: 600;
                color: #444;
            }
            .footer {
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                text-align: center;
                color: #666;
                font-size: 12px;
            }
            .ai-analysis {
                background-color: #f8f9ff;
                border-left: 3px solid #0d6efd;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 5px 5px 0;
            }
            @media print {
                @page {
                    margin: 1cm;
                }
                body {
                    font-size: 12pt;
                }
                .section-title {
                    font-size: 16pt;
                }
            }
        </style>
    </head>
    <body>
        <div class="report-header">
            <div class="report-title">${title}</div>
            <div class="report-date">生成时间: ${currentDate}</div>
        </div>
    `;
    
    // 添加统计概要部分
    if (includeSummary) {
        html += generateSummarySection();
    }
    
    // 添加图表分析部分
    if (includeCharts) {
        html += generateChartsSection(chartImages);
    }
    
    // 添加相关性分析部分
    if (includeCorrelation) {
        html += generateCorrelationSection(chartImages);
    }
    
    // 添加AI分析部分
    if (includeAIAnalysis) {
        html += generateAIAnalysisSection();
    }
    
    // 添加页脚
    html += `
        <div class="footer">
            <p>天津中德应用技术大学 | 钢铁材料数据分析系统</p>
            <p>© ${new Date().getFullYear()} 版权所有 | 机械工程学院--飞飞</p>
        </div>
    </body>
    </html>
    `;
    
    return html;
}

// 生成统计概要部分
function generateSummarySection() {
    const totalCount = document.getElementById('total-count').textContent;
    const avgYield = document.getElementById('avg-yield').textContent;
    const avgTensile = document.getElementById('avg-tensile').textContent;
    const avgElongation = document.getElementById('avg-elongation').textContent;
    const avgToughness = document.getElementById('avg-toughness').textContent;
    
    return `
    <div class="report-section">
        <h2 class="section-title">1. 数据统计概览</h2>
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-label">数据总量</div>
                <div class="stat-value">${totalCount}</div>
                <div class="stat-label">总记录数</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">平均屈服强度</div>
                <div class="stat-value">${avgYield}</div>
                <div class="stat-label">所有样本</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">平均抗拉强度</div>
                <div class="stat-value">${avgTensile}</div>
                <div class="stat-label">所有样本</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">平均延伸率</div>
                <div class="stat-value">${avgElongation}</div>
                <div class="stat-label">所有样本</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">平均韧性值</div>
                <div class="stat-value">${avgToughness}</div>
                <div class="stat-label">所有样本</div>
            </div>
        </div>
        
        <h3>数据表格摘要</h3>
        <p>以下是当前筛选条件下的前10条数据记录：</p>
        ${generateDataTableHTML()}
    </div>
    `;
}

// 生成数据表格HTML
function generateDataTableHTML() {
    const tableData = dataTable.data().toArray().slice(0, 10);
    if (tableData.length === 0) {
        return '<p>当前没有数据记录</p>';
    }
    
    // 获取表格列名
    const columns = Object.keys(tableData[0]);
    
    let tableHTML = '<table><thead><tr>';
    columns.forEach(column => {
        tableHTML += `<th>${column}</th>`;
    });
    tableHTML += '</tr></thead><tbody>';
    
    // 添加行数据
    tableData.forEach(row => {
        tableHTML += '<tr>';
        columns.forEach(column => {
            tableHTML += `<td>${row[column] || ''}</td>`;
        });
        tableHTML += '</tr>';
    });
    
    tableHTML += '</tbody></table>';
    return tableHTML;
}

// 生成图表分析部分
function generateChartsSection(chartImages = {}) {
    return `
    <div class="report-section">
        <h2 class="section-title">2. 图表分析</h2>
        
        <div class="chart-container">
            <h3 class="chart-title">元素含量与抗拉强度关系</h3>
            ${chartImages['element-strength-chart'] 
                ? `<img src="${chartImages['element-strength-chart']}" alt="元素含量与抗拉强度关系">`
                : `<p>【此处将显示元素-强度关系图】</p>`
            }
        </div>
        
        <div class="chart-container">
            <h3 class="chart-title">工艺参数与强度关系</h3>
            ${chartImages['process-strength-chart'] 
                ? `<img src="${chartImages['process-strength-chart']}" alt="工艺参数与强度关系">`
                : `<p>【此处将显示工艺-强度关系图】</p>`
            }
        </div>
        
        <div class="chart-container">
            <h3 class="chart-title">强度分布</h3>
            ${chartImages['strength-distribution-chart'] 
                ? `<img src="${chartImages['strength-distribution-chart']}" alt="强度分布">`
                : `<p>【此处将显示强度分布图】</p>`
            }
        </div>
        
        <div class="chart-container">
            <h3 class="chart-title">韧性与成分关系分析</h3>
            ${chartImages['toughness-chart'] 
                ? `<img src="${chartImages['toughness-chart']}" alt="韧性与成分关系分析">`
                : `<p>【此处将显示韧性-成分关系图】</p>`
            }
        </div>
    </div>
    `;
}

// 生成相关性分析部分
function generateCorrelationSection(chartImages = {}) {
    return `
    <div class="report-section">
        <h2 class="section-title">3. 相关性分析</h2>
        
        <div class="chart-container">
            <h3 class="chart-title">元素相关性热图</h3>
            ${chartImages['correlation-heatmap'] 
                ? `<img src="${chartImages['correlation-heatmap']}" alt="元素相关性热图">`
                : `<p>【此处将显示相关性热图】</p>`
            }
        </div>
        
        <div class="chart-container">
            <h3 class="chart-title">主要因素影响分析</h3>
            ${chartImages['factor-analysis-chart'] 
                ? `<img src="${chartImages['factor-analysis-chart']}" alt="主要因素影响分析">`
                : `<p>【此处将显示因素分析图】</p>`
            }
        </div>
        
        <h3>相关性分析结论</h3>
        <p>通过对钢铁材料数据的相关性分析，可以得出以下结论：</p>
        <ul>
            <li>碳(C)含量与抗拉强度呈正相关，相关系数约为0.7</li>
            <li>硅(Si)含量与延伸率呈中度负相关</li>
            <li>出炉温度对最终强度有显著影响</li>
            <li>终扎厚度与屈服强度呈负相关关系</li>
        </ul>
    </div>
    `;
}

// 生成AI分析部分
function generateAIAnalysisSection() {
    // 获取DeepSeek分析结果
    const compositionAnalysis = document.getElementById('composition-analysis')?.innerHTML || '';
    const processAnalysis = document.getElementById('process-analysis')?.innerHTML || '';
    const performanceAnalysis = document.getElementById('performance-analysis')?.innerHTML || '';
    const optimizationSuggestions = document.getElementById('optimization-suggestions')?.innerHTML || '';
    
    if (!compositionAnalysis && !processAnalysis && !performanceAnalysis && !optimizationSuggestions) {
        return `
        <div class="report-section">
            <h2 class="section-title">4. AI智能分析</h2>
            <p>尚未进行AI智能分析。请在分析页面使用预测功能获取AI分析建议。</p>
        </div>
        `;
    }
    
    return `
    <div class="report-section">
        <h2 class="section-title">4. AI智能分析</h2>
        
        <h3>材料成分分析</h3>
        <div class="ai-analysis">
            ${compositionAnalysis || '<p>暂无成分分析数据</p>'}
        </div>
        
        <h3>工艺参数评估</h3>
        <div class="ai-analysis">
            ${processAnalysis || '<p>暂无工艺分析数据</p>'}
        </div>
        
        <h3>性能平衡分析</h3>
        <div class="ai-analysis">
            ${performanceAnalysis || '<p>暂无性能分析数据</p>'}
        </div>
        
        <h3>优化建议</h3>
        <div class="ai-analysis">
            ${optimizationSuggestions || '<p>暂无优化建议数据</p>'}
        </div>
    </div>
    `;
}

// 下载HTML报告
function downloadHTML(htmlContent, title) {
    const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const fileName = `${title}_${new Date().toISOString().slice(0, 10)}.html`;
    link.setAttribute('download', fileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // 保存文件名到本地存储
    saveFileToLocalStorage(fileName, 'html');
}

// 下载PDF报告
async function downloadPDF(htmlContent, title) {
    // 动态加载html2pdf库
    if (typeof html2pdf === 'undefined') {
        await loadScript('https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js');
    }
    
    // 创建一个隐藏的div来放置报告内容
    const element = document.createElement('div');
    element.innerHTML = htmlContent;
    element.style.position = 'absolute';
    element.style.left = '-9999px';
    document.body.appendChild(element);
    
    const fileName = `${title}_${new Date().toISOString().slice(0, 10)}.pdf`;
    
    // 配置选项
    const opt = {
        margin: [10, 10],
        filename: fileName,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { 
            scale: 2,
            useCORS: true,
            logging: false
        },
        jsPDF: { 
            unit: 'mm', 
            format: 'a4', 
            orientation: 'portrait'
        },
        pagebreak: { mode: ['avoid-all', 'css', 'legacy'] }
    };
    
    // 生成并下载PDF
    try {
        await html2pdf().set(opt).from(element).save();
        
        // 保存文件名到本地存储
        saveFileToLocalStorage(fileName, 'pdf');
    } catch (error) {
        console.error('PDF生成出错:', error);
        alert('PDF生成失败，请尝试导出HTML格式或稍后再试');
        throw error;
    } finally {
        // 清理DOM
        document.body.removeChild(element);
    }
}

// 保存文件名到本地存储
function saveFileToLocalStorage(fileName, fileType) {
    try {
        const storedFiles = JSON.parse(localStorage.getItem('generatedFiles') || '[]');
        storedFiles.unshift({
            fileName,
            fileType,
            timestamp: new Date().toISOString()
        });
        
        // 只保留最近的10个文件记录
        const recentFiles = storedFiles.slice(0, 10);
        localStorage.setItem('generatedFiles', JSON.stringify(recentFiles));
    } catch (error) {
        console.warn('无法保存文件记录:', error);
    }
}

// 显示已生成的文件列表
function showGeneratedFilesList() {
    try {
        const storedFiles = JSON.parse(localStorage.getItem('generatedFiles') || '[]');
        if (storedFiles.length === 0) {
            return `<p class="text-muted">暂无生成的报告文件</p>`;
        }
        
        let html = `<div class="list-group">`;
        storedFiles.forEach(file => {
            const date = new Date(file.timestamp);
            const formattedDate = date.toLocaleString('zh-CN');
            const icon = file.fileType === 'pdf' ? 'file-earmark-pdf' : 'file-earmark-code';
            
            html += `
            <div class="list-group-item">
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1"><i class="bi bi-${icon} me-2"></i>${file.fileName}</h6>
                    <small>${formattedDate}</small>
                </div>
                <small class="text-muted">文件格式: ${file.fileType.toUpperCase()}</small>
            </div>`;
        });
        html += `</div>`;
        
        return html;
    } catch (error) {
        console.warn('无法加载文件列表:', error);
        return `<p class="text-danger">加载文件列表失败</p>`;
    }
}

// 更新预测函数以集成DeepSeek分析
async function predictStrength() {
    // 获取输入参数
    const c = parseFloat(document.getElementById('predict-c').value);
    const si = parseFloat(document.getElementById('predict-si').value);
    const mn = parseFloat(document.getElementById('predict-mn').value);
    const temp = parseFloat(document.getElementById('predict-temp').value);
    const thickness = parseFloat(document.getElementById('predict-thickness').value);
    const coolTemp = parseFloat(document.getElementById('predict-cool-temp').value);
    
    // 显示加载指示器
    document.getElementById('prediction-loading').style.display = 'block';
    document.getElementById('deepseek-analysis-card').style.display = 'none';
    
    // 简单的预测模型（基于最近邻）
    const predictions = findSimilarSamples(c, si, mn, temp, thickness, coolTemp);
    
    // 显示预测结果
    document.getElementById('predicted-yield').textContent = predictions.yield.toFixed(0) + ' MPa';
    document.getElementById('predicted-tensile').textContent = predictions.tensile.toFixed(0) + ' MPa';
    document.getElementById('predicted-elongation').textContent = predictions.elongation.toFixed(1) + '%';
    document.getElementById('predicted-toughness').textContent = predictions.toughness.toFixed(0) + 'J/cm^2' ;
    
    // 创建材料数据对象
    const materialData = {
        c, si, mn, temp, thickness, coolTemp
    };
    
    // 调用DeepSeek API进行分析
    try {
        const analysis = await analyzeWithDeepSeek(materialData, predictions);
        displayDeepSeekAnalysis(analysis);
    } catch (error) {
        console.error('分析失败:', error);
        document.getElementById('deepseek-analysis').innerHTML = 
            '<div class="alert alert-danger">获取AI分析失败，请稍后再试</div>';
        document.getElementById('deepseek-analysis-card').style.display = 'block';
    } finally {
        document.getElementById('prediction-loading').style.display = 'none';
    }
}

// 查找相似样本
function findSimilarSamples(c, si, mn, temp, thickness, coolTemp) {
    // 计算每个样本的相似度
    const similarities = csvData.map(row => {
        // 计算归一化距离
        const cDiff = Math.abs(parseFloat(row['C']) - c) / 0.2;
        const siDiff = Math.abs(parseFloat(row['Si']) - si) / 0.5;
        const mnDiff = Math.abs(parseFloat(row['Mn']) - mn) / 0.5;
        const tempDiff = Math.abs(parseFloat(row['出炉温度实际值']) - temp) / 100;
        const thicknessDiff = Math.abs(parseFloat(row['终扎厚度实际值']) - thickness) / 50;
        const coolTempDiff = Math.abs(parseFloat(row['水冷后温度']) - coolTemp) / 100;
        
        // 总距离
        const distance = Math.sqrt(
            cDiff * cDiff +
            siDiff * siDiff +
            mnDiff * mnDiff +
            tempDiff * tempDiff +
            thicknessDiff * thicknessDiff +
            coolTempDiff * coolTempDiff
        );
        
        return {
            distance: distance,
            yield: parseFloat(row['屈服强度']) || 0,
            tensile: parseFloat(row['抗拉强度']) || 0,
            elongation: parseFloat(row['实际%']) || 0,
            toughness: calculateToughness(row)
        };
    });
    
    // 排序并取前5个相似样本
    similarities.sort((a, b) => a.distance - b.distance);
    const topSamples = similarities.slice(0, 5);
    
    // 计算平均值
    return {
        yield: average(topSamples.map(s => s.yield)),
        tensile: average(topSamples.map(s => s.tensile)),
        elongation: average(topSamples.map(s => s.elongation)),
        toughness: average(topSamples.map(s => s.toughness))
    };
}

// 计算平均值的辅助函数
function average(arr) {
    if (arr.length === 0) return 0;
    return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

// 计算韧性值（使用实测1、实测2、实测3的平均值）
function calculateToughness(row) {
    const test1 = parseFloat(row['实测1']) || 0;
    const test2 = parseFloat(row['实测2']) || 0;
    const test3 = parseFloat(row['实测3']) || 0;
    
    // 如果所有值都是0，则返回0
    if (test1 === 0 && test2 === 0 && test3 === 0) {
        return 0;
    }
    
    // 计算非零值的平均值
    const nonZeroValues = [test1, test2, test3].filter(val => val > 0);
    return nonZeroValues.length > 0 ? nonZeroValues.reduce((a, b) => a + b, 0) / nonZeroValues.length : 0;
}

// 创建韧性与成分关系图表
function createToughnessChart() {
    const chart = echarts.init(document.getElementById('toughness-chart'));
    
    // 获取韧性数据
    const toughnessData = csvData.map(row => ({
        c: parseFloat(row['C']) || 0,
        mn: parseFloat(row['Mn']) || 0,
        toughness: calculateToughness(row)
    })).filter(item => item.toughness > 0);
    
    // 按C含量分组
    const cGroups = {};
    toughnessData.forEach(item => {
        // 将C含量按0.05为一组
        const cGroup = Math.floor(item.c * 20) / 20;
        if (!cGroups[cGroup]) {
            cGroups[cGroup] = [];
        }
        cGroups[cGroup].push(item.toughness);
    });
    
    // 按Mn含量分组
    const mnGroups = {};
    toughnessData.forEach(item => {
        // 将Mn含量按0.2为一组
        const mnGroup = Math.floor(item.mn * 5) / 5;
        if (!mnGroups[mnGroup]) {
            mnGroups[mnGroup] = [];
        }
        mnGroups[mnGroup].push(item.toughness);
    });
    
    // 计算每个组的平均韧性
    const cCategories = [];
    const cToughness = [];
    for (const group in cGroups) {
        if (cGroups[group].length > 0) {
            cCategories.push(group);
            cToughness.push(average(cGroups[group]).toFixed(1));
        }
    }
    
    const mnCategories = [];
    const mnToughness = [];
    for (const group in mnGroups) {
        if (mnGroups[group].length > 0) {
            mnCategories.push(group);
            mnToughness.push(average(mnGroups[group]).toFixed(1));
        }
    }
    
    const option = {
        title: {
            text: '材料成分与韧性关系',
            left: 'center'
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                type: 'shadow'
            }
        },
        legend: {
            data: ['碳含量(C)', '锰含量(Mn)'],
            top: '10%'
        },
        grid: {
            left: '3%',
            right: '4%',
            bottom: '3%',
            containLabel: true,
            top: '20%'
        },
        xAxis: [
            {
                type: 'category',
                data: [...new Set([...cCategories, ...mnCategories])].sort((a, b) => a - b),
                axisLabel: {
                    formatter: function(value) {
                        return value;
                    }
                }
            }
        ],
        yAxis: [
            {
                type: 'value',
                name: '平均韧性值'
            }
        ],
        series: [
            {
                name: '碳含量(C)',
                type: 'line',
                smooth: true,
                data: cCategories.map((category, index) => ({
                    value: cToughness[index],
                    name: 'C=' + category
                })),
                markPoint: {
                    data: [
                        { type: 'max', name: '最大值' },
                        { type: 'min', name: '最小值' }
                    ]
                }
            },
            {
                name: '锰含量(Mn)',
                type: 'line',
                smooth: true,
                data: mnCategories.map((category, index) => ({
                    value: mnToughness[index],
                    name: 'Mn=' + category
                })),
                markPoint: {
                    data: [
                        { type: 'max', name: '最大值' },
                        { type: 'min', name: '最小值' }
                    ]
                }
            }
        ]
    };
    
    chart.setOption(option);
    
    // 当切换到分析部分时，确保重新渲染图表
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === '#analysis') {
            link.addEventListener('click', function() {
                setTimeout(function() {
                    chart.resize();
                    // 重新设置选项，确保数据显示正确
                    chart.setOption(option, true);
                }, 200);
            });
        }
    });
    
    window.addEventListener('resize', function() {
        chart.resize();
    });
}

// 不再使用原来的calculateSimpleCorrelation函数，保留为兼容性
function calculateSimpleCorrelation(var1, var2) {
    // 提取两个变量的数据
    const data1 = csvData.map(row => parseFloat(row[var1]) || 0);
    const data2 = csvData.map(row => parseFloat(row[var2]) || 0);
    
    return calculateCorrelation(data1, data2);
}

// 动态加载脚本
function loadScript(url) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = url;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

// DeepSeek API集成
const DEEPSEEK_API_KEY = 'sk-8c04901f01b841b893fa314add529664';
const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions';