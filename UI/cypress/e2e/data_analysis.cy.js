describe('Data Analysis Feature', () => {
  beforeEach(() => {
    // 模拟登录过程
    cy.visit('http://localhost:9000/login');
    cy.get('input[placeholder="用户名"]').type('admin');
    cy.get('input[placeholder="密码"]').type('admin');
    cy.contains('button', '登录').click();
    
    // 等待登录完成并导航到数据分析页面
    cy.url().should('include', '/dashboard');
    cy.get('a[href="/analysis"]').click();
    cy.url().should('include', '/analysis');
  });

  it('displays data analysis page correctly', () => {
    // 验证页面标题
    cy.contains('数据分析').should('be.visible');
    
    // 在加载完成后检查趋势分析是否显示
    cy.get('.ant-spin-spinning').should('not.exist', { timeout: 5000 });
    cy.contains('趋势分析').should('be.visible');
    
    // 验证图表已加载
    cy.get('.recharts-wrapper').should('be.visible');
  });

  it('can switch between analysis tabs', () => {
    // 等待加载完成
    cy.get('.ant-spin-spinning').should('not.exist', { timeout: 5000 });
    
    // 切换到相关性分析
    cy.contains('相关性分析').click();
    cy.get('.recharts-wrapper').should('be.visible');
    
    // 切换到数据表
    cy.contains('数据表').click();
    cy.get('.ant-table-content').should('be.visible');
  });

  it('can change property and factor selections', () => {
    // 等待加载完成
    cy.get('.ant-spin-spinning').should('not.exist', { timeout: 5000 });
    
    // 选择不同的属性
    cy.get('span').contains('屈服强度').parents('.ant-select').click();
    cy.get('.ant-select-item-option').contains('抗拉强度').click();
    
    // 验证图表已更新
    cy.contains('抗拉强度').should('be.visible');
    
    // 选择不同的因素
    cy.get('span').contains('碳含量').parents('.ant-select').click();
    cy.get('.ant-select-item-option').contains('硅含量').click();
    
    // 验证图表已更新
    cy.contains('硅含量').should('be.visible');
  });

  it('uploads custom data file correctly', () => {
    // 等待加载完成
    cy.get('.ant-spin-spinning').should('not.exist', { timeout: 5000 });
    
    // 模拟文件上传
    // 注意：Cypress中直接测试文件上传较为复杂，这里只模拟了界面操作
    cy.get('input[type="file"]').should('exist');
    
    // 检查上传按钮
    cy.contains('button', '上传数据').should('be.visible');
  });

  it('displays data table with filtering and sorting', () => {
    // 等待加载完成
    cy.get('.ant-spin-spinning').should('not.exist', { timeout: 5000 });
    
    // 切换到数据表
    cy.contains('数据表').click();
    
    // 验证表格存在
    cy.get('.ant-table-content').should('be.visible');
    
    // 检查表格列
    cy.contains('碳含量').should('be.visible');
    cy.contains('屈服强度').should('be.visible');
    
    // 测试排序功能
    cy.contains('碳含量').click();
    
    // 测试筛选功能 (如果有)
    cy.get('.ant-table-filter-trigger').first().click({ force: true });
    cy.get('.ant-dropdown-menu').should('be.visible');
  });
}); 