/**
 * 此脚本用于运行所有测试并生成测试报告
 */

// 直接运行npm test命令并带上覆盖率参数
const { exec } = require('child_process');

console.log('开始运行测试并生成覆盖率报告...');

exec('npm test -- --coverage', (error, stdout, stderr) => {
  if (error) {
    console.error(`测试执行出错: ${error.message}`);
    return;
  }
  
  if (stderr) {
    console.error(`测试错误输出: ${stderr}`);
  }
  
  console.log(stdout);
  console.log('测试完成，覆盖率报告已生成。');
}); 