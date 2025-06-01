// 导入Jest DOM扩展以提供额外的匹配器
require('@testing-library/jest-dom');

// 模拟localStorage
class LocalStorageMock {
  constructor() {
    this.store = {};
  }

  clear() {
    this.store = {};
  }

  getItem(key) {
    return this.store[key] || null;
  }

  setItem(key, value) {
    this.store[key] = String(value);
  }

  removeItem(key) {
    delete this.store[key];
  }
}

// 设置全局localStorage模拟
global.localStorage = new LocalStorageMock();

// 处理一些全局变量缺失的问题
global.matchMedia = global.matchMedia || function() {
  return {
    matches: false,
    addListener: function() {},
    removeListener: function() {}
  };
}; 