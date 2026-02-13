---
name: test_suite
description: SmartBot Skill系统能力测试套件。用于验证路由、执行、错误处理等核心能力。
triggers:
  - test_suite
  - 测试套件
  - skill系统测试
  - run skill tests
anti_triggers:
  - test_router
  - test_executor
  - test_error_handling
  - test_parameters
  - test_edge_cases
  - test_fallback
cost_hint: low
---

# Test Suite - Skill能力测试套件

这是一个元测试skill，用于验证SmartBot的skill系统能力。

## 测试维度

### 1. 路由能力测试
- 触发词精确匹配
- 触发词模糊匹配
- 语义召回
- 硬触发（$skill_name）
- 优先级排序

### 2. 执行能力测试
- instruction类型执行
- script类型执行
- openclaw类型执行
- 参数传递
- 上下文传递

### 3. 错误处理测试
- 可重试错误
- 致命错误
- Fallback链
- 修复循环

### 4. 边界测试
- 空输入处理
- 超长输入处理
- 特殊字符处理

## 使用方法

```bash
# 运行完整测试套件
smart_bot agent -m "test_suite run all"

# 运行路由测试
smart_bot agent -m "test_suite run routing"

# 运行错误处理测试
smart_bot agent -m "test_suite run error_handling"
```
