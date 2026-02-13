---
name: test_parameters
description: 参数处理测试skill。验证参数提取、传递和类型转换。
triggers:
  - test_parameters
  - 参数测试
  - parameter test
cost_hint: low
---

# Parameters Test Skill

用于测试SmartBot的参数处理能力。

## 测试场景

### 场景1: 基本参数提取
输入: `test_parameters name=John age=25`
期望: 参数被正确识别和传递

### 场景2: URL参数
输入: `test_parameters url=https://example.com/path?query=value`
期望: URL完整保留，不被截断

### 场景3: 中文参数
输入: `test_parameters 姓名=张三 城市=北京`
期望: 中文参数正确处理

### 场景4: 特殊字符参数
输入: `test_parameters data={"key":"value"} path=/tmp/test.log`
期望: JSON和路径正确处理

### 场景5: 多值参数
输入: `test_parameters tags=python tags=ruby tags=go`
期望: 多值参数以数组形式传递

### 场景6: 空参数
输入: `test_parameters`
期望: 无参数时正常执行

## 预期输出

```
📋 Parameters Test Results:
═══════════════════════════════════════
Raw input: <原始输入>
Extracted parameters:
  - name: John
  - age: 25
  - url: https://example.com/path?query=value
Parameter count: 4
Status: ✅ PASS
═══════════════════════════════════════
```

## CLI测试命令

```bash
# 基本参数测试
smart_bot agent -m "test_parameters name=test value=123"

# URL参数测试
smart_bot agent -m "test_parameters url=https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 中文参数测试
smart_bot agent -m "test_parameters 姓名=测试 用户=用户名"

# JSON参数测试
smart_bot agent -m 'test_parameters config={"debug":true,"port":3000}'

# 多值参数测试
smart_bot agent -m "test_parameters items=apple items=banana items=orange"
```
