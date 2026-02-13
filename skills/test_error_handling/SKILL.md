---
name: test_error_handling
description: 错误处理测试skill。验证重试机制、fallback链和错误分类。
triggers:
  - test_error_handling
  - 错误处理测试
  - error test
  - retry test
cost_hint: low
---

# Error Handling Test Skill

用于测试SmartBot的错误处理和试错机制。

## 测试场景

### 场景1: 可重试错误模拟
通过特定参数触发可重试错误，验证系统是否会重试。

触发词: `retryable_error`

### 场景2: 致命错误模拟
通过特定参数触发致命错误，验证系统是否会跳过重试直接fallback。

触发词: `fatal_error`, `permission denied`, `not found`

### 场景3: Fallback链测试
当主skill失败时，验证是否正确触发fallback链。

### 场景4: 修复循环测试
触发需要修复的错误，验证RepairLoop是否工作。

触发词: `repair_test`, `parameter error`

## 错误类型定义

根据 `fallback.rb:121-128`，以下错误被认为是不可重试的：
- `permission denied`
- `not found`
- `invalid.*format`
- `capability.*not.*available`

其他错误被认为是可重试的，最多重试1次 (`MAX_RETRIES = 1`)。

## 预期行为

### 可重试错误
```
尝试1: 失败 (retryable error)
尝试2: 重试中...
结果: 达到最大重试次数，触发fallback
```

### 致命错误
```
尝试1: 失败 (fatal error - permission denied)
结果: 跳过重试，直接触发fallback
```

## CLI测试命令

```bash
# 测试可重试错误
smart_bot agent -m "test_error_handling retryable_error"

# 测试致命错误
smart_bot agent -m "test_error_handling fatal_error permission denied"

# 测试参数错误（应触发修复循环）
smart_bot agent -m "test_error_handling repair_test parameter error"
```
