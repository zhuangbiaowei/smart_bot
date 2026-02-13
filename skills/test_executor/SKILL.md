---
name: test_executor
description: 执行能力测试skill。验证不同类型skill的执行和参数传递。
triggers:
  - test_executor
  - 执行测试
  - executor test
cost_hint: low
---

# Executor Test Skill

用于测试SmartBot的执行系统能力。

## 测试场景

### 场景1: 基本执行
验证skill能够被正确执行并返回结果。

### 场景2: 参数传递
验证参数能够正确从用户输入传递到skill。

### 场景3: 上下文传递
验证LLM上下文能够正确传递。

### 场景4: 类型适配
验证instruction类型skill能够被正确解析和执行。

## 预期输出

```
✅ Executor Test Results:
   - Execution type: instruction
   - Parameters received: { task: "<user_input>" }
   - Context received: { llm: "<llm_name>" }
   - Status: success
```

## CLI测试命令

```bash
# 基本执行测试
smart_bot agent -m "test_executor 基本执行"

# 带参数测试
smart_bot agent -m "test_executor 参数测试 param1=value1 param2=value2"

# 验证上下文传递
smart_bot agent -m "test_executor 上下文测试" -l deepseek
```
