---
name: test_script_type
description: 脚本类型执行测试skill。验证script类型skill的执行能力。
triggers:
  - test_script
  - 脚本测试
  - script test
cost_hint: low
---

# Script Type Test Skill

用于测试SmartBot对script类型skill的执行能力。

## 脚本文件

此skill包含可执行脚本 `scripts/echo.py`，用于验证：
- 脚本发现和执行
- 参数传递给脚本
- 脚本输出捕获
- 错误处理

## 脚本类型Skill执行流程

```
skill.yaml + SKILL.md
    ↓
MarkdownSkillAdapter.load_full_skill()
    ↓
Sandbox.invoke_script_skill()
    ↓
execute_script(script_path, parameters)
    ↓
ExecutionResult
```

## 预期输出

当执行 `scripts/echo.py` 时，返回：

```json
{
  "success": true,
  "output": "Echo: <用户输入>",
  "execution_time": "<耗时>",
  "script": "echo.py"
}
```

## CLI测试命令

```bash
# 测试脚本执行
smart_bot agent -m "test_script hello world"

# 测试脚本参数传递
smart_bot agent -m "test_script --verbose param1=value1"
```
