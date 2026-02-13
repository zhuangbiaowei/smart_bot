---
name: test_router
description: 路由能力测试skill。验证触发词匹配、语义召回和优先级排序。
triggers:
  - test_router
  - 路由测试
  - routing test
anti_triggers:
  - 下载
  - download
cost_hint: low
---

# Router Test Skill

用于测试SmartBot的路由系统能力。

## 测试场景

### 场景1: 触发词精确匹配
输入包含 `test_router` 或 `路由测试` 时应被路由到此skill。

### 场景2: 语义匹配
输入 "测试路由功能" 或 "检查路由系统" 等语义相关内容应被召回。

### 场景3: 硬触发
使用 `$test_router` 应强制路由到此skill。

### 场景4: 反触发测试
输入 "下载test_router" 或 "test_router download" 时，由于包含反触发词，应降低匹配分数。

## 预期行为

当被路由到此skill时，返回JSON格式的路由测试结果：

```json
{
  "test": "router",
  "status": "success",
  "matched_triggers": [],
  "match_type": "rule|semantic|forced"
}
```

## CLI测试命令

```bash
# 测试触发词匹配
smart_bot agent -m "test_router 基本测试"

# 测试中文触发词
smart_bot agent -m "路由测试功能验证"

# 测试硬触发
smart_bot agent -m "$test_router 强制路由"

# 测试反触发（应该降低优先级或匹配到其他skill）
smart_bot agent -m "下载test_router相关资源"
```
