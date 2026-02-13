---
name: test_fallback_a
description: Fallback测试主技能。此skill被设计为失败，以触发fallback到test_fallback_b。
triggers:
  - test_fallback
  - fallback测试
cost_hint: low
---

# Fallback Test - Primary Skill (A)

这是fallback链测试的主技能。**故意设计为总是失败**，用于测试fallback机制。

## 行为

此skill收到任何请求都会返回错误：

```
❌ test_fallback_a: Intentional failure for fallback testing
```

## Fallback链

当此skill失败后，系统应该尝试：
1. 重试 test_fallback_a (最多1次)
2. Fallback到 test_fallback_b
3. Fallback到 generic_tools

## 与 test_fallback_b 的关系

- test_fallback_a: 主技能，总是失败
- test_fallback_b: 备用技能，应该成功

两个skill共享触发词 `test_fallback`，用于测试多技能匹配和fallback。

## CLI测试命令

```bash
# 应该先尝试A，失败后fallback到B
smart_bot agent -m "test_fallback 测试fallback机制"
```

## 预期输出

```
⚠️ Skill test_fallback_a failed: Intentional failure for fallback testing
 FALLBACK: Trying test_fallback_b...
✅ test_fallback_b: Fallback successful!
```
