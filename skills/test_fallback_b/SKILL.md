---
name: test_fallback_b
description: Fallback测试备用技能。作为test_fallback_a的fallback目标。
triggers:
  - test_fallback
  - fallback备用
cost_hint: low
---

# Fallback Test - Secondary Skill (B)

这是fallback链测试的备用技能。当test_fallback_a失败时，系统应该fallback到此skill。

## 行为

此skill会成功执行并返回：

```
✅ test_fallback_b: Fallback successful!
   - Previous skill: test_fallback_a
   - Fallback reason: primary skill failed
   - Test result: PASS
```

## Fallback链位置

1. test_fallback_a (失败) →
2. **test_fallback_b (当前，成功)** →
3. generic_tools (不需要)

## CLI测试命令

```bash
# 整个fallback链测试
smart_bot agent -m "test_fallback 测试fallback机制"
```

## 验证点

- [ ] test_fallback_a 被首先选中
- [ ] test_fallback_a 失败后触发重试
- [ ] 重试失败后触发fallback
- [ ] test_fallback_b 被成功执行
- [ ] 返回最终成功结果
