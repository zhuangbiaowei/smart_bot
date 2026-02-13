# SmartBot Skill 系统测试指南

## 测试Skill概览

本测试套件包含以下测试Skill，用于验证SmartBot的skill系统能力：

| Skill名称 | 类型 | 测试目标 |
|-----------|------|----------|
| test_suite | instruction | 元测试套件，协调所有测试 |
| test_router | instruction | 路由能力测试 |
| test_executor | instruction | 执行能力测试 |
| test_error_handling | instruction | 错误处理和试错机制测试 |
| test_fallback_a | instruction | Fallback主技能（故意失败） |
| test_fallback_b | instruction | Fallback备用技能 |
| test_parameters | instruction | 参数处理测试 |
| test_edge_cases | instruction | 边界条件测试 |
| test_script_type | script | 脚本类型执行测试 |

## 快速开始

### 一键运行所有测试

```bash
./scripts/run_skill_tests.sh
```

### 手动测试

```bash
# 进入项目目录
cd /home/mlf/smart_ai/smart_bot

# 运行单个测试
smart_bot agent -m "test_router 路由测试"
```

---

## 详细测试场景

### 1. 路由能力测试 (test_router)

**目的**: 验证路由系统的触发词匹配、语义召回和优先级排序能力。

#### 1.1 触发词精确匹配

```bash
smart_bot agent -m "test_router 基本匹配测试"
```

**预期结果**:
- 被路由到 `test_router` skill
- 返回包含 "router" 或 "路由" 的输出

#### 1.2 硬触发测试

```bash
smart_bot agent -m "\$test_router 强制路由"
```

**预期结果**:
- 使用 `$` 前缀强制路由到指定skill
- 忽略其他可能的匹配

#### 1.3 中文触发词测试

```bash
smart_bot agent -m "路由测试功能验证"
```

**预期结果**:
- 匹配到中文触发词 "路由测试"
- 正确路由到 `test_router`

#### 1.4 反触发测试

```bash
smart_bot agent -m "下载test_router相关资源"
```

**预期结果**:
- 由于包含反触发词 "下载"，优先级降低
- 可能被其他skill匹配或正常处理

---

### 2. 执行能力测试 (test_executor)

**目的**: 验证不同类型skill的执行能力。

#### 2.1 instruction类型执行

```bash
smart_bot agent -m "test_executor 基本执行测试"
```

**预期结果**:
- 成功执行instruction类型skill
- 返回执行状态和相关参数信息

#### 2.2 script类型执行

```bash
smart_bot agent -m "test_script hello world"
```

**预期结果**:
- 执行 `scripts/echo.py` 脚本
- 返回JSON格式的执行结果

#### 2.3 上下文传递测试

```bash
smart_bot agent -m "test_executor 上下文测试" -l deepseek
```

**预期结果**:
- LLM名称正确传递到上下文
- 输出中包含模型信息

---

### 3. 参数处理测试 (test_parameters)

**目的**: 验证参数提取、传递和类型转换能力。

#### 3.1 基本参数

```bash
smart_bot agent -m "test_parameters name=test value=123"
```

**预期结果**:
- 参数被正确识别
- 输出显示 `name: test, value: 123`

#### 3.2 URL参数

```bash
smart_bot agent -m "test_parameters url=https://example.com/test?query=value"
```

**预期结果**:
- URL完整保留，不被截断
- 特殊字符（?、=）正确处理

#### 3.3 中文参数

```bash
smart_bot agent -m "test_parameters 姓名=张三 城市=北京"
```

**预期结果**:
- 中文字符正确编码
- 参数值完整保留

#### 3.4 JSON参数

```bash
smart_bot agent -m 'test_parameters config={"debug":true,"port":3000}'
```

**预期结果**:
- JSON格式正确解析
- 不破坏JSON结构

---

### 4. 错误处理测试 (test_error_handling)

**目的**: 验证重试机制、错误分类和修复循环。

#### 4.1 可重试错误

```bash
smart_bot agent -m "test_error_handling retryable_error"
```

**预期行为**:
```
尝试1: 失败
尝试2: 重试
达到最大重试次数后触发fallback
```

#### 4.2 致命错误

```bash
smart_bot agent -m "test_error_handling fatal_error permission denied"
```

**预期行为**:
```
尝试1: 失败 (permission denied)
跳过重试（致命错误）
直接触发fallback
```

#### 4.3 修复循环触发

```bash
smart_bot agent -m "test_error_handling repair_test parameter error"
```

**预期行为**:
```
检测到可修复错误
启动RepairLoop
应用补丁后重试
```

---

### 5. Fallback链测试 (test_fallback_a → test_fallback_b)

**目的**: 验证当主技能失败时，fallback机制是否正常工作。

#### 5.1 完整fallback链测试

```bash
smart_bot agent -m "test_fallback 测试fallback机制"
```

**预期行为**:
```
1. 路由到 test_fallback_a（主技能）
2. test_fallback_a 执行失败（故意设计）
3. 重试 test_fallback_a（最多1次）
4. 触发fallback到 test_fallback_b
5. test_fallback_b 执行成功
6. 返回最终成功结果
```

**验证点**:
- [ ] test_fallback_a 被首先选中
- [ ] test_fallback_a 失败后触发重试
- [ ] 重试失败后触发fallback
- [ ] test_fallback_b 被成功执行
- [ ] 返回成功结果而非错误

---

### 6. 边界条件测试 (test_edge_cases)

**目的**: 验证异常输入的处理能力。

#### 6.1 空输入

```bash
smart_bot agent -m "test_edge_cases"
```

**预期结果**:
- 正常执行，不崩溃
- 返回无参数提示

#### 6.2 特殊字符

```bash
smart_bot agent -m 'test_edge_cases special: <>&"'\''/@#$%^*()[]{}|'
```

**预期结果**:
- 特殊字符被正确处理或转义
- 不导致解析错误

#### 6.3 Unicode/Emoji

```bash
smart_bot agent -m "test_edge_cases emoji: 😀🎉🚀 中文：测试 日本語：テスト"
```

**预期结果**:
- Unicode字符正确编码
- Emoji正确显示

#### 6.4 注入模式测试

```bash
smart_bot agent -m 'test_edge_cases injection: ${dangerous} {{template}} <script>alert(1)</script>'
```

**预期结果**:
- 不执行任何模板代码
- 不执行任何脚本代码
- 安全处理输入

---

### 7. 单元测试

运行RSpec单元测试：

```bash
# 运行所有skill系统测试
bundle exec rspec spec/skill_system/

# 单独运行路由测试
bundle exec rspec spec/skill_system/routing/

# 单独运行执行测试
bundle exec rspec spec/skill_system/execution/

# 单独运行核心测试
bundle exec rspec spec/skill_system/core/
```

---

## 错误分类参考

根据 `fallback.rb` 实现，以下错误类型区别处理：

### 不可重试错误（致命错误）

```ruby
non_retryable = [
  /permission denied/i,
  /not found/i,
  /invalid.*format/i,
  /capability.*not.*available/i
]
```

这些错误会**跳过重试**，直接触发fallback。

### 可重试错误

其他所有错误都被认为是可重试的，最多重试1次（`MAX_RETRIES = 1`）。

---

## 修复循环触发条件

根据 `repair_loop.rb` 实现，以下错误类型会触发修复循环：

```ruby
repairable_patterns = [
  /parameter/i,
  /missing.*field/i,
  /not found/i,
  /path.*error/i,
  /template/i,
  /reference/i
]
```

---

## 测试输出解读

### 成功输出示例

```
✅ Router Test Results:
   - Matched skill: test_router
   - Match type: rule
   - Triggers hit: ["test_router"]
```

### 失败输出示例

```
⚠️ Skill execution failed: test_fallback_a
   Error: Intentional failure for fallback testing
   FALLBACK: Trying test_fallback_b...
```

### 最终失败示例

```
❌ All skills failed. Last error: All fallback options exhausted
```

---

## 常见问题排查

### Q: 路由不到预期的skill

**检查项**:
1. skill是否在正确目录（`skills/`）
2. SKILL.md frontmatter格式是否正确
3. 触发词是否包含在输入中
4. 是否有其他skill优先级更高

### Q: 执行失败但无明确错误

**检查项**:
1. 查看日志输出
2. 检查skill类型是否与执行方式匹配
3. 验证参数传递是否正确

### Q: fallback没有触发

**检查项**:
1. 主skill是否真的失败了
2. 是否有fallback_chain定义
3. 错误是否被判定为可重试

---

## 测试套件文件结构

```
skills/
├── test_suite/
│   └── SKILL.md              # 元测试套件
├── test_router/
│   └── SKILL.md              # 路由测试
├── test_executor/
│   └── SKILL.md              # 执行测试
├── test_error_handling/
│   └── SKILL.md              # 错误处理测试
├── test_fallback_a/
│   └── SKILL.md              # Fallback主技能
├── test_fallback_b/
│   └── SKILL.md              # Fallback备用技能
├── test_parameters/
│   └── SKILL.md              # 参数测试
├── test_edge_cases/
│   └── SKILL.md              # 边界测试
└── test_script_type/
    ├── SKILL.md              # 脚本测试
    ├── skill.yaml            # 脚本配置
    └── scripts/
        └── echo.py           # 测试脚本

scripts/
└── run_skill_tests.sh        # 一键测试脚本

spec/
└── skill_system/
    ├── routing/              # 路由单元测试
    ├── execution/            # 执行单元测试
    └── core/                 # 核心单元测试
```
