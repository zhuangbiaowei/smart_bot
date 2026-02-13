# SmartBot Skill System 实施计划

## 概述

根据架构设计文档 `docs/skill_system_architecture.md`，制定以下分阶段实施计划。

---

## 当前状态分析

### 已有组件
- ✅ `lib/smart_bot/skill.rb` - 基础 Skill DSL
- ✅ `lib/smart_bot/skill_adapters/unified_skill_loader.rb` - 统一加载器
- ✅ `lib/smart_bot/skill_adapters/claude_skill_adapter.rb` - Claude 格式适配
- ✅ `lib/smart_bot/skill_routing/orchestrator.rb` - 基础委派
- ✅ `lib/smart_bot/agent/context.rb` - Skills 上下文

### 主要缺口
- ❌ 统一元数据模型（SkillPackage, SkillMetadata）
- ❌ 路由系统（Router, Scorer）
- ❌ 执行系统（Executor, Sandbox）
- ❌ 回退和修复机制
- ❌ 渐进加载实现

---

## 实施路线图

### Phase 1: 基础架构（Week 1-2）

**目标**: 建立核心数据模型和加载机制

#### 1.1 创建目录结构
```
lib/smart_bot/skill_system/
├── core/
│   ├── skill_package.rb
│   ├── metadata.rb
│   ├── loader.rb
│   └── registry.rb
├── routing/
│   ├── router.rb
│   ├── scorer.rb
│   └── activation_plan.rb
├── execution/
│   ├── executor.rb
│   ├── sandbox.rb
│   ├── fallback.rb
│   └── repair_loop.rb
├── adapters/
│   ├── base.rb
│   ├── ruby_adapter.rb
│   └── markdown_adapter.rb
└── skill_system.rb          # 主入口
```

#### 1.2 实现 Core Layer
- [ ] `SkillPackage` - 统一 Skill 表示
- [ ] `SkillMetadata` - 元数据解析（frontmatter + skill.yaml）
- [ ] `UnifiedLoader` - 重构现有加载器
- [ ] `SkillRegistry` - 带索引的注册表

**验收标准**:
- 能正确加载所有现有 skills
- 元数据解析正确
- 注册表索引工作正常

---

### Phase 2: 路由系统（Week 2-3）

**目标**: 实现智能 Skill 路由

#### 2.1 实现 Routing Layer
- [ ] `Router` - 两阶段路由（召回 + 打分）
- [ ] `SkillScorer` - 打分算法
- [ ] `ActivationPlan` - 激活计划构建

#### 2.2 集成到现有系统
- [ ] 更新 `Agent::Context::Skills` 使用新 Registry
- [ ] 添加路由事件追踪

**验收标准**:
- 路由能正确匹配 skills
- 打分算法符合 spec
- 激活计划构建正确

---

### Phase 3: 执行系统（Week 3-4）

**目标**: 实现安全执行和失败处理

#### 3.1 实现 Execution Layer
- [ ] `SkillExecutor` - 执行协调器
- [ ] `Sandbox` - 权限控制和沙箱
- [ ] `FallbackStateMachine` - 回退状态机
- [ ] `RepairLoop` - 自我修复（可选，Phase 3.5）

#### 3.2 适配器重构
- [ ] 重构 `RubySkillAdapter`
- [ ] 重构 `MarkdownSkillAdapter`
- [ ] 统一适配器接口

**验收标准**:
- Skills 能正确执行
- 权限检查工作
- 回退机制正常

---

### Phase 4: 集成与配置（Week 4-5）

**目标**: 集成到 Agent Loop 和 CLI

#### 4.1 Agent 集成
- [ ] 更新 `Agent::Loop` 使用新路由系统
- [ ] 添加 Skill 自动触发
- [ ] 保留向后兼容

#### 4.2 CLI 增强
- [ ] `skill install` - 从 git/registry 安装
- [ ] `skill list` - 列出现有 skills
- [ ] `skill info` - 查看 skill 详情
- [ ] `skill run` - 直接运行 skill

#### 4.3 配置系统
- [ ] 添加 `config/smart_bot.yml` 配置
- [ ] 支持路由参数调整
- [ ] 支持执行策略配置

**验收标准**:
- Agent 能自动路由到 skills
- CLI 命令工作正常
- 配置可热重载

---

### Phase 5: 测试与优化（Week 5-6）

**目标**: 确保质量和性能

#### 5.1 测试
- [ ] 单元测试（所有核心类）
- [ ] 集成测试（端到端场景）
- [ ] 性能测试（路由性能）

#### 5.2 文档
- [ ] API 文档
- [ ] Skill 开发指南
- [ ] 迁移指南

#### 5.3 优化
- [ ] 性能优化（索引、缓存）
- [ ] 错误处理改进
- [ ] 日志和观测性

---

## 优先级矩阵

| 组件 | 优先级 | 依赖 | 预计工时 |
|------|--------|------|----------|
| SkillPackage | P0 | 无 | 4h |
| SkillMetadata | P0 | 无 | 4h |
| UnifiedLoader | P0 | SkillPackage | 6h |
| SkillRegistry | P0 | SkillPackage | 6h |
| Router | P0 | Registry | 8h |
| Scorer | P0 | Router | 6h |
| ActivationPlan | P1 | Router | 4h |
| SkillExecutor | P1 | Router | 8h |
| Sandbox | P1 | Executor | 6h |
| FallbackStateMachine | P1 | Executor | 6h |
| RepairLoop | P2 | Fallback | 8h |
| CLI 命令 | P2 | 全部 | 8h |
| 测试 | P1 | 全部 | 16h |

---

## 风险与缓解

### 风险 1: 向后兼容性问题
**影响**: 现有 skills 可能无法工作
**缓解**: 
- 保留旧的 Skill 注册方式
- 提供 LegacyAdapter
- 渐进迁移

### 风险 2: 性能下降
**影响**: 路由增加延迟
**缓解**:
- 实现索引缓存
- 异步加载
- 性能基准测试

### 风险 3: 复杂度增加
**影响**: 代码难以维护
**缓解**:
- 清晰的接口契约
- 完善的测试覆盖
- 详细的文档

---

## 最小可行产品（MVP）

如果资源有限，优先实现 MVP：

1. **Core Layer**: SkillPackage + SkillMetadata + Registry
2. **Simple Router**: 仅支持 Hard Trigger 和 Rule Matching
3. **Basic Executor**: 直接执行，无沙箱
4. **Backward Compatibility**: 保持现有功能

MVP 可在 **1 周**内完成，提供基础路由能力。

---

## 下一步行动

1. **立即开始**: Phase 1.1 - 创建目录结构
2. **本周完成**: Phase 1.2 - Core Layer 实现
3. **下周开始**: Phase 2 - Routing Layer

建议按 Phase 顺序实施，每个 Phase 完成后进行代码审查和测试。
