# SmartBot Skill System

## 概述

SmartBot Skill System 是一个统一的 Skill 管理和执行框架，实现了智能路由、安全执行和失败回退机制。

## 架构

```
┌─────────────────────────────────────────┐
│              CLI / API                  │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│            Routing Layer                │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
│  │  Recall │ │  Score  │ │   Plan   │  │
│  └─────────┘ └─────────┘ └──────────┘  │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│           Execution Layer               │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
│  │ Sandbox │ │ Fallback│ │  Repair  │  │
│  └─────────┘ └─────────┘ └──────────┘  │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│              Core Layer                 │
│  ┌─────────┐ ┌─────────┐ ┌──────────┐  │
│  │ Package │ │Registry │ │  Loader  │  │
│  └─────────┘ └─────────┘ └──────────┘  │
└─────────────────────────────────────────┘
```

## 快速开始

### 加载 Skills

```ruby
require "smart_bot/skill_system"

# 加载所有 skills
SmartBot::SkillSystem.load_all

# 查看统计
stats = SmartBot::SkillSystem.registry.stats
# => { total: 5, available: 4, always: 1, by_type: {...} }
```

### 路由查询

```ruby
# 路由用户查询到合适的 skills
plan = SmartBot::SkillSystem.route("What's the weather today?")

if plan.empty?
  puts "No matching skills found"
else
  puts "Primary skill: #{plan.primary_skill.name}"
  puts "All skills: #{plan.skills.map(&:name).join(', ')}"
end
```

### 执行 Skill

```ruby
# 直接运行（路由 + 执行）
result = SmartBot::SkillSystem.run("$weather Shanghai")

if result.success?
  puts result.value
else
  puts "Error: #{result.error}"
end
```

## Core Layer

### SkillPackage

统一的 Skill 表示，支持多种格式：

```ruby
package = SmartBot::SkillSystem::SkillPackage.new(
  name: "my_skill",
  source_path: "/path/to/skill",
  metadata: metadata,
  type: :instruction  # :instruction, :script, :ruby_native
)

# 检查触发器
package.matches_trigger?("This is a test")  # => true/false
package.matches_anti_trigger?("Exclude this")  # => true/false

# 渐进加载
package.load_full_content  # 加载 SKILL.md 内容
package.scripts            # 发现 scripts/ 目录
package.references         # 发现 references/ 目录
```

### SkillMetadata

元数据模型，解析 skill.yaml 和 SKILL.md frontmatter：

```ruby
# 从 skill.yaml 解析
metadata = SmartBot::SkillSystem::SkillMetadata.from_skill_yaml(yaml_hash)

# 从 frontmatter 解析
metadata = SmartBot::SkillSystem::SkillMetadata.from_frontmatter(frontmatter_hash)

# 检查可用性
metadata.available?  # 检查 prerequisites

# 成本惩罚（用于路由打分）
metadata.cost_penalty  # => 0.0, -0.05, or -0.10
```

### SkillRegistry

带索引的 Skill 注册表：

```ruby
registry = SmartBot::SkillSystem::SkillRegistry.new

# 注册 skill
registry.register(package)

# 查找
registry.find("skill_name")           # 按名称
registry.find_by_trigger("query")     # 按触发器
registry.list_available               # 可用的 skills
registry.list_always                  # always_load 的 skills

# 统计
registry.stats  # => { total: 5, available: 4, ... }
```

### UnifiedLoader

统一加载器，支持多种格式：

```ruby
loader = SmartBot::SkillSystem::UnifiedLoader.new(
  workspace: "~/.smart_bot/workspace"
)

# 加载所有 skills
skills = loader.load_all

# 加载特定 skill
skill = loader.load_skill("weather")

# 从目录加载
skills = loader.load_from_directory("/path/to/skills")
```

## Routing Layer

### Router

两阶段路由器：

```ruby
router = SmartBot::SkillSystem::Router.new(registry: registry)

plan = router.route(
  query: "What's the weather?",
  context: { user: "john" },
  history: [],
  stats: { "weather" => { successes: 10, total: 12 } }
)
```

路由阶段：
1. **Hard Triggers**: `$skill_name`, `使用 skill_name skill`
2. **Rule Matching**: 基于 triggers 关键词
3. **Semantic Search**: 基于描述相似度

### SkillScorer

加权打分算法：

```ruby
scorer = SmartBot::SkillSystem::SkillScorer.new

score = scorer.score(
  candidate: candidate,
  query: "user query",
  context: {},
  stats: {}
)
# 权重：intent_match(0.4), trigger_match(0.2), success_rate(0.15), ...
```

### ActivationPlan

执行计划：

```ruby
plan.skills           # 选中的 skills
plan.primary_skill    # 主 skill
plan.fallback_chain   # 回退链
plan.parallel_groups  # 可并行执行的组
plan.estimated_cost   # 预估成本
```

## Execution Layer

### SkillExecutor

执行协调器：

```ruby
executor = SmartBot::SkillSystem::SkillExecutor.new

# 执行计划
result = executor.execute(plan, context: {})

# 执行单个 skill
result = executor.execute_skill(skill, { task: "do something" })
```

### Sandbox

权限控制和沙箱执行：

```ruby
sandbox = SmartBot::SkillSystem::Sandbox.new

# 检查权限
sandbox.check_permissions(skill.metadata.permissions)

# 执行
result = sandbox.execute(skill, parameters, context)
```

支持多种沙箱模式：
- `:none` - 无限制
- `:process` - 进程级限制（默认）
- `:container` - 容器（未实现）

### FallbackStateMachine

失败回退状态机：

```ruby
fsm = SmartBot::SkillSystem::FallbackStateMachine.new(
  plan: plan,
  executor: executor
)

result = fsm.run(context: {})
```

状态流转：
```
selected -> running -> success
              |
              v
        retryable_failure -> running (retry)
              |
              v
        fatal_failure -> fallback -> success/exit
```

## CLI 命令

```bash
# 列出 skills
smart_bot skill list

# 查看 skill 详情
smart_bot skill info weather

# 运行 skill
smart_bot skill run weather "Shanghai"

# 测试路由
smart_bot skill route "What's the weather?"

# 安装 skill
smart_bot skill install /path/to/skill
smart_bot skill install https://github.com/user/skill.git
```

## 配置

```yaml
# ~/.smart_bot/smart_bot.yml
skill_system:
  semantic_top_k: 3
  selection_threshold: 0.65
  max_parallel_skills: 2
  max_retry_per_path: 1
  default_timeout: 120
```

## 测试

```bash
# 运行所有 skill system 测试
bundle exec rspec spec/skill_system/

# 运行特定测试
bundle exec rspec spec/skill_system/core/
bundle exec rspec spec/skill_system/routing/
bundle exec rspec spec/skill_system/execution/
```

## 迁移指南

### 从旧 Skill 系统迁移

旧代码：
```ruby
SmartBot::Skill.load_all(skills_dir)
SmartBot::Skill.activate_all!
```

新代码：
```ruby
SmartBot::SkillSystem.load_all
# 自动注册到 registry，无需手动激活
```

### 向后兼容

旧的 `SmartBot::Skill` API 仍然可用，但建议使用新的 `SkillSystem` API。
