# Skill Routing Spec

版本: `v0.1`
状态: `draft`
适用范围: `SmartBot skill 选择与执行策略`

## 1. 目标

- 在多 skill 环境中提高命中率和执行成功率。
- 控制上下文体积、token 成本和执行时延。
- 提供稳定的失败回退路径，避免反复无效尝试。
- 为后续自动调参与 A/B 提供可观测数据。

## 2. 非目标

- 不定义单个 skill 的业务逻辑细节。
- 不替代现有 tool 安全策略。
- 不在本版本实现在线学习或复杂强化学习。

## 3. 术语

- `SkillCandidate`: 路由阶段的候选 skill。
- `ActivationPlan`: 最终执行的 skill 集合与顺序。
- `Hard Trigger`: 用户显式指令触发，例如 `$skill-name`。
- `Soft Match`: 语义召回或规则匹配触发。
- `Fallback`: 某 skill 执行失败后的降级路径。

## 4. 输入与输出

### 4.1 输入

- 用户消息 `query`
- 会话上下文 `history`
- 可用 skills 清单 `skills[]`
- 每个 skill 的元数据 `metadata`
- 历史统计 `skill_stats`

### 4.2 输出

- `ActivationPlan`
- 路由解释 `routing_reason`
- 观测日志事件 `routing_events[]`

## 5. Skill 元数据规范

每个 skill 建议在 frontmatter 增加以下字段:

```yaml
name: invoice-organizer
description: Organize invoice files
triggers:
  - "发票"
  - "invoice"
anti_triggers:
  - "天气"
cost_hint: medium # low|medium|high
prerequisites:
  bins: ["pdftotext"]
  env: ["OPENAI_API_KEY"]
parallel_safe: false
always: false
```

字段说明:

- `triggers`: 高置信关键词/短语，提升召回分。
- `anti_triggers`: 排斥词，命中时强惩罚。
- `cost_hint`: 成本等级，影响打分惩罚项。
- `prerequisites`: 可执行前置条件。
- `parallel_safe`: 是否允许与其他 skill 并行执行。
- `always`: 会话启动即注入（慎用）。

## 6. 路由流程

### 6.1 阶段一: 快速路由

1. 识别 `Hard Trigger`:
- 命中 `$skill-name` 或明确“使用 X skill”时，加入候选并标记 `forced=true`。

2. 基于规则召回:
- 使用 `triggers` 与意图分类结果生成候选集合。

3. 语义召回:
- 对所有可用 skill 做 top-k 召回，默认 `k=3`。

4. 合并去重:
- 得到 `SkillCandidate[]`，保留来源标签: `forced/rule/semantic`。

### 6.2 阶段二: 打分与裁剪

对每个候选计算:

```text
Score =
  0.40 * intent_match +
  0.20 * trigger_match +
  0.15 * success_rate +
  0.10 * context_readiness +
  0.10 * cost_penalty +
  0.05 * conflict_penalty
```

约束:

- `forced=true` 的 skill 默认跳过阈值过滤，但仍进行安全校验。
- 若命中 `anti_triggers`，`conflict_penalty` 置为高惩罚。
- 前置条件不满足时直接标记 `unavailable`。

阈值:

- 默认阈值 `T=0.65`。
- 无候选过阈值时:
- 若存在 `forced`，仅执行 `forced`。
- 否则走无 skill 通用流程。

## 7. 多 Skill 组合策略

- 目标是“最小覆盖集”，避免把高重叠 skill 同时拉起。
- 覆盖度相近时优先:
- 成本更低的 skill。
- 最近成功率更高的 skill。
- `parallel_safe=true` 且无资源冲突时可并行执行。
- 并行上限建议 `max_parallel_skills=2`。

## 8. 执行策略

- 仅按需读取 `SKILL.md` 必要段落。
- 若引用 `references/`，只加载被当前任务命中的文件。
- 优先使用 skill 自带 `scripts/`、模板和资产，减少手工拼接。
- 执行前生成 `ActivationPlan`，包含:
- skill 名称
- 执行顺序/并行组
- 预估成本
- 回退路径

## 9. 冲突处理优先级

1. 用户显式指令
2. 安全/权限约束
3. 系统策略与路由阈值
4. skill 默认流程

当同级冲突无法自动解决时，触发澄清问题或选用低风险路径。

## 10. 失败回退状态机

状态:

- `selected` -> `running` -> `success`
- `running` -> `retryable_failure`
- `running` -> `fatal_failure`
- `retryable_failure` -> `running` (最多 1 次重试)
- `retryable_failure` -> `fallback`
- `fatal_failure` -> `fallback`
- `fallback` -> `success` 或 `exit`

规则:

- 同一路径连续失败 2 次，禁止本轮继续重试该路径。
- 前置条件缺失属于 `fatal_failure`，直接切换 fallback。
- fallback 优先级:
1. 同意图次优 skill
2. 无 skill 通用工具链
3. 请求用户补充信息

## 11. 可修复失败判定

当某 skill 执行失败时，先判断是否进入自修复流程。仅在“接近成功”时允许修复。

### 11.1 可修复失败类型

- 参数映射错误或字段缺失。
- `SKILL.md` 指令不完整或示例命令不可执行。
- skill 内脚本出现小范围实现问题。
- skill 内引用路径错误或模板引用缺失。
- 前置条件声明缺失但可明确补齐。

### 11.2 不可修复失败类型

- 任务目标与该 skill 核心能力不匹配。
- 关键外部依赖不可用且无法在当前环境补齐。
- 权限/安全限制导致无法继续。
- 多步骤链路持续失败且无稳定中间产物。

### 11.3 “接近成功”判定

建议使用打分门槛，满足以下条件中的多数才可进入自修复:

- `task_progress >= 0.70`
- `root_cause_count <= 2`
- `change_scope_within_skill = true`
- `estimated_fix_attempts <= 1`
- 最近一次执行存在有效中间产物

若不满足，则直接进入常规 fallback，不执行 skill 文件修改。

## 12. 自修复状态机

状态:

- `failure_detected` -> `repair_eligibility_check`
- `repair_eligibility_check` -> `repair_planning`
- `repair_eligibility_check` -> `fallback` (不可修复)
- `repair_planning` -> `patching`
- `patching` -> `retry_execution`
- `retry_execution` -> `success`
- `retry_execution` -> `repair_evaluation`
- `repair_evaluation` -> `repair_planning` (仍可修复且预算未耗尽)
- `repair_evaluation` -> `fallback` (无改进或不可修复)

修复约束:

- 仅允许修改当前 skill 目录内文件:
- `SKILL.md`
- `scripts/*`
- `references/*`
- 每次修复必须输出结构化记录:
- `error_signature`
- `root_cause`
- `patched_files`
- `expected_effect`
- 禁止自动修改全局安全策略、权限策略和无关 skill。

## 13. 重试预算

为避免无限重试和退化行为，引入统一预算控制:

- `max_repair_attempts_per_run = 2`
- `max_total_retries_per_skill = 2`
- `max_patched_files_per_attempt = 3`
- `max_patch_hunks_per_attempt = 8`
- `max_added_tokens_in_skill_doc = 300`

提前终止条件:

- 连续 1 次修复后无指标改进。
- 新错误数量高于修复前。
- 触发安全约束或超出修改范围。

改进判定建议:

- 执行步数推进。
- 错误从 `fatal` 降级为 `retryable`。
- 输出产物完整度提升。

## 14. 观测与埋点

每次路由记录:

- `route_id`
- `query_type`
- `candidate_count`
- `selected_skills`
- `score_breakdown`
- `latency_ms`
- `tokens_in/tokens_out`
- `execution_result`
- `fallback_used`

核心指标:

- `trigger_rate`
- `selection_precision`
- `success_rate`
- `fallback_rate`
- `p50/p95_latency`
- `avg_token_cost`

## 15. 调参与发布策略

- 使用小流量灰度调权重和阈值。
- 每周按任务类型复盘 `p50/p95` 和成功率。
- 清理低收益规则:
- 触发高但成功率持续低于阈值的触发词。
- 对新 skill 启用冷启动保护:
- 前 `N=20` 次请求中降低最高优先级，防止误触发放大。

## 16. 伪代码

```ruby
def route_skills(query:, skills:, stats:)
  candidates = []
  candidates += hard_triggers(query, skills)
  candidates += rule_recall(query, skills)
  candidates += semantic_top_k(query, skills, k: 3)
  candidates = dedup(candidates)

  scored = candidates.map { |c| [c, score(c, query, stats)] }
  selected = pick_by_threshold(scored, t: 0.65)
  selected = forced_only(candidates) if selected.empty? && forced_exists?(candidates)

  plan = build_activation_plan(selected)
  execute_with_fallback(plan)
end
```

## 17. 与当前代码的最小集成点

- `lib/smart_bot/agent/context.rb`
- 在 `Skills` 类中补充 frontmatter 字段解析与可用性检查。

- `lib/smart_bot/skill_adapters/unified_skill_loader.rb`
- 在加载阶段读取并缓存 skill 元数据，供路由器使用。

- 新增 `lib/smart_bot/skill_routing/`
- `router.rb`: 两阶段路由逻辑
- `scorer.rb`: 打分与阈值
- `fallback.rb`: 回退状态机
- `repair_loop.rb`: 可修复失败判定与自修复重试
- `events.rb`: 埋点结构

## 18. 默认参数建议

- `semantic_top_k = 3`
- `selection_threshold = 0.65`
- `max_parallel_skills = 2`
- `max_retry_per_path = 1`
- `cool_start_requests = 20`
- `max_repair_attempts_per_run = 2`
- `max_skill_delegate_depth = 2`

以上参数建议放入 `config/smart_bot.yml`，便于运行时调整。

## 19. MVP 实现说明（已落地）

当前最小可运行版本提供一个统一的技能委派入口工具:

- 工具名: `run_skill`
- 目标: 在当前任务中把子任务委派给指定 skill 执行
- 执行方式: 通过子代理异步执行，并在完成后回传结果

CLI 兼容入口（`smart_bot agent`）:

- 斜杠命令: `/run_skill <skill_name> <task>`
- 消息语法: `run_skill <skill_name>: <task>`
- 安全限制: 同样执行深度限制和循环检测

核心行为:

1. 校验 skill 是否存在（`skills/<name>/SKILL.md`）。
2. 自动注入委派上下文（`target_skill`、`call_chain`、`skill_file`）。
3. 执行循环检测（例如 `a -> b -> a`）。
4. 执行深度限制（默认 `max_depth=2`）。
5. 返回结构化结果（`delegated` 或 `error`）。

实现落点:

- `lib/smart_bot/skill_routing/orchestrator.rb`
- `lib/smart_bot/tools/run_skill.rb`
- `lib/smart_bot/agent/loop.rb`
- `lib/smart_bot/agent/subagent.rb`
