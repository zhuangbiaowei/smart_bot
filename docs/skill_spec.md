# Skill Specification

**Version:** 0.1
**Status:** Draft (Implementable)
**Audience:** Agent Runtime Developers / Skill Authors / Platform Operators

---

## 1. 目标与设计原则

### 1.1 目标

本规范用于定义一种 **可创建、可安装、可分享、可执行、可治理** 的 Agent Skill 统一规范，满足以下要求：

* 兼容主流 Agent 工具（Claude Code / Codex / OpenCode / OpenClaw）
* 支持 **指令型（instruction-only）** 与 **执行型（script-backed）** Skill
* 支持依赖声明、权限控制、沙箱执行与审批机制
* 支持可信分发（签名、可追溯、可审计）
* 适用于个人、团队与组织级 Skill 生态

### 1.2 设计原则

1. **最小认知负担**：Skill 作者主要编写 `SKILL.md`
2. **渐进加载**：仅在需要时加载完整内容，节省上下文
3. **安全默认**：执行、权限、网络、密钥均默认收敛
4. **可投影性**：本规范是“超集”，可生成各平台兼容格式
5. **确定性优先**：执行结果可复现、可回放、可审计

---

## 2. Skill 类型

### 2.1 指令型 Skill（Instruction-only）

* 仅包含 **操作说明、决策流程、模板**
* 不直接执行本地或远程代码
* 适合 SOP、评审规范、分析方法、写作框架

### 2.2 执行型 Skill（Script-backed）

* 包含可执行脚本（Python / JS / Shell / Ruby 等）
* 由 Agent Runtime 调用并执行
* **必须**声明权限、依赖与执行策略
* **必须**在受控环境中运行

---

## 3. Skill 包结构

```
<skill-root>/
├─ SKILL.md                # 必须：人类可读指令定义
├─ skill.yaml              # 必须：机器可读超集清单
├─ scripts/                # 可选：可执行入口
├─ references/             # 可选：长文档、SOP、API说明
├─ assets/                 # 可选：模板、schema、prompt片段
├─ agents/                 # 可选：平台投影文件
│   ├─ openai.yaml
│   ├─ opencode.yaml
│   └─ openclaw.json
└─ signatures/             # 可选：签名与SBOM
    ├─ skill.sig
    └─ sbom.spdx.json
```

---

## 4. SKILL.md 规范（人类可读）

### 4.1 Frontmatter（YAML，必须）

```yaml
---
name: git_commit_reviewer
description: >
  Review git commits for message quality, diff correctness,
  and potential risks before merge.
version: 1.0.0
license: Apache-2.0
---
```

**强制字段**

| 字段          | 说明                                |
| ----------- | --------------------------------- |
| name        | Skill 全局唯一标识（建议 `publisher/name`） |
| description | 触发与使用场景描述                         |
| version     | 语义化版本                             |
| license     | 开源或使用许可                           |

### 4.2 指令正文（Markdown）

* **When to use**
* **When NOT to use**
* **Step-by-step instructions**
* **Input / Output expectations**
* **Failure handling**
* **Examples**

> SKILL.md 中 **不得**直接包含机密信息或不可审计的执行指令。

---

## 5. skill.yaml 规范（机器可读超集）

### 5.1 基本信息

```yaml
apiVersion: skill.dev/v1
kind: Skill
metadata:
  name: git_commit_reviewer
  publisher: example.org
  version: 1.0.0
  description: Review git commits before merge
```

### 5.2 Skill 类型

```yaml
spec:
  type: instruction | script
```

---

## 6. 执行型 Skill 规范（script-backed）

### 6.1 执行入口

```yaml
spec:
  entrypoints:
    - name: review
      runtime: python
      command: scripts/review.py
      inputs:
        repo_path: string
        commit_range: string
      outputs:
        report: markdown
```

### 6.2 依赖声明

```yaml
spec:
  dependencies:
    system:
      - git
    python:
      - pygit2>=1.13
```

### 6.3 权限模型（强制）

```yaml
spec:
  permissions:
    filesystem:
      read:
        - .
      write:
        - ./reports
    network:
      outbound: false
    environment:
      allow: []
```

> Runtime **必须强制执行** 权限最小化。

---

## 7. 执行策略（Execution Policy）

```yaml
spec:
  execution:
    sandbox: container        # none | process | container | microvm
    approval: ask             # auto | ask | manual
    timeout: 120s
```

* **auto**：低风险、确定性任务
* **ask**：涉及写文件、执行命令、网络
* **manual**：只生成计划或补丁，不执行

---

## 8. 机密与凭证（Secrets）

```yaml
spec:
  secrets:
    - name: GITHUB_TOKEN
      source: secret-store
      injectAs: env
```

规则：

* 不得写入 prompt、日志、SKILL.md
* 不得回传给模型作为上下文
* 仅在执行环境中注入

---

## 9. 渐进加载模型（Progressive Loading）

1. **Discovery 阶段**：
   加载 `name / description / version`
2. **Selection 阶段**：
   加载完整 `SKILL.md`
3. **Execution 阶段**：
   按需加载 `references / scripts / assets`

---

## 10. Skill 安装与发现

### 10.1 发现路径（优先级从高到低）

1. `<workspace>/skills`
2. `<repo-root>/.agents/skills`
3. `<repo-root>/.claude/skills`
4. `~/.agents/skills`
5. `~/.claude/skills`
6. `/etc/agent/skills`
7. runtime 内置 skills

### 10.2 安装行为（建议）

* 支持 Git / Registry / OCI / HTTP
* 安装时锁定内容摘要（digest）
* 支持 enable / disable

---

## 11. 分享与分发（Registry）

### 11.1 必须能力

* 发布者身份绑定（Git / OIDC）
* 版本化发布
* 内容摘要（不可变）
* 签名校验（Sigstore / Cosign）
* SBOM + 静态扫描

### 11.2 客户端安全策略

* 默认仅安装 **已签名 Skill**
* 权限声明不匹配 → 降权或拒绝
* 高风险 Skill 强制审批

---

## 12. 平台兼容性（Projection）

本规范 **不强制 Runtime 实现平台格式**，但建议提供自动投影：

| 平台          | 输出                   |
| ----------- | -------------------- |
| Codex       | `agents/openai.yaml` |
| OpenCode    | `.opencode/skills/*` |
| OpenClaw    | `openclaw.json`      |
| Claude Code | `.claude/skills/*`   |

---

## 13. 版本与兼容策略

* `apiVersion` 不兼容升级需显式声明
* Runtime **必须**拒绝未知高危字段
* Skill 作者应遵循语义化版本

---

## 14. 安全约束（强制）

* Skill **不得**自修改权限
* Skill **不得**绕过审批执行命令
* Skill **不得**请求未声明的资源
* Runtime **必须**记录执行审计日志

---

## 15. 附录：最小 Skill 示例

**Instruction-only**

```
my-skill/
├─ SKILL.md
└─ skill.yaml
```

**Script-backed**

```
my-skill/
├─ SKILL.md
├─ skill.yaml
└─ scripts/
   └─ run.py
```