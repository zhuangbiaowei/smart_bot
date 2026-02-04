# SmartBot

🤖 **SmartBot** - 一个轻量级个人 AI 助手，基于 Ruby 开发，灵感来自 [nanobot](https://github.com/HKUDS/nanobot)。

## 功能特性

### 🤖 核心功能
- **多提供商 LLM 支持** - 支持 DeepSeek、SiliconFlow、阿里云、Kimi、OpenRouter、Anthropic、OpenAI、Gemini 等
- **工具调用** - LLM 可以使用各种工具来完成任务
- **对话记忆** - 自动保存对话历史，支持长期记忆
- **定时任务** - Cron 风格的任务调度
- **子代理** - 后台任务执行

### 🛠️ 可用工具
| 工具 | 描述 |
|------|------|
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件 |
| `edit_file` | 编辑文件（查找替换） |
| `list_dir` | 列出目录内容 |
| `exec` | 执行 shell 命令 |
| `web_search` | 网络搜索（需要 Brave API Key） |
| `web_fetch` | 抓取网页内容 |
| `message` | 发送消息到聊天频道 |
| `spawn` | 生成子代理执行后台任务 |

### 📡 聊天频道（可选）
- **Telegram** - 通过 Telegram Bot 聊天
- **WhatsApp** - 预留接口（未实现）

## 快速开始

### 1. 初始化配置

```bash
cd ~/smart_ai/smart_bot
bundle exec bin/smart_bot onboard
```

这会创建：
- `~/.smart_bot/config.json` - 配置文件
- `~/.smart_bot/workspace/` - 工作空间
- 默认的 AGENTS.md, SOUL.md, USER.md 等引导文件

### 2. 配置 API Key

编辑 `~/.smart_bot/config.json`，添加你的 API Key：

```json
{
  "model": "deepseek-chat",
  "providers": {
    "deepseek": {
      "api_key": "sk-your-deepseek-key",
      "api_base": "https://api.deepseek.com"
    },
    "siliconflow": {
      "api_key": "sk-your-siliconflow-key",
      "api_base": "https://api.siliconflow.cn/v1/"
    },
    "aliyun": {
      "api_key": "sk-your-aliyun-key",
      "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1/"
    },
    "kimi_coding": {
      "api_key": "sk-your-kimi-key",
      "api_base": "https://api.kimi.com/coding/v1"
    }
  }
}
```

### 3. 开始对话

**单次对话：**
```bash
smart_bot agent -m "你好，请介绍一下你自己"
```

**交互模式：**
```bash
smart_bot agent
```

**指定会话（隔离对话历史）：**
```bash
smart_bot agent -s "project1" -m "记住这是项目1"
```

## CLI 命令

| 命令 | 描述 |
|------|------|
| `smart_bot onboard` | 初始化配置和工作空间 |
| `smart_bot agent -m "消息"` | 单次对话模式 |
| `smart_bot agent` | 交互对话模式 |
| `smart_bot status` | 查看配置状态 |
| `smart_bot gateway` | 启动网关（Telegram 等） |

### Cron 定时任务

```bash
# 添加定时任务 - 每小时执行
smart_bot cron add --name "hourly_check" --message "检查系统状态" --every 3600

# 添加 Cron 表达式任务 - 每天 9:00
smart_bot cron add --name "morning" --message "早上好！" --cron "0 9 * * *"

# 列出所有任务
smart_bot cron list

# 删除任务
smart_bot cron remove <job_id>

# 手动执行任务
smart_bot cron execute <job_id>
```

## 项目结构

```
~/.smart_bot/
├── config.json          # 配置文件
└── workspace/
    ├── AGENTS.md        # Agent 指令
    ├── SOUL.md          # Bot 个性设定
    ├── USER.md          # 用户信息
    ├── TOOLS.md         # 工具说明（可选）
    ├── IDENTITY.md      # 身份设定（可选）
    ├── memory/
    │   ├── MEMORY.md    # 长期记忆
    │   └── 2026-02-04.md # 每日笔记
    └── skills/          # 自定义技能
        └── my_skill/
            └── SKILL.md
```

## 配置说明

### 支持的 LLM 提供商

| 提供商 | 配置键 | 默认 Base URL |
|--------|--------|---------------|
| DeepSeek | `deepseek` | https://api.deepseek.com |
| SiliconFlow | `siliconflow` | https://api.siliconflow.cn/v1/ |
| 阿里云 | `aliyun` | https://dashscope.aliyuncs.com/compatible-mode/v1/ |
| Kimi Coding | `kimi_coding` | https://api.kimi.com/coding/v1 |
| OpenRouter | `openrouter` | https://openrouter.ai/api/v1 |
| Anthropic | `anthropic` | - |
| OpenAI | `openai` | - |
| Gemini | `gemini` | - |

### 模型推荐

**DeepSeek:**
- `deepseek-chat` - 通用对话
- `deepseek-reasoner` - 推理模型

**SiliconFlow:**
- `deepseek-ai/DeepSeek-V3` - DeepSeek V3
- `Qwen/Qwen2.5-72B-Instruct` - Qwen 2.5

**阿里云:**
- `qwen-plus` - 通义千问 Plus
- `qwen-max` - 通义千问 Max

### 可选工具配置

```json
{
  "tools": {
    "web_search": {
      "api_key": "BSA-your-brave-key",
      "max_results": 5
    }
  }
}
```

获取 Brave API Key: https://brave.com/search/api/

### Telegram 配置（可选）

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "token": "YOUR_BOT_TOKEN",
      "allow_from": ["YOUR_USER_ID"]
    }
  }
}
```

1. 从 [@BotFather](https://t.me/BotFather) 创建 Bot 获取 token
2. 从 [@userinfobot](https://t.me/userinfobot) 获取你的 user ID
3. 启动网关: `smart_bot gateway`

## 开发

### 运行测试

```bash
cd ~/smart_ai/smart_bot
bundle install
bundle exec rspec
```

### 项目结构

```
smart_bot/
├── bin/smart_bot        # 可执行文件
├── lib/
│   └── smart_bot/
│       ├── agent/       # Agent 核心逻辑
│       ├── bus/         # 消息总线
│       ├── channels/    # 聊天频道集成
│       ├── cli/         # 命令行界面
│       ├── config/      # 配置管理
│       ├── cron/        # 定时任务
│       ├── heartbeat/   # 心跳系统
│       ├── providers/   # LLM 提供商
│       ├── session/     # 会话管理
│       ├── tools/       # 工具实现
│       └── utils/       # 工具函数
└── skills/              # 内置技能
```

### 添加自定义技能

在工作空间创建 `skills/my_skill/SKILL.md`：

```markdown
---
name: my_skill
description: 我的自定义技能
---

# My Skill

这里写技能的使用说明...
```

## 故障排除

### 会话历史问题
如果对话出现异常，可以清除会话历史：
```bash
rm -rf ~/.smart_bot/sessions/
```

### API 错误
检查 API Key 是否正确：
```bash
smart_bot status
```

### 依赖问题
确保 Ruby 版本 >= 3.2.0：
```bash
ruby -v
```

## 许可证

MIT License - 详见 LICENSE 文件

## 致谢

灵感来自 [nanobot](https://github.com/HKUDS/nanobot) 项目。
