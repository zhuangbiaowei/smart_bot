# SmartBot

🤖 **SmartBot** - 一个基于 [SmartAgent](https://github.com/zhuangbiaowei/smart_agent) 框架的个人 AI 助手，使用 Ruby 开发。

## 🏗️ 架构

SmartBot 构建于 **SmartAgent** 和 **SmartPrompt** 框架之上：

```
┌─────────────────────────────────────────────┐
│              SmartBot CLI/Agent             │
├─────────────────────────────────────────────┤
│              SmartAgent::Engine             │
│  ┌─────────────────────────────────────┐   │
│  │         SmartPrompt::Engine         │   │
│  │  ┌─────────┐ ┌─────────┐ ┌──────┐  │   │
│  │  │Workers  │ │Adapters │ │Tools │  │   │
│  │  └─────────┘ └─────────┘ └──────┘  │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

- **SmartPrompt**: 提供多 LLM 适配、Worker 定义、Prompt 模板
- **SmartAgent**: 提供 Agent 编排、工具调用、MCP 集成

## 功能特性

### 🤖 核心功能
- **多提供商 LLM 支持** - 基于 SmartPrompt，支持 DeepSeek、SiliconFlow、阿里云、Kimi 等
- **工具调用** - 基于 SmartAgent Tool 系统，自动编排工具调用
- **对话记忆** - 基于 SmartPrompt Conversation，自动管理对话历史
- **配置驱动** - YAML 配置文件，轻松切换模型和提供商
- **定时任务** - Cron 风格的任务调度
- **子代理** - 后台任务执行

### 🛠️ 可用工具

基于 **SmartAgent::Tool** 框架：

| 工具 | 描述 |
|------|------|
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件 |
| `edit_file` | 编辑文件（查找替换） |
| `list_dir` | 列出目录内容 |
| `exec` | 执行 shell 命令 |
| `web_search` | 网络搜索（需要 Brave API Key） |
| `web_fetch` | 抓取网页内容 |

## 快速开始

### 1. 安装依赖

```bash
cd ~/smart_ai/smart_bot
bundle install
```

### 2. 初始化配置

```bash
bundle exec bin/smart_bot onboard
```

这会创建：
- `~/.smart_bot/smart_prompt.yml` - SmartPrompt 配置文件
- `~/.smart_bot/agent.yml` - SmartAgent 配置文件
- `~/.smart_bot/workspace/` - 工作空间
- 默认的 AGENTS.md, SOUL.md, USER.md 等引导文件

### 3. 配置 API Key

编辑 `~/.smart_bot/smart_prompt.yml`：

```yaml
adapters:
  openai: OpenAIAdapter

llms:
  deepseek:
    adapter: openai
    url: https://api.deepseek.com
    api_key: "sk-your-deepseek-key"
    model: deepseek-chat  # 注意：使用 model 而非 default_model
  
  siliconflow:
    adapter: openai
    url: https://api.siliconflow.cn/v1/
    api_key: "sk-your-siliconflow-key"
    model: deepseek-ai/DeepSeek-V3
  
  aliyun:
    adapter: openai
    url: https://dashscope.aliyuncs.com/compatible-mode/v1/
    api_key: "sk-your-aliyun-key"
    model: qwen-plus

default_llm: deepseek
```

### 4. 开始对话

**单次对话：**
```bash
smart_bot agent -m "你好，请介绍一下你自己"
```

**交互模式：**
```bash
smart_bot agent
```

**交互模式命令：**

| 命令 | 描述 |
|------|------|
| `/models` | 列出所有可用 LLM 提供商 |
| `/llm <name>` | 切换到指定提供商 (如 deepseek, siliconflow) |
| `/help` | 显示帮助信息 |
| `Ctrl+C` | 退出交互模式 |

示例：
```
You: /models
📋 Available LLMs:
  → deepseek: deepseek-chat
    siliconflow: deepseek-ai/DeepSeek-V3
    aliyun: qwen-plus
    kimi: kimi-k2.5

You: /llm siliconflow
✓ Switched to LLM: siliconflow (deepseek-ai/DeepSeek-V3)
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
| `smart_bot skill NAME` | 创建新 Skill |
| `smart_bot cron list` | 列出定时任务 |
| `smart_bot cron add ...` | 添加定时任务 |

**交互模式命令：**

| 命令 | 描述 |
|------|------|
| `/models` | 列出所有可用 LLM 提供商 |
| `/llm <name>` | 切换到指定提供商 |
| `/skills` | 列出已加载的 Skills |
| `/help` | 显示帮助信息 |
| `Ctrl+C` | 退出交互模式 |

## 项目结构

```
~/.smart_bot/
├── smart_prompt.yml     # SmartPrompt 配置（LLM、API Keys）
├── agent.yml            # SmartAgent 配置
├── workspace/           # 工作空间
│   ├── AGENTS.md        # Agent 指令
│   ├── SOUL.md          # Bot 个性设定
│   ├── USER.md          # 用户信息
│   └── memory/          # 记忆文件
└── logs/                # 日志文件

~/smart_ai/smart_bot/
├── agents/
│   ├── smart_bot.rb     # Agent 定义（SmartAgent.define）
│   ├── workers/         # Workers（SmartPrompt.define_worker）
│   │   └── smart_bot.rb
│   ├── tools/           # Tools（SmartAgent::Tool.define）
│   │   ├── read_file.rb
│   │   ├── write_file.rb
│   │   └── ...
│   └── mcp_clients/     # MCP 客户端配置
│       └── all_in_one.rb
├── skills/              # Skill 插件目录
│   ├── search/          # 搜索 Skill
│   │   ├── skill.rb
│   │   └── SKILL.md
│   └── weather/         # 天气 Skill
│       ├── skill.rb
│       └── SKILL.md
└── config/
    └── smart_bot.yml    # 默认配置模板
```

## 配置说明

### LLM 配置格式

```yaml
llms:
  <provider_name>:
    adapter: openai       # 适配器类型
    url: <api_base_url>   # API 基础 URL
    api_key: "<api_key>"  # API Key（支持 ENV['KEY_NAME'] 格式）
    model: <model_name>   # 默认模型名称
    temperature: 0.7      # 可选：温度参数
```

### 支持的 LLM 提供商

| 提供商 | 配置键 | 推荐模型 |
|--------|--------|----------|
| DeepSeek | `deepseek` | deepseek-chat, deepseek-reasoner |
| SiliconFlow | `siliconflow` | deepseek-ai/DeepSeek-V3, Qwen/Qwen2.5-72B-Instruct |
| 阿里云 | `aliyun` | qwen-plus, qwen-max, qwen-coder-plus |
| Kimi | `kimi` | kimi-k2.5 |

### 可选工具配置

**MCP 服务（推荐优先使用）：**

SmartBot 已集成 DePHY Mesh API MCP 服务，提供强大的搜索、网页抓取、数据分析等功能。

```ruby
# 配置位于 agents/mcp_clients/all_in_one.rb
SmartAgent::MCPClient.define :all_in_one do
  type :sse
  url "https://mesh-api.dephy.io/mcp/d766aab9-eefb-4c82-b132-959370a131d8/sse"
end
```

**MCP 提供的工具：**
- `search` - 通用网络搜索（**优先使用**）
- `scrape` - 网页抓取
- `linkedin-search` / `company-search` - 商业信息搜索
- `maps_search_places` / `maps_directions` - 地图服务
- `get_token_price` / `get_wallet_activities` - 区块链数据
- 更多工具...

**使用方法：**
```bash
# 使用 MCP 搜索（默认优先）
smart_bot agent -m "搜索 OpenAI 最新动态"

# 抓取网页
smart_bot agent -m "抓取 https://example.com"
```

**启用网络搜索（推荐 SerpAPI）：**

SerpAPI 支持 Google、Bing、Baidu、Yahoo 等多种搜索引擎，返回结果更丰富。

1. 从 https://serpapi.com/ 注册获取 API Key
2. 设置环境变量：
```bash
export SERP_API_KEY="your-serpapi-key"
```

**使用方法：**
```bash
# 使用 Google 搜索（默认）
smart_bot agent -m "搜索 Ruby on Rails 最新版本"

# 使用 Baidu 搜索
smart_bot agent -m "baidu 搜索 Python 教程"

# 使用 Bing 搜索
smart_bot agent -m "bing 搜索 OpenAI API"
```

**备用方案 - Brave Search：**

如果无法使用 SerpAPI，可以使用 Brave Search：
```bash
export BRAVE_API_KEY="BSA-your-brave-key"
```

## Skill 系统

SmartBot 支持类似 OpenClaw 的 Skill 插件机制，可以方便地扩展功能。

### 创建 Skill

```bash
# 创建新 skill
smart_bot skill my_skill -d "My skill description" -a "Author Name"

# 这会创建：
# ~/smart_ai/smart_bot/skills/my_skill/
#   ├── skill.rb   # Skill 定义
#   └── SKILL.md   # 文档
```

### Skill 结构

```ruby
# skill.rb
SmartBot::Skill.register :my_skill do
  desc "My skill description"
  ver "1.0.0"
  author_name "Author Name"

  # 注册工具
  register_tool :my_tool do
    desc "Tool description"
    param_define :param1, "Parameter", :string
    
    tool_proc do
      { result: "success" }
    end
  end

  # 注册 MCP 客户端
  register_mcp :my_mcp do
    type :sse
    url "https://example.com/mcp/sse"
  end

  on_activate do
    SmartAgent.logger&.info "my_skill activated!"
  end
end
```

### 内置 Skills

| Skill | 描述 | 工具 |
|-------|------|------|
| `search` | 多源搜索 | `smart_search`, `smart_scrape` |
| `weather` | 天气查询 | `get_weather`, `get_forecast` |

### 查看已加载 Skills

```bash
# 在交互模式中
> /skills

🛠️  Loaded Skills:
  • search - 多源搜索功能
    Version: 1.0.0
    Tools: smart_search, smart_scrape
  • weather - 获取天气信息
    Version: 1.0.0
    Tools: get_weather, get_forecast
```

## 开发

### 基于 SmartAgent 扩展

**添加新 Worker：**

```ruby
# ~/.smart_bot/workers/my_worker.rb
SmartPrompt.define_worker :my_worker do
  use "deepseek"
  model "deepseek-chat"
  sys_msg "你是一个专业的助手"
  prompt params[:text]
  send_msg
end
```

**添加新 Tool：**

```ruby
# ~/.smart_bot/tools/my_tool.rb
SmartAgent::Tool.define :my_tool do
  desc "工具描述"
  param_define :param1, "参数说明", :string
  
  tool_proc do
    # 实现逻辑
    { result: "成功" }
  end
end
```

### 运行测试

```bash
cd ~/smart_ai/smart_bot
bundle exec rspec
```

## 故障排除

### 检查配置
```bash
smart_bot status
```

### 查看日志
```bash
tail -f ~/.smart_bot/logs/smart_prompt.log
tail -f ~/.smart_bot/logs/smart_agent.log
```

### 清除对话历史
```bash
rm -rf ~/.smart_bot/sessions/
```

### 依赖问题
确保 Ruby 版本 >= 3.2.0：
```bash
ruby -v
```

## 相关项目

- [SmartAgent](https://github.com/zhuangbiaowei/smart_agent) - Agent 框架
- [SmartPrompt](https://github.com/zhuangbiaowei/smart_prompt) - LLM 交互框架
- [nanobot](https://github.com/HKUDS/nanobot) - 灵感来源

## 许可证

MIT License - 详见 LICENSE 文件
