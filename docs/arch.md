# SmartBot 架构设计文档

## 1. 架构概览

SmartBot 采用**模块化、分层架构**设计，灵感来自 [nanobot](https://github.com/HKUDS/nanobot)。核心设计原则：

- **单一职责** - 每个模块只负责一个功能领域
- **依赖注入** - 通过构造函数注入依赖，便于测试和替换
- **事件驱动** - 使用消息总线解耦组件
- **可扩展** - 插件式工具、技能系统

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI / Gateway                        │
│                   (bin/smart_bot, Thor)                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      Agent::Loop                             │
│              (主代理循环，协调对话和工具调用)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐  ┌────────▼────────┐  ┌──────▼──────┐
│   Provider   │  │     Tools       │  │   Session   │
│  (LLM API)   │  │  (工具注册表)    │  │  (对话历史)  │
└──────────────┘  └─────────────────┘  └─────────────┘
        │                   │                   │
┌───────▼──────┐  ┌────────▼────────┐  ┌──────▼──────┐
│    Context   │  │   Subagent      │  │   Memory    │
│  (Prompt构建) │  │   (子代理管理)   │  │  (长期记忆)  │
└──────────────┘  └─────────────────┘  └─────────────┘
        │
┌───────▼───────────────────────────────────────────────────┐
│                      Bus::Queue                            │
│              (消息总线，解耦频道和代理)                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐  ┌────────▼────────┐  ┌──────▼──────┐
│   Channels   │  │   Cron Service  │  │  Heartbeat  │
│ (Telegram等) │  │   (定时任务)     │  │  (心跳系统)  │
└──────────────┘  └─────────────────┘  └─────────────┘
```

## 2. 核心模块

### 2.1 Agent 层 (`lib/smart_bot/agent/`)

#### Agent::Loop
- **职责**: 主代理循环，协调对话流程
- **核心方法**:
  - `process_message()` - 处理单条消息
  - `process_direct()` - CLI 直接调用
  - `run()` - 启动消息循环（Gateway 模式）
- **迭代逻辑**: 支持多轮工具调用，最多 `max_iterations` 轮

#### Agent::Context
- **职责**: Prompt 构建和上下文管理
- **功能**:
  - 加载引导文件 (AGENTS.md, SOUL.md, USER.md)
  - 整合长期记忆和每日笔记
  - 构建 OpenAI 格式的消息列表

#### Agent::SubagentManager
- **职责**: 后台子代理管理
- **特点**:
  - 独立线程运行
  - 独立的工具集（不含 message/spawn）
  - 完成后通过消息总线汇报结果

### 2.2 Provider 层 (`lib/smart_bot/providers/`)

#### Providers::OpenRouterProvider
- **职责**: 与 LLM API 通信
- **当前实现**: 基于 OpenAI 兼容 API
- **可扩展**: 继承 `Base` 类可添加其他提供商

```ruby
class OpenRouterProvider < Base
  def chat(messages:, tools:, model:)
    # 构建 HTTP 请求
    # 解析响应，返回 LLMResponse
  end
end
```

### 2.3 工具层 (`lib/smart_bot/tools/`)

#### 工具注册表 (Tools::Registry)
```ruby
@tools = Tools::Registry.new
@tools.register(Tools::ReadFileTool.new)
result = @tools.execute(:read_file, { path: "file.txt" })
```

#### 内置工具

| 工具类 | 功能 | 安全级别 |
|--------|------|----------|
| `ReadFileTool` | 读取文件 | 安全 |
| `WriteFileTool` | 写入文件 | 中等 |
| `EditFileTool` | 编辑文件 | 中等 |
| `ListDirTool` | 列出目录 | 安全 |
| `ShellTool` | 执行 shell | 危险 |
| `WebSearchTool` | 网络搜索 | 安全 |
| `WebFetchTool` | 网页抓取 | 安全 |
| `MessageTool` | 发送消息 | 安全 |
| `SpawnTool` | 生成子代理 | 安全 |
| `RunSkillTool` | 委派子任务给指定 Skill | 中等 |

#### 工具定义格式 (OpenAI Function Calling)
```ruby
{
  type: "function",
  function: {
    name: "read_file",
    description: "...",
    parameters: {
      type: "object",
      properties: { ... },
      required: ["path"]
    }
  }
}
```

### 2.4 消息总线 (`lib/smart_bot/bus/`)

#### Bus::Queue
- **职责**: 解耦消息生产者和消费者
- **队列**:
  - `inbound` - 用户输入消息
  - `outbound` - 代理输出消息
- **订阅模式**: 频道可订阅 outbound 消息

```ruby
bus = Bus::Queue.new
bus.publish_inbound(message)
msg = bus.consume_inbound(timeout: 1)
```

### 2.5 会话管理 (`lib/smart_bot/session/`)

#### Session::Manager
- **存储**: JSON Lines 格式 (`~/.smart_bot/sessions/{session_key}.jsonl`)
- **结构**:
  - 首行: 元数据 (创建时间、更新时间)
  - 后续: 消息记录
- **缓存**: 内存缓存活跃会话

### 2.6 定时任务 (`lib/smart_bot/cron/`)

#### Cron::Service
- **存储**: `~/.smart_bot/cron/jobs.json`
- **调度类型**:
  - `every` - 固定间隔（秒）
  - `cron` - Cron 表达式
  - `at` - 一次性任务
- **线程模型**: 独立线程 + 条件变量等待

### 2.7 频道集成 (`lib/smart_bot/channels/`)

#### Channels::Manager
- **职责**: 管理多个聊天频道
- **当前支持**: Telegram
- **扩展**: 实现 `Base` 类可添加新频道

## 3. 数据流

### 3.1 单次对话流程 (CLI)

```
┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────┐
│ 用户输入 │───▶│ Agent::Loop │───▶│  build_msg  │───▶│ Context │
└─────────┘    └─────────────┘    └─────────────┘    └────┬────┘
                                                          │
┌─────────┐    ┌─────────────┐    ┌─────────────┐        │
│ 显示响应 │◄───│   format    │◄───│ LLM Response│◄───────┘
└─────────┘    └─────────────┘    └─────────────┘
                      ▲                   │
                      │            ┌──────┴──────┐
                      └────────────│ Tool Calls? │
                                   └──────┬──────┘
                                          │ 是
                                   ┌──────▼──────┐
                                   │ Execute Tool│
                                   └──────┬──────┘
                                          │
                                          └──────────────▶
```

### 3.2 工具调用流程

```ruby
# 1. 发送请求（含工具定义）
response = provider.chat(messages: msgs, tools: tool_defs)

# 2. 检查是否需要调用工具
if response.has_tool_calls?
  response.tool_calls.each do |tc|
    # 3. 执行工具
    result = tools.execute(tc.name, tc.arguments)
    
    # 4. 将结果加入上下文
    messages << { role: "tool", tool_call_id: tc.id, content: result }
  end
  
  # 5. 再次请求 LLM
  response = provider.chat(messages: messages, tools: tool_defs)
end
```

## 4. 设计模式

### 4.1 注册表模式 (Registry)
用于工具管理：
```ruby
class Tools::Registry
  def register(tool)
    @tools[tool.name] = tool
  end
  
  def execute(name, params)
    @tools[name].execute(**params)
  end
end
```

### 4.2 策略模式 (Strategy)
用于 LLM 提供商：
```ruby
class Agent::Loop
  def initialize(provider:)
    @provider = provider  # 可替换为任何 Provider
  end
end
```

### 4.3 工厂模式
用于创建会话：
```ruby
class Session::Manager
  def get_or_create(key)
    @cache[key] ||= load(key) || Session.new(key)
  end
end
```

### 4.4 观察者模式
通过消息总线实现：
```ruby
bus.subscribe_outbound("telegram") do |msg|
  telegram_channel.send(msg)
end
```

## 5. 扩展点

### 5.1 添加新工具

```ruby
# lib/smart_bot/tools/my_tool.rb
class MyTool < Base
  def initialize
    super(
      name: :my_tool,
      description: "...",
      parameters: { ... }
    )
  end
  
  def execute(param1:)
    # 实现逻辑
  end
end

# 在 Agent::Loop#register_default_tools 中注册
@tools.register(Tools::MyTool.new)
```

### 5.2 添加新 Provider

```ruby
class MyProvider < Base
  def chat(messages:, tools:, model:)
    # HTTP 请求实现
    LLMResponse.new(content: "...", tool_calls: [])
  end
end
```

### 5.3 添加新频道

```ruby
class MyChannel < Channels::Base
  def start
    # 启动监听线程
  end
  
  def send(message)
    # 发送消息实现
  end
end
```

## 6. 配置架构

### 6.1 配置优先级
1. 环境变量（预留）
2. `~/.smart_bot/config.json`
3. 代码默认值 (`Config::DEFAULTS`)

### 6.2 配置结构
```ruby
{
  model: String,           # 默认模型
  providers: Hash,         # API 配置
  channels: Hash,          # 频道配置
  tools: Hash,             # 工具配置
  workspace: String,       # 工作空间路径
  max_tool_iterations: Integer
}
```

## 7. 安全考虑

### 7.1 工具执行安全
- **ShellTool**: 在配置的工作目录下执行
- **文件操作**: 限制在工作空间内
- **Web 访问**: 仅允许出站 HTTP

### 7.2 API Key 管理
- 存储在本地配置文件
- 不记录到日志
- 支持多个提供商轮换

## 8. 性能优化

### 8.1 内存管理
- 会话使用 LRU 缓存
- 大文件读取限制行数
- 工具结果截断

### 8.2 并发
- 每个频道独立线程
- 子代理异步执行
- Cron 服务独立线程

## 9. 依赖关系

```
smart_bot
├── thor (~> 1.3)          # CLI 框架
├── base64 (~> 0.2)        # Ruby 3.4+ 兼容
├── mime-types (~> 3.0)    # 文件类型检测
│
├── smart_agent (local)    # 代理框架
├── smart_prompt (local)   # Prompt 管理
├── smart_rag (local)      # RAG 和记忆
└── ruby_rich (local)      # 控制台输出
```

## 10. 未来扩展

- [ ] MCP (Model Context Protocol) 支持
- [ ] 向量数据库存储记忆
- [ ] 多模态输入（图片、音频）
- [ ] 插件热加载
- [ ] Web UI 界面
