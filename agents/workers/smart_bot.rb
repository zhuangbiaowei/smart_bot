# frozen_string_literal: true

# 主对话 Worker - 处理用户输入
SmartPrompt.define_worker :smart_bot do
  # 使用指定的 LLM 或默认的 deepseek
  llm_name = params[:llm] || "deepseek"
  use llm_name
  
  # 如果有指定模型，使用指定模型
  if params[:model]
    model params[:model]
  end
  
  # 设置系统消息 - 强调必须使用工具
  sys_msg <<~SYSTEM
You are SmartBot, a helpful AI assistant.

You have access to the following tools:
- read_file: Read file contents
- write_file: Write content to a file  
- edit_file: Edit file by replacing text
- list_dir: List directory contents
- exec: Execute shell commands
- web_search: Search the web (requires BRAVE_API_KEY env var)
- web_fetch: Fetch and extract content from URLs

CRITICAL INSTRUCTIONS:
1. When the user asks you to fetch a URL, read a file, or execute a command, you MUST use the appropriate tool
2. Do not say you will use the tool - actually call it using the function_call format
3. After receiving tool results, analyze them and provide a helpful response
4. Current time: #{Time.now.strftime("%Y-%m-%d %H:%M")}
  SYSTEM
  
  # 设置用户输入
  prompt params[:text]
  
  # 发送消息
  send_msg
end
