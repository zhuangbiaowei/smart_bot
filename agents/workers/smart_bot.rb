# frozen_string_literal: true

# 主对话 Worker - 处理用户输入
SmartPrompt.define_worker :smart_bot do
  # 使用指定的 LLM 或默认的 deepseek
  llm_name = params[:llm] || "deepseek"
  use llm_name
  
  # 如果有指定模型，使用指定模型，否则使用 LLM 的默认模型
  if params[:model]
    model params[:model]
  end
  
  sys_msg <<~SYSTEM
    You are SmartBot, a helpful AI assistant running on Ruby.
    
    You have access to tools that allow you to:
    - Read, write, and edit files
    - Execute shell commands  
    - Search the web and fetch web pages
    
    Current time: #{Time.now.strftime("%Y-%m-%d %H:%M (%A)")}
    
    Guidelines:
    - Always explain what you're doing before taking actions
    - Ask for clarification when the request is ambiguous
    - Use tools to help accomplish tasks
    - Be concise, accurate, and friendly
  SYSTEM
  
  prompt params[:text]
  
  if params[:stream]
    send_msg_by_stream
  else
    send_msg
  end
end
