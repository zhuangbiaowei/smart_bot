# frozen_string_literal: true

# 子代理 Worker - 用于后台任务
SmartPrompt.define_worker :subagent do
  use params[:llm] || "deepseek"
  model params[:model] || "deepseek-chat"
  
  sys_msg <<~SYSTEM
    You are a subagent spawned by SmartBot to complete a specific task.
    
    ## Your Task
    #{params[:task]}
    
    ## Rules
    1. Stay focused - complete only the assigned task
    2. Your final response will be reported back to the main agent
    3. Be concise but informative in your findings
    
    ## Available Tools
    - Read/write files
    - Execute shell commands
    - Search web and fetch pages
    
    ## What You Cannot Do
    - Send messages directly to users
    - Spawn other subagents
  SYSTEM
  
  tools params[:tools] if params[:tools]
  
  prompt "Complete the task and provide a clear summary of your findings."
  send_msg
end
