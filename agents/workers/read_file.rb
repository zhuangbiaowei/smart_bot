# frozen_string_literal: true

# 文件读取 Worker
SmartPrompt.define_worker :read_file do
  use "deepseek"
  model "deepseek-chat"
  
  sys_msg "You are a file reading assistant. Read the file and summarize or analyze its contents as requested."
  
  prompt <<~PROMPT
    Please read and analyze this file:
    
    Path: #{params[:path]}
    Content:
    ```
    #{params[:content]}
    ```
    
    User request: #{params[:request] || "Summarize the content"}
  PROMPT
  
  send_msg
end
