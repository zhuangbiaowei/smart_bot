# frozen_string_literal: true

# Shell 命令执行 Worker
SmartPrompt.define_worker :exec_shell do
  use "deepseek"
  model "deepseek-chat"
  
  sys_msg "You are a shell command assistant. Analyze command output and provide insights."
  
  prompt <<~PROMPT
    Command executed: `#{params[:command]}`
    Working directory: #{params[:working_dir] || 'current'}
    
    Output:
    ```
    #{params[:output]}
    ```
    
    #{params[:error] ? "Error:\n```\n#{params[:error]}\n```" : ""}
    
    Please analyze the output and provide a clear summary.
  PROMPT
  
  send_msg
end
