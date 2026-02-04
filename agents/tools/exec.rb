# frozen_string_literal: true

# Shell 命令执行工具
SmartAgent::Tool.define :exec do
  desc "Execute a shell command and return its output"
  param_define :command, "The shell command to execute", :string
  param_define :working_dir, "Optional working directory", :string
  
  tool_proc do
    require "open3"
    
    command = input_params["command"]
    working_dir = input_params["working_dir"]
    
    # 安全检查
    dangerous_patterns = [/rm\s+-rf\s+\//, />>\s*\/dev\/(null|zero|random)/]
    if dangerous_patterns.any? { |p| command.match?(p) }
      return { error: "Potentially dangerous command blocked" }
    end
    
    begin
      opts = {}
      opts[:chdir] = File.expand_path(working_dir) if working_dir && !working_dir.empty?
      
      stdout, stderr, status = Open3.capture3(command, opts)
      
      {
        command: command,
        stdout: stdout,
        stderr: stderr,
        exit_code: status.exitstatus,
        success: status.success?
      }
    rescue => e
      { error: "Error executing command: #{e.message}" }
    end
  end
end
