# frozen_string_literal: true

require "open3"
require "timeout"

module SmartBot
  module Tools
    class ShellTool < Base
      def initialize(timeout: 60, working_dir: nil)
        @timeout = timeout
        @working_dir = working_dir || Dir.pwd
        
        super(
          name: :exec,
          description: "Execute a shell command and return its output. Use with caution.",
          parameters: {
            type: "object",
            properties: {
              command: { type: "string", description: "The shell command to execute" },
              working_dir: { type: "string", description: "Optional working directory for the command" }
            },
            required: ["command"]
          }
        )
      end

      def execute(command:, working_dir: nil)
        cwd = working_dir || @working_dir
        
        stdout_str, stderr_str, status = nil
        
        begin
          Timeout.timeout(@timeout) do
            stdout_str, stderr_str, status = Open3.capture3(command, chdir: cwd)
          end
        rescue Timeout::Error
          return "Error: Command timed out after #{@timeout} seconds"
        end

        output_parts = []
        output_parts << stdout_str if stdout_str && !stdout_str.empty?
        
        if stderr_str && !stderr_str.empty?
          output_parts << "STDERR:\n#{stderr_str}"
        end

        if status && !status.success?
          output_parts << "\nExit code: #{status.exitstatus}"
        end

        result = output_parts.empty? ? "(no output)" : output_parts.join("\n")
        
        # Truncate very long output
        max_len = 10000
        if result.length > max_len
          result = result[0...max_len] + "\n... (truncated, #{result.length - max_len} more chars)"
        end

        result
      rescue => e
        "Error executing command: #{e.message}"
      end
    end
  end
end
