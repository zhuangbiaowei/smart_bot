# frozen_string_literal: true

require "shellwords"

module SmartBot
  module SkillSystem
    module Execution
      # Validates and adapts commands from SKILL.md before execution
      class CommandValidator
        # Common executable patterns and their detection methods
        EXECUTABLE_PATTERNS = {
          /\byt-dlp\b/ => { type: :command, name: "yt-dlp", check: "which yt-dlp" },
          /\bpython3?\b/ => { type: :command, name: "python", check: "which python3 || which python" },
          /\bnode\b/ => { type: :command, name: "node", check: "which node" },
          /\bnpm\b/ => { type: :command, name: "npm", check: "which npm" },
          /\bruby\b/ => { type: :command, name: "ruby", check: "which ruby" },
          /\bcurl\b/ => { type: :command, name: "curl", check: "which curl" },
          /\bwget\b/ => { type: :command, name: "wget", check: "which wget" },
          /\bgit\b/ => { type: :command, name: "git", check: "which git" },
          /\bffmpeg\b/ => { type: :command, name: "ffmpeg", check: "which ffmpeg" },
          /\bmessage\b/ => { type: :command, name: "message", check: "which message" }
        }.freeze

        # Path patterns that may need adaptation
        PATH_PATTERNS = {
          %r{/root/clawd/} => -> { find_clawd_path },
          %r{/root/\.smart_bot/} => -> { File.expand_path("~/.smart_bot") },
          %r{/tmp/} => -> { Dir.tmpdir }
        }.freeze

        # Alternative commands for common failures
        COMMAND_ALTERNATIVES = {
          "yt-dlp" => [
            { check: "which yt-dlp", command: "yt-dlp" },
            { check: "which youtube-dl", command: "youtube-dl" },
            { check: "test -f ~/.local/bin/yt-dlp", command: "~/.local/bin/yt-dlp" }
          ],
          "python3" => [
            { check: "which python3", command: "python3" },
            { check: "which python", command: "python" },
            { check: "which python3.11", command: "python3.11" },
            { check: "which python3.10", command: "python3.10" }
          ],
          "node" => [
            { check: "which node", command: "node" },
            { check: "which nodejs", command: "nodejs" },
            { check: "test -f ~/.nvm/current/bin/node", command: "~/.nvm/current/bin/node" }
          ]
        }.freeze

        attr_reader :validation_errors, :adaptations_made

        def initialize
          @validation_errors = []
          @adaptations_made = []
          @available_tools = {}
          @confirmed_commands = {}
        end

        # Main entry point: validate and adapt a command
        # @param command [String] Original command from SKILL.md
        # @param options [Hash] Options including :require_confirmation
        # @return [Hash] Result with :valid, :command, :errors, :adaptations
        def validate_and_adapt(command, options = {})
          @validation_errors = []
          @adaptations_made = []

          result = {
            valid: true,
            original_command: command,
            command: command,
            errors: [],
            adaptations: [],
            requires_confirmation: false,
            confirmation_prompt: nil
          }

          # Step 1: Check for dangerous commands
          unless command_safe?(command)
            result[:valid] = false
            result[:errors] << "Command blocked by safety filter"
            return result
          end

          # Step 2: Detect required tools
          required_tools = detect_required_tools(command)

          # Step 3: Check tool availability and find alternatives
          missing_tools = []
          required_tools.each do |tool|
            unless tool_available?(tool)
              alternative = find_alternative_tool(tool)
              if alternative
                result[:command] = replace_tool_in_command(result[:command], tool, alternative)
                result[:adaptations] << "Replaced '#{tool}' with '#{alternative}'"
              else
                missing_tools << tool
              end
            end
          end

          unless missing_tools.empty?
            result[:valid] = false
            result[:errors] << "Missing required tools: #{missing_tools.join(', ')}"
          end

          # Step 4: Adapt paths
          adapted_command = adapt_paths(result[:command])
          if adapted_command != result[:command]
            result[:adaptations] << "Adapted paths for current environment"
            result[:command] = adapted_command
          end

          # Step 5: Check if confirmation is needed
          if options[:require_confirmation] && command_requires_confirmation?(result[:command])
            result[:requires_confirmation] = true
            result[:confirmation_prompt] = build_confirmation_prompt(result[:command])
          end

          result[:errors] = @validation_errors if @validation_errors.any?
          result
        end

        # Check if a command is safe to execute
        def command_safe?(command)
          blocked_patterns = [
            /\brm\s+-rf\s+\//i,  # rm -rf /
            /\bsudo\b/i,
            /\bapt(-get)?\s+(install|remove|purge)/i,
            /\byum\s+(install|remove)/i,
            /\bbrew\s+(install|uninstall)/i,
            /\bpip\s+install/i,
            /\bnpm\s+install\s+-g/i,
            /\bcurl\s+.*\|\s*(sh|bash)/i,
            /\bwget\s+.*\|\s*(sh|bash)/i,
            />\s*\/etc\//i,
            />\s*\/usr\//i
          ]

          blocked_patterns.none? { |pattern| command.match?(pattern) }
        end

        # Mark a command as confirmed by user
        def confirm_command!(command_hash)
          @confirmed_commands[command_hash[:command]] = true
        end

        # Check if command was already confirmed
        def confirmed?(command)
          @confirmed_commands.key?(command)
        end

        private

        # Detect which tools are required by the command
        def detect_required_tools(command)
          tools = []
          EXECUTABLE_PATTERNS.each do |pattern, info|
            tools << info[:name] if command.match?(pattern)
          end
          tools.uniq
        end

        # Check if a tool is available in the system
        def tool_available?(tool_name)
          return @available_tools[tool_name] if @available_tools.key?(tool_name)

          pattern = EXECUTABLE_PATTERNS.find { |_, info| info[:name] == tool_name }
          return false unless pattern

          check_cmd = pattern[1][:check]
          available = system("#{check_cmd} > /dev/null 2>&1")
          @available_tools[tool_name] = available
          available
        end

        # Find an alternative tool if the primary one is not available
        def find_alternative_tool(tool_name)
          return tool_name if tool_available?(tool_name)

          alternatives = COMMAND_ALTERNATIVES[tool_name]
          return nil unless alternatives

          alternatives.each do |alt|
            if system("#{alt[:check]} > /dev/null 2>&1")
              return alt[:command]
            end
          end

          nil
        end

        # Replace tool name in command
        def replace_tool_in_command(command, old_tool, new_tool)
          # Replace at word boundaries to avoid partial matches
          command.gsub(/\b#{Regexp.escape(old_tool)}\b/, new_tool)
        end

        # Adapt hardcoded paths to current environment
        def adapt_paths(command)
          adapted = command.dup

          PATH_PATTERNS.each do |pattern, resolver|
            if adapted.match?(pattern)
              new_path = resolver.call
              adapted = adapted.gsub(pattern, new_path + "/")
              @adaptations_made << "Replaced #{pattern.source} with #{new_path}"
            end
          end

          # Handle environment-specific home directory
          adapted = adapted.gsub(/\$HOME|\~\//, File.expand_path("~/"))

          adapted
        end

        # Check if command requires user confirmation
        def command_requires_confirmation?(command)
          risky_patterns = [
            /\brm\s+/i,
            /\bmv\s+/i,
            /\bcp\s+.*\s+\//i,
            />\s*\//i,
            /\bchmod\s+.*\s+\//i,
            /\bchown\s+/i
          ]

          risky_patterns.any? { |pattern| command.match?(pattern) }
        end

        # Build confirmation prompt for user
        def build_confirmation_prompt(command)
          <<~PROMPT
            ⚠️  The following command requires confirmation:

               #{command}

            This command may modify files or system state.
            Do you want to proceed? (yes/no)
          PROMPT
        end

        # Find the actual Clawd installation path
        def self.find_clawd_path
          # Check common locations
          possible_paths = [
            File.expand_path("~/clawd"),
            File.expand_path("~/.clawd"),
            "/opt/clawd",
            "/usr/local/clawd"
          ]

          possible_paths.find { |p| Dir.exist?(p) } || File.expand_path("~/clawd")
        end

        def find_clawd_path
          self.class.find_clawd_path
        end
      end
    end
  end
end
