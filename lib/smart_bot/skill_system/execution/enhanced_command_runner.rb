# frozen_string_literal: true

require_relative "command_validator"
require_relative "retry_executor"

module SmartBot
  module SkillSystem
    module Execution
      # Enhanced command execution with validation, adaptation, retry, and confirmation
      # Integrates with existing CLI command execution flow
      class EnhancedCommandRunner
        attr_reader :validator, :retry_executor, :last_result

        def initialize(options = {})
          @validator = CommandValidator.new
          @retry_executor = RetryExecutor.new(validator: @validator)
          @options = {
            require_confirmation: options[:require_confirmation] || false,
            timeout: options[:timeout] || 30,
            max_retries: options[:max_retries] || 2
          }
          @last_result = nil
        end

        # Main method: Run a command from SKILL.md with full validation and safety
        # @param command [String] Raw command from SKILL.md
        # @param context [Hash] Execution context
        # @option context [Array<String>] :urls URLs to substitute
        # @option context [String] :task Original task description
        # @option context [Boolean] :interactive Whether running in interactive mode
        # @return [Hash] Execution result
        def run(command, context = {})
          @last_result = nil

          # Step 1: Pre-process command (URL substitution, etc.)
          processed_command = preprocess_command(command, context)

          # Step 2: Validate and adapt
          validation = @validator.validate_and_adapt(
            processed_command,
            require_confirmation: should_require_confirmation?(context)
          )

          unless validation[:valid]
            @last_result = {
              success: false,
              stage: :validation,
              error: validation[:errors].join("; "),
              original_command: command,
              processed_command: processed_command,
              adaptations: validation[:adaptations]
            }
            return @last_result
          end

          # Step 3: Handle confirmation if needed
          if validation[:requires_confirmation] && context[:interactive]
            confirmed = prompt_for_confirmation(validation[:confirmation_prompt])
            unless confirmed
              @last_result = {
                success: false,
                stage: :confirmation,
                error: "User declined execution",
                command: validation[:command]
              }
              return @last_result
            end
            @validator.confirm_command!(validation)
          end

          # Step 4: Execute with retry and fallback
          execution_options = build_execution_options(context, validation)
          result = @retry_executor.execute_with_retry(
            validation[:command],
            execution_options
          )

          @last_result = enhance_result(result, command, validation, context)
          @last_result
        end

        # Run multiple commands from SKILL.md
        # @param commands [Array<String>] List of commands
        # @param context [Hash] Execution context
        # @return [Array<Hash>] Results for each command
        def run_multiple(commands, context = {})
          results = []

          commands.each do |cmd|
            result = run(cmd, context)
            results << result

            # Stop on first failure unless :continue_on_error is set
            break if !result[:success] && !context[:continue_on_error]
          end

          results
        end

        # Check if a command would be valid without executing it
        # @param command [String] Command to check
        # @param context [Hash] Execution context
        # @return [Hash] Validation result
        def dry_run(command, context = {})
          processed_command = preprocess_command(command, context)
          validation = @validator.validate_and_adapt(processed_command, {})

          {
            valid: validation[:valid],
            would_execute: validation[:command],
            errors: validation[:errors],
            adaptations: validation[:adaptations],
            requires_confirmation: validation[:requires_confirmation],
            missing_tools: detect_missing_tools(processed_command)
          }
        end

        # Generate a report of what would happen for a set of commands
        # @param commands [Array<String>] Commands to analyze
        # @param context [Hash] Execution context
        # @return [String] Human-readable report
        def preview(commands, context = {})
          previews = commands.map { |cmd| dry_run(cmd, context) }

          report = []
          report << "=" * 60
          report << "Command Execution Preview"
          report << "=" * 60

          commands.each_with_index do |original_cmd, i|
            preview = previews[i]
            report << "\nCommand #{i + 1}:"
            report << "  Original: #{original_cmd}"

            if preview[:valid]
              report << "  Status: ✓ Valid"
              report << "  Would execute: #{preview[:would_execute]}"

              if preview[:adaptations].any?
                report << "  Adaptations:"
                preview[:adaptations].each { |a| report << "    - #{a}" }
              end

              if preview[:requires_confirmation]
                report << "  ⚠ Requires user confirmation"
              end
            else
              report << "  Status: ✗ Invalid"
              report << "  Errors: #{preview[:errors].join(', ')}"
            end

            if preview[:missing_tools].any?
              report << "  Missing tools: #{preview[:missing_tools].join(', ')}"
            end
          end

          report << "\n" + "=" * 60
          report.join("\n")
        end

        private

        def preprocess_command(command, context)
          processed = command.dup

          # Substitute URLs
          if context[:urls] && context[:urls].any?
            escaped_url = Shellwords.escape(context[:urls].first)
            processed = processed.gsub("VIDEO_URL", escaped_url)
            processed = processed.gsub(/\$\{?URL\}?/i, escaped_url)
          end

          # Substitute other context variables
          if context[:task]
            processed = processed.gsub("TASK_DESCRIPTION", Shellwords.escape(context[:task]))
          end

          processed
        end

        def should_require_confirmation?(context)
          return true if context[:require_confirmation]
          return true if @options[:require_confirmation]
          return false unless context[:interactive]

          # In interactive mode, require confirmation for risky commands
          true
        end

        def prompt_for_confirmation(prompt_text)
          return true unless prompt_text

          puts "\n#{prompt_text}"
          print "Proceed? (yes/no): "
          response = gets.to_s.strip.downcase
          %w[y yes].include?(response)
        end

        def build_execution_options(context, validation)
          {
            timeout: context[:timeout] || @options[:timeout],
            skip_confirmation: !context[:interactive],
            fallback_commands: generate_fallbacks(validation[:command], context)
          }
        end

        def generate_fallbacks(command, context)
          fallbacks = []

          # Try alternative tools
          if command.include?("yt-dlp")
            fallbacks << command.gsub("yt-dlp", "youtube-dl")
          end

          # Try with different options
          if command.include?("--format")
            fallbacks << command.gsub(/--format\s+\S+/, "")
          end

          fallbacks
        end

        def enhance_result(result, original_command, validation, context)
          enhanced = result.dup

          enhanced[:original_command] = original_command
          enhanced[:adaptations] = validation[:adaptations]
          enhanced[:context] = context.slice(:task, :urls)

          if enhanced[:success]
            enhanced[:summary] = build_success_summary(enhanced)
          else
            enhanced[:summary] = build_failure_summary(enhanced)
          end

          enhanced
        end

        def build_success_summary(result)
          parts = ["✓ Command executed successfully"]
          parts << "  Command: #{result[:command]}"
          parts << "  Time: #{result[:execution_time]&.round(2)}s" if result[:execution_time]

          if result[:note]
            parts << "  Note: #{result[:note]}"
          end

          parts.join("\n")
        end

        def build_failure_summary(result)
          parts = ["✗ Command execution failed"]
          parts << "  Stage: #{result[:stage]}"
          parts << "  Error: #{result[:error]}"

          if result[:attempts_made]
            parts << "  Attempts: #{result[:attempts_made]}"
          end

          parts.join("\n")
        end

        def detect_missing_tools(command)
          tools = []
          CommandValidator::EXECUTABLE_PATTERNS.each do |pattern, info|
            if command.match?(pattern)
              check_cmd = info[:check]
              tools << info[:name] unless system("#{check_cmd} > /dev/null 2>&1")
            end
          end
          tools
        end
      end
    end
  end
end
