# frozen_string_literal: true

module SmartBot
  module SkillSystem
    module Execution
      # Handles command execution with retry logic and fallback strategies
      class RetryExecutor
        MAX_RETRIES = 2
        RETRY_DELAY = 1

        attr_reader :validator, :execution_log

        def initialize(validator: nil)
          @validator = validator || CommandValidator.new
          @execution_log = []
        end

        # Execute a command with validation, retry, and fallback
        # @param command [String] Original command from SKILL.md
        # @param options [Hash] Execution options
        # @option options [Boolean] :require_confirmation Ask user before executing
        # @option options [Integer] :timeout Execution timeout in seconds
        # @option options [Array<String>] :fallback_commands Alternative commands to try
        # @option options [Proc] :confirmation_callback Callback for user confirmation
        # @return [Hash] Execution result
        def execute_with_retry(command, options = {})
          @execution_log = []

          result = validate_and_prepare(command, options)
          return result unless result[:valid]

          if result[:requires_confirmation] && !options[:skip_confirmation]
            confirmed = request_confirmation(result, options[:confirmation_callback])
            unless confirmed
              return {
                success: false,
                error: "User declined execution",
                command: result[:command],
                stage: :confirmation
              }
            end
          end

          execute_with_fallbacks(result[:command], options)
        end

        # Execute multiple commands in sequence
        # @param commands [Array<String>] List of commands
        # @param options [Hash] Execution options
        # @return [Array<Hash>] Results for each command
        def execute_pipeline(commands, options = {})
          results = []

          commands.each do |cmd|
            result = execute_with_retry(cmd, options)
            results << result

            break unless result[:success]
          end

          results
        end

        private

        def validate_and_prepare(command, options)
          validation = @validator.validate_and_adapt(command, options)

          unless validation[:valid]
            return {
              success: false,
              valid: false,
              error: validation[:errors].join("; "),
              original_command: command,
              stage: :validation
            }
          end

          validation.merge(success: true)
        end

        def request_confirmation(validation_result, callback)
          if callback && callback.respond_to?(:call)
            callback.call(validation_result[:confirmation_prompt], validation_result[:command])
          else
            @validator.confirm_command!(validation_result)
            true
          end
        end

        def execute_with_fallbacks(command, options)
          fallback_commands = options[:fallback_commands] || []
          timeout = options[:timeout] || 30

          attempts = [command] + fallback_commands
          last_error = nil

          attempts.each_with_index do |cmd, index|
            attempt_result = try_execute(cmd, timeout, index + 1)
            @execution_log << attempt_result

            if attempt_result[:success]
              return format_success_result(attempt_result, index + 1, attempts.length)
            end

            last_error = attempt_result[:error]

            if attempt_result[:retryable] && index < MAX_RETRIES
              sleep(RETRY_DELAY)
              retry_result = try_execute(cmd, timeout, index + 1, true)
              @execution_log << retry_result

              if retry_result[:success]
                return format_success_result(retry_result, index + 1, attempts.length, true)
              end

              last_error = retry_result[:error]
            end
          end

          format_failure_result(last_error, attempts.length)
        end

        def try_execute(command, timeout, attempt_number, is_retry = false)
          start_time = Time.now

          stdout, stderr, status = nil

          begin
            Timeout.timeout(timeout) do
              stdout, stderr, status = Open3.capture3("bash", "-lc", command)
            end

            execution_time = Time.now - start_time

            if status&.success?
              {
                success: true,
                command: command,
                stdout: stdout.to_s,
                stderr: stderr.to_s,
                exit_code: status.exitstatus,
                execution_time: execution_time,
                attempt: attempt_number,
                retry: is_retry
              }
            else
              {
                success: false,
                command: command,
                stdout: stdout.to_s,
                stderr: stderr.to_s,
                exit_code: status&.exitstatus || -1,
                execution_time: execution_time,
                attempt: attempt_number,
                retry: is_retry,
                retryable: retryable_error?(stderr.to_s + stdout.to_s),
                error: "Exit code #{status&.exitstatus || 'unknown'}: #{stderr.to_s[0..200]}"
              }
            end
          rescue Timeout::Error
            {
              success: false,
              command: command,
              stdout: "",
              stderr: "Command timed out after #{timeout}s",
              exit_code: -1,
              execution_time: timeout,
              attempt: attempt_number,
              retry: is_retry,
              retryable: true,
              error: "Timeout after #{timeout} seconds"
            }
          rescue => e
            {
              success: false,
              command: command,
              stdout: "",
              stderr: e.message,
              exit_code: -1,
              execution_time: Time.now - start_time,
              attempt: attempt_number,
              retry: is_retry,
              retryable: retryable_error?(e.message),
              error: e.message
            }
          end
        end

        def retryable_error?(error_message)
          error_text = error_message.to_s.downcase

          retryable_patterns = [
            /timeout/i,
            /connection.*refused/i,
            /network.*unreachable/i,
            /temporarily.*unavailable/i,
            /rate.*limit/i,
            /too.*many.*requests/i,
            /resource.*busy/i,
            /try.*again/i,
            /retry/i
          ]

          non_retryable_patterns = [
            /permission.*denied/i,
            /not.*found/i,
            /invalid.*argument/i,
            /bad.*request/i,
            /unauthorized/i,
            /forbidden/i,
            /does not exist/i,
            /no such file/i
          ]

          return false if non_retryable_patterns.any? { |p| error_text =~ p }
          retryable_patterns.any? { |p| error_text =~ p }
        end

        def format_success_result(attempt_result, attempt_number, total_attempts, was_retried = false)
          result = {
            success: true,
            command: attempt_result[:command],
            stdout: attempt_result[:stdout],
            stderr: attempt_result[:stderr],
            exit_code: attempt_result[:exit_code],
            execution_time: attempt_result[:execution_time],
            attempts_made: attempt_number,
            total_attempts: total_attempts
          }

          if was_retried
            result[:note] = "Succeeded on retry"
          elsif attempt_number > 1
            result[:note] = "Succeeded using fallback command ##{attempt_number}"
          end

          result
        end

        def format_failure_result(last_error, total_attempts)
          {
            success: false,
            error: last_error || "All execution attempts failed",
            attempts_made: total_attempts,
            total_attempts: total_attempts,
            execution_log: @execution_log,
            stage: :execution
          }
        end
      end
    end
  end
end
