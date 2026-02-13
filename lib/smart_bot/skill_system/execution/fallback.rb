# frozen_string_literal: true

require_relative "executor"

module SmartBot
  module SkillSystem
    # Fallback state machine for handling execution failures
    class FallbackStateMachine
      STATES = %i[selected running success retryable_failure
                  fatal_failure fallback exit].freeze

      MAX_RETRIES = 1

      attr_reader :plan, :executor, :observer, :state

      def initialize(plan:, executor:, observer: nil)
        @plan = plan
        @executor = executor
        @observer = observer
        @state = :selected
        @retry_count = 0
        @current_skill_index = 0
        @result = nil
        @last_error = nil
      end

      def run(context: {})
        transition_to(:running)

        loop do
          case @state
          when :running
            execute_current_skill(context)
          when :retryable_failure
            handle_retryable_failure
          when :fatal_failure
            handle_fatal_failure
          when :fallback
            execute_fallback(context)
          when :success, :exit
            break
          end
        end

        @result
      end

      private

      def execute_current_skill(context)
        skill = current_skill
        return transition_to(:fallback) unless skill

        result = @executor.execute_skill(skill, @plan.parameters, context)

        if result.success?
          @result = result
          transition_to(:success)
        elsif retryable?(result)
          @last_error = result.error
          transition_to(:retryable_failure)
        else
          @last_error = result.error
          transition_to(:fatal_failure)
        end
      end

      def handle_retryable_failure
        if @retry_count < MAX_RETRIES
          @retry_count += 1
          @observer&.retry_attempted(current_skill, @retry_count)
          transition_to(:running)
        else
          transition_to(:fallback)
        end
      end

      def handle_fatal_failure
        transition_to(:fallback)
      end

      def execute_fallback(context)
        fallback_skill = next_fallback_skill

        if fallback_skill == :generic_tools
          @result = execute_generic_tools(context)
          transition_to(@result.success? ? :success : :exit)
        elsif fallback_skill
          @current_skill_index = @plan.skills.index(fallback_skill) || @current_skill_index + 1
          @retry_count = 0
          transition_to(:running)
        else
          @result = ExecutionResult.failure(
            skill: nil,
            error: "All fallback options exhausted. Last error: #{@last_error}"
          )
          transition_to(:exit)
        end
      end

      def current_skill
        @plan.skills[@current_skill_index]
      end

      def next_fallback_skill
        fallback_index = @plan.fallback_chain.index do |s|
          s == :generic_tools ||
            (@plan.skills.index(s) || -1) > @current_skill_index
        end

        return nil unless fallback_index

        @plan.fallback_chain[fallback_index..].find do |s|
          s == :generic_tools || !@plan.skills[0..@current_skill_index].include?(s)
        end
      end

      def retryable?(result)
        error = result.error.to_s.downcase

        non_retryable = [
          /permission denied/i,
          /not found/i,
          /invalid.*format/i,
          /capability.*not.*available/i
        ]

        !non_retryable.any? { |p| error =~ p }
      end

      def execute_generic_tools(_context)
        ExecutionResult.failure(
          skill: nil,
          error: "All skills failed. Last error: #{@last_error}"
        )
      end

      def transition_to(new_state)
        old_state = @state
        @state = new_state
        @observer&.state_transition(old_state, new_state)
      end
    end
  end
end
