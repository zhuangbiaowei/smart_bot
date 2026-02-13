# frozen_string_literal: true

module SmartBot
  module SkillSystem
    # Result object for skill execution
    class ExecutionResult
      attr_reader :skill, :value, :error, :metadata

      def self.success(skill:, value:, metadata: {})
        new(skill: skill, value: value, metadata: metadata, success: true)
      end

      def self.failure(skill:, error:, metadata: {})
        new(skill: skill, error: error, metadata: metadata, success: false)
      end

      def initialize(skill:, value: nil, error: nil, metadata: {}, success: false)
        @skill = skill
        @value = value
        @error = error
        @metadata = metadata
        @success = success
      end

      def success?
        @success
      end

      def failure?
        !@success
      end

      def to_h
        {
          skill: @skill&.name,
          success: @success,
          value: @value,
          error: @error,
          metadata: @metadata
        }
      end
    end

    # Execution observer for logging and metrics
    class ExecutionObserver
      def plan_started(plan)
        logger.info "[SkillSystem] Plan started: #{plan.skills.map(&:name).join(', ')}"
      end

      def plan_completed(plan, result)
        logger.info "[SkillSystem] Plan completed: success=#{result.success?}"
      end

      def plan_failed(plan, error)
        logger.error "[SkillSystem] Plan failed: #{error.message}"
      end

      def skill_started(skill, parameters)
        logger.info "[SkillSystem] Skill started: #{skill.name}"
      end

      def skill_completed(skill, result)
        logger.info "[SkillSystem] Skill completed: #{skill.name}, success=#{result.success?}"
      end

      def skill_failed(skill, error)
        logger.error "[SkillSystem] Skill failed: #{skill.name}, error=#{error.message}"
      end

      def state_transition(from, to)
        logger.debug "[SkillSystem] State transition: #{from} -> #{to}"
      end

      def retry_attempted(skill, attempt)
        logger.info "[SkillSystem] Retry attempt #{attempt} for #{skill.name}"
      end

      def repair_attempted(skill, attempt)
        logger.info "[SkillSystem] Repair attempt #{attempt} for #{skill.name}"
      end

      private

      def logger
        @logger ||= defined?(SmartBot) && SmartBot.respond_to?(:logger) ? SmartBot.logger : default_logger
      end

      def default_logger
        require "logger"
        Logger.new(File::NULL)
      end

      def repair_succeeded(skill, attempt)
        SmartBot.logger&.info "[SkillSystem] Repair succeeded for #{skill.name} after #{attempt} attempts"
      end

      def repair_failed(skill, attempt, reason)
        SmartBot.logger&.warn "[SkillSystem] Repair failed for #{skill.name} (attempt #{attempt}): #{reason}"
      end

      def repair_skipped(skill, result, reason)
        SmartBot.logger&.info "[SkillSystem] Repair skipped for #{skill.name}: #{reason}"
      end

      def repair_no_improvement(skill, attempt)
        SmartBot.logger&.warn "[SkillSystem] Repair no improvement for #{skill.name} (attempt #{attempt})"
      end
    end
  end
end
