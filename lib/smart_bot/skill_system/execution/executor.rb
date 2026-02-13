# frozen_string_literal: true

require_relative "sandbox"

module SmartBot
  module SkillSystem
    # Main skill execution coordinator
    class SkillExecutor
      attr_reader :sandbox, :observer

      def initialize(sandbox: nil, observer: nil)
        @sandbox = sandbox || Sandbox.new
        @observer = observer || ExecutionObserver.new
      end

      def execute(plan, context: {})
        @observer.plan_started(plan)

        result = execute_plan(plan, context)

        @observer.plan_completed(plan, result)
        result
      rescue => e
        @observer.plan_failed(plan, e)
        raise
      end

      def execute_skill(skill, parameters, context = {})
        @observer.skill_started(skill, parameters)

        # Check permissions with skill type for OpenClaw compatibility
        unless @sandbox.check_permissions(skill.metadata.permissions, skill.type)
          result = ExecutionResult.failure(
            skill: skill,
            error: "Permission check failed"
          )
          @observer.skill_failed(skill, StandardError.new(result.error))
          return result
        end

        # Execute in sandbox
        result = @sandbox.execute(skill, parameters, context)

        if result.success?
          @observer.skill_completed(skill, result)
        else
          @observer.skill_failed(skill, StandardError.new(result.error))
        end

        result
      rescue => e
        @observer.skill_failed(skill, e)
        ExecutionResult.failure(skill: skill, error: e.message)
      end

      private

      def execute_plan(plan, context)
        results = []

        plan.parallel_groups.each do |group|
          if group.size == 1
            result = execute_skill(group.first, plan.parameters, context)
            results << result

            # Stop if failed
            break if result.failure?
          else
            group_results = execute_parallel(group, plan.parameters, context)
            results.concat(group_results)

            # Stop if any failed
            break if group_results.any?(&:failure?)
          end
        end

        ExecutionResult.new(
          success: results.all?(&:success?),
          value: results.map(&:value),
          metadata: { results: results.map(&:to_h) }
        )
      end

      def execute_parallel(skills, parameters, context)
        # Simple sequential execution for now
        # Could be enhanced with threads/fibers
        skills.map do |skill|
          execute_skill(skill, parameters, context)
        end
      end
    end
  end
end
