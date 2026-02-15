# frozen_string_literal: true

# Load legacy Skill system first (required for loading Ruby skills)
begin
  require_relative "skill"
rescue LoadError
  # Legacy skill system not available
end

require_relative "skill_system/core/value_objects"
require_relative "skill_system/core/metadata"
require_relative "skill_system/core/skill_package"
require_relative "skill_system/core/registry"
require_relative "skill_system/adapters/openclaw_adapter"
require_relative "skill_system/core/loader"
require_relative "skill_system/routing/router"
require_relative "skill_system/routing/scorer"
require_relative "skill_system/routing/activation_plan"
require_relative "skill_system/execution/result"
require_relative "skill_system/execution/openclaw_executor"
require_relative "skill_system/execution/sandbox"
require_relative "skill_system/execution/executor"
require_relative "skill_system/execution/fallback"
require_relative "skill_system/execution/repair_loop"
require_relative "skill_system/installer"

module SmartBot
  module SkillSystem
    class << self
      def configure
        @config ||= Configuration.new
        yield @config if block_given?
        @config
      end

      def registry
        @registry ||= SkillRegistry.new
      end

      def loader
        @loader ||= UnifiedLoader.new(
          workspace: config.workspace,
          repo_root: config.repo_root,
          home: config.home
        )
      end

      def router
        @router ||= Router.new(registry: registry)
      end

      def executor
        @executor ||= SkillExecutor.new
      end

      def config
        @config ||= Configuration.new
      end

      def load_all
        skills = loader.load_all
        skills.each { |skill| registry.register(skill) }
        router.refresh_semantic_index if router.semantic_index
        registry.stats
      end

      def route(query, context: {}, history: [], stats: {})
        router.route(
          query: query,
          context: context,
          history: history,
          stats: stats
        )
      end

      def execute(plan, context: {}, enable_repair: true, repair_confirmation_callback: nil)
        if plan.parallel_groups.size > 1 || plan.fallback_chain.any?
          fsm = FallbackStateMachine.new(
            plan: plan,
            executor: executor
          )
          fsm.run(context: context)
        elsif enable_repair && plan.primary_skill
          # Use repair loop for single skill with auto-repair
          repair_loop = RepairLoop.new(
            executor: executor,
            repair_confirmation_callback: repair_confirmation_callback
          )
          repair_loop.execute_with_repair(
            plan.primary_skill,
            plan.parameters,
            context
          )
        else
          executor.execute(plan, context: context)
        end
      end

      def run(query, context: {}, history: [], stats: {})
        plan = route(query, context: context, history: history, stats: stats)

        if plan.empty?
          return ExecutionResult.failure(
            skill: nil,
            error: "No matching skills found for: #{query}"
          )
        end

        execute(plan, context: context)
      end

      def reset!
        @registry = nil
        @loader = nil
        @router = nil
        @executor = nil
      end
    end

    class Configuration
      attr_accessor :workspace, :repo_root, :home
      attr_accessor :semantic_top_k, :selection_threshold, :max_parallel_skills
      attr_accessor :max_retry_per_path, :max_skill_delegate_depth
      attr_accessor :default_timeout, :max_repair_attempts

      def initialize
        @workspace = File.expand_path("~/.smart_bot/workspace")
        @repo_root = Dir.pwd
        @home = Dir.home

        @semantic_top_k = 3
        @selection_threshold = 0.65
        @max_parallel_skills = 2
        @max_retry_per_path = 1
        @max_skill_delegate_depth = 2
        @default_timeout = 120
        @max_repair_attempts = 2
      end
    end
  end
end
