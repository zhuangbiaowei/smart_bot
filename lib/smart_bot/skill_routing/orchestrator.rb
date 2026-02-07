# frozen_string_literal: true

require "json"
require "pathname"

module SmartBot
  module SkillRouting
    # Minimal orchestrator for delegated skill execution.
    # It validates skill availability and prevents runaway delegation.
    class Orchestrator
      DEFAULT_MAX_DEPTH = 2

      def initialize(workspace:, subagent_manager:, max_depth: DEFAULT_MAX_DEPTH)
        @workspace = Pathname.new(workspace)
        @subagent_manager = subagent_manager
        @max_depth = max_depth
      end

      def delegate(skill_name:, task:, parent_skill: nil, chain: nil, max_depth: nil, origin_channel: "cli", origin_chat_id: "direct")
        normalized_skill = normalize_name(skill_name)
        return error("skill_name is required") if normalized_skill.empty?
        return error("task is required") if task.to_s.strip.empty?

        effective_max_depth = (max_depth || @max_depth).to_i
        effective_max_depth = DEFAULT_MAX_DEPTH if effective_max_depth <= 0

        current_chain = parse_chain(chain)
        current_chain << normalize_name(parent_skill) unless parent_skill.to_s.strip.empty?
        current_chain = current_chain.reject(&:empty?)

        if current_chain.include?(normalized_skill)
          return error("Delegation cycle detected: #{(current_chain + [normalized_skill]).join(' -> ')}")
        end

        if current_chain.length >= effective_max_depth
          return error("Delegation depth limit reached (max_depth=#{effective_max_depth})")
        end

        skill_path = find_skill_path(normalized_skill)
        return error("Skill not found: #{normalized_skill}") unless skill_path

        next_chain = current_chain + [normalized_skill]
        delegated_task = build_delegated_task(skill_name: normalized_skill, skill_path: skill_path, task: task, chain: next_chain)

        spawn_result = @subagent_manager.spawn(
          task: delegated_task,
          label: "skill:#{normalized_skill}",
          origin_channel: origin_channel,
          origin_chat_id: origin_chat_id
        )

        {
          status: "delegated",
          skill: normalized_skill,
          chain: next_chain,
          max_depth: effective_max_depth,
          skill_file: skill_path.to_s,
          message: spawn_result
        }
      rescue => e
        error("Delegation failed: #{e.message}")
      end

      private

      def build_delegated_task(skill_name:, skill_path:, task:, chain:)
        <<~TASK
          Delegated Skill Task
          - target_skill: #{skill_name}
          - call_chain: #{chain.join(" -> ")}
          - skill_file: #{skill_path}

          Execution requirements:
          1. Read and follow the instructions in the skill file above.
          2. Complete the task below using available tools.
          3. Return a concise result, plus blockers if any.

          Task:
          #{task}
        TASK
      end

      def find_skill_path(skill_name)
        local_path = @workspace / "skills" / skill_name / "SKILL.md"
        return local_path if local_path.exist?

        built_in_path = Pathname.new(File.expand_path("~/smart_ai/smart_bot/skills/#{skill_name}/SKILL.md"))
        return built_in_path if built_in_path.exist?

        nil
      end

      def parse_chain(chain)
        return [] if chain.nil?

        case chain
        when Array
          chain.map { |item| normalize_name(item) }
        when String
          chain.split(/\s*(?:->|>)\s*/).map { |item| normalize_name(item) }
        else
          []
        end
      end

      def normalize_name(name)
        name.to_s.strip.downcase.gsub(/[^a-z0-9_]/, "_").gsub(/_+/, "_").gsub(/^_|_$/, "")
      end

      def error(message)
        { status: "error", error: message }
      end
    end
  end
end
