# frozen_string_literal: true

require "json"
require_relative "base"

module SmartBot
  module Tools
    class RunSkillTool < Base
      attr_accessor :origin_channel, :origin_chat_id

      def initialize(orchestrator)
        @orchestrator = orchestrator
        @origin_channel = "cli"
        @origin_chat_id = "direct"
        @default_chain = nil
        @default_parent_skill = nil

        super(
          name: :run_skill,
          description: "Delegate a subtask to a specific skill. Use this when another skill is better suited for the subtask.",
          parameters: {
            type: "object",
            properties: {
              skill_name: { type: "string", description: "Target skill name (directory name in skills/)" },
              task: { type: "string", description: "Task for the delegated skill" },
              parent_skill: { type: "string", description: "Optional current skill name for cycle prevention" },
              chain: {
                type: "string",
                description: "Optional delegation chain, e.g. 'planner>invoice_organizer'"
              },
              max_depth: { type: "integer", description: "Optional max delegation depth override" }
            },
            required: ["skill_name", "task"]
          }
        )
      end

      def set_context(channel, chat_id)
        @origin_channel = channel
        @origin_chat_id = chat_id
      end

      def set_delegation_defaults(chain: nil, parent_skill: nil)
        @default_chain = chain
        @default_parent_skill = parent_skill
      end

      def execute(skill_name:, task:, parent_skill: nil, chain: nil, max_depth: nil)
        result = @orchestrator.delegate(
          skill_name: skill_name,
          task: task,
          parent_skill: parent_skill || @default_parent_skill,
          chain: chain || @default_chain,
          max_depth: max_depth,
          origin_channel: @origin_channel,
          origin_chat_id: @origin_chat_id
        )

        JSON.pretty_generate(result)
      end
    end
  end
end
