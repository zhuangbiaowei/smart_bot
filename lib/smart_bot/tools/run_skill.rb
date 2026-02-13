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
          description: "CRITICAL: Use this tool to delegate tasks to specialized skills. " \
                       "When a user's request matches any skill description in the system prompt, " \
                       "you MUST call this tool instead of handling the task yourself. " \
                       "Examples: YouTube videos -> youtube-summarizer, weather queries -> weather, " \
                       "invoice organization -> invoice-organizer. Pass the user's complete request as the 'task' parameter.",
          parameters: {
            type: "object",
            properties: {
              skill_name: { type: "string", description: "Target skill name from the available skills list (e.g., 'youtube-summarizer', 'weather', 'invoice-organizer')" },
              task: { type: "string", description: "The complete user request to pass to the skill. Include all context, URLs, and requirements from the user's message." },
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
