# frozen_string_literal: true

require "securerandom"

module SmartBot
  module Tools
    class SpawnTool < Base
      attr_accessor :origin_channel, :origin_chat_id

      def initialize(manager)
        @manager = manager
        @origin_channel = "cli"
        @origin_chat_id = "direct"
        
        super(
          name: :spawn,
          description: "Spawn a subagent to handle a task in the background. " \
                       "Use this for complex or time-consuming tasks that can run independently. " \
                       "The subagent will complete the task and report back when done.",
          parameters: {
            type: "object",
            properties: {
              task: { type: "string", description: "The task for the subagent to complete" },
              label: { type: "string", description: "Optional short label for the task (for display)" }
            },
            required: ["task"]
          }
        )
      end

      def set_context(channel, chat_id)
        @origin_channel = channel
        @origin_chat_id = chat_id
      end

      def execute(task:, label: nil)
        @manager.spawn(
          task: task,
          label: label,
          origin_channel: @origin_channel,
          origin_chat_id: @origin_chat_id
        )
      end
    end
  end
end
