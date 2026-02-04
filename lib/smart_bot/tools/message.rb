# frozen_string_literal: true

module SmartBot
  module Tools
    class MessageTool < Base
      attr_accessor :send_callback, :default_channel, :default_chat_id

      def initialize(send_callback: nil, default_channel: "", default_chat_id: "")
        @send_callback = send_callback
        @default_channel = default_channel
        @default_chat_id = default_chat_id
        
        super(
          name: :message,
          description: "Send a message to the user. Use this when you want to communicate something.",
          parameters: {
            type: "object",
            properties: {
              content: { type: "string", description: "The message content to send" },
              channel: { type: "string", description: "Optional: target channel (telegram, discord, etc.)" },
              chat_id: { type: "string", description: "Optional: target chat/user ID" }
            },
            required: ["content"]
          }
        )
      end

      def set_context(channel, chat_id)
        @default_channel = channel
        @default_chat_id = chat_id
      end

      def execute(content:, channel: nil, chat_id: nil)
        target_channel = channel || @default_channel
        target_chat_id = chat_id || @default_chat_id

        return "Error: No target channel/chat specified" if target_channel.empty? || target_chat_id.empty?
        return "Error: Message sending not configured" unless @send_callback

        msg = Bus::OutboundMessage.new(
          channel: target_channel,
          chat_id: target_chat_id,
          content: content
        )

        begin
          @send_callback.call(msg)
          "Message sent to #{target_channel}:#{target_chat_id}"
        rescue => e
          "Error sending message: #{e.message}"
        end
      end
    end
  end
end
