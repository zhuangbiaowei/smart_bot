# frozen_string_literal: true

require "date"

module SmartBot
  module Bus
    class InboundMessage
      attr_reader :channel, :sender_id, :chat_id, :content, :timestamp, :media, :metadata

      def initialize(channel:, sender_id:, chat_id:, content:, media: [], metadata: {})
        @channel = channel
        @sender_id = sender_id
        @chat_id = chat_id
        @content = content
        @timestamp = Time.now
        @media = media
        @metadata = metadata
      end

      def session_key
        "#{@channel}:#{@chat_id}"
      end
    end

    class OutboundMessage
      attr_reader :channel, :chat_id, :content, :reply_to, :media, :metadata

      def initialize(channel:, chat_id:, content:, reply_to: nil, media: [], metadata: {})
        @channel = channel
        @chat_id = chat_id
        @content = content
        @reply_to = reply_to
        @media = media
        @metadata = metadata
      end
    end
  end
end
