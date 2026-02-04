# frozen_string_literal: true

module SmartBot
  module Channels
    class Base
      attr_reader :name, :config, :bus

      def initialize(name, config, bus)
        @name = name
        @config = config
        @bus = bus
        @running = false
      end

      def start
        raise NotImplementedError
      end

      def stop
        raise NotImplementedError
      end

      def send(message)
        raise NotImplementedError
      end

      def allowed?(sender_id)
        allow_list = @config[:allow_from] || []
        return true if allow_list.empty?
        
        sender_str = sender_id.to_s
        return true if allow_list.include?(sender_str)
        return true if sender_str.include?("|") && sender_str.split("|").any? { |p| allow_list.include?(p) }
        false
      end

      def running?
        @running
      end

      protected

      def handle_message(sender_id:, chat_id:, content:, media: [], metadata: {})
        return unless allowed?(sender_id)

        msg = Bus::InboundMessage.new(
          channel: @name,
          sender_id: sender_id.to_s,
          chat_id: chat_id.to_s,
          content: content,
          media: media,
          metadata: metadata
        )
        @bus.publish_inbound(msg)
      end
    end
  end
end
