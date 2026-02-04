# frozen_string_literal: true

require "thread"
require "timeout"

module SmartBot
  module Bus
    class Queue
      attr_reader :inbound, :outbound

      def initialize
        @inbound = ::Queue.new
        @outbound = ::Queue.new
        @outbound_subscribers = {}
        @running = false
        @mutex = Mutex.new
      end

      def publish_inbound(message)
        @inbound << message
        SmartBot.logger.debug "Published inbound message to #{message.channel}"
      end

      def consume_inbound(timeout: 1)
        @inbound.pop(true)
      rescue ThreadError
        sleep(timeout)
        nil
      end

      def publish_outbound(message)
        @outbound << message
        dispatch_outbound(message)
        SmartBot.logger.debug "Published outbound message to #{message.channel}"
      end

      def consume_outbound(timeout: 1)
        @outbound.pop(true)
      rescue ThreadError
        sleep(timeout)
        nil
      end

      def subscribe_outbound(channel, &block)
        @mutex.synchronize do
          @outbound_subscribers[channel] ||= []
          @outbound_subscribers[channel] << block
        end
      end

      def dispatch_outbound(message)
        subscribers = @mutex.synchronize { @outbound_subscribers[message.channel] || [] }
        subscribers.each do |callback|
          begin
            callback.call(message)
          rescue => e
            SmartBot.logger.error "Error dispatching to #{message.channel}: #{e.message}"
          end
        end
      end

      def start_dispatch_loop
        @running = true
        Thread.new do
          while @running
            message = consume_outbound
            dispatch_outbound(message) if message
          end
        end
      end

      def stop
        @running = false
      end

      def inbound_size
        @inbound.size
      end

      def outbound_size
        @outbound.size
      end
    end
  end
end
