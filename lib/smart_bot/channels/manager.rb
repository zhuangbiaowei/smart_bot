# frozen_string_literal: true

module SmartBot
  module Channels
    class Manager
      attr_reader :channels, :bus

      def initialize(config, bus)
        @config = config
        @bus = bus
        @channels = {}
        @dispatch_thread = nil
        
        init_channels
      end

      def init_channels
        # Telegram
        if @config.channels[:telegram]&.dig(:enabled)
          begin
            tg_config = @config.channels[:telegram].merge(
              groq_api_key: @config.providers[:groq]&.dig(:api_key)
            )
            @channels[:telegram] = TelegramChannel.new(tg_config, @bus)
            SmartBot.logger.info "Telegram channel enabled"
          rescue => e
            SmartBot.logger.warn "Telegram channel not available: #{e.message}"
          end
        end

        # WhatsApp (placeholder)
        if @config.channels[:whatsapp]&.dig(:enabled)
          SmartBot.logger.warn "WhatsApp channel not yet implemented"
        end
      end

      def start_all
        return if @channels.empty?

        # Start outbound dispatcher
        @dispatch_thread = Thread.new { dispatch_outbound }

        # Start all channels
        threads = @channels.map do |name, channel|
          SmartBot.logger.info "Starting #{name} channel..."
          Thread.new { channel.start }
        end

        threads.each(&:join)
      end

      def stop_all
        SmartBot.logger.info "Stopping all channels..."
        @dispatch_thread&.kill

        @channels.each do |name, channel|
          begin
            channel.stop
            SmartBot.logger.info "Stopped #{name} channel"
          rescue => e
            SmartBot.logger.error "Error stopping #{name}: #{e.message}"
          end
        end
      end

      def enabled_channels
        @channels.keys
      end

      def get_channel(name)
        @channels[name.to_sym]
      end

      private

      def dispatch_outbound
        loop do
          message = @bus.consume_outbound(timeout: 1)
          next unless message

          channel = @channels[message.channel.to_sym]
          if channel
            begin
              channel.send(message)
            rescue => e
              SmartBot.logger.error "Error sending to #{message.channel}: #{e.message}"
            end
          else
            SmartBot.logger.warn "Unknown channel: #{message.channel}"
          end
        end
      end
    end
  end
end
