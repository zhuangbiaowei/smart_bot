# frozen_string_literal: true

require "net/http"
require "json"
require "uri"

module SmartBot
  module Channels
    class TelegramChannel < Base
      API_BASE = "https://api.telegram.org/bot"

      def initialize(config, bus)
        super(:telegram, config, bus)
        @token = config[:token]
        @groq_api_key = config[:groq_api_key]
        @chat_ids = {}
        @offset = 0
        @mutex = Mutex.new
      end

      def start
        return unless @token

        @running = true
        SmartBot.logger.info "Starting Telegram bot..."

        # Get bot info
        me = api_request("getMe")
        if me&.dig("ok")
          username = me.dig("result", "username")
          SmartBot.logger.info "Telegram bot @#{username} connected"
        end

        # Start polling
        Thread.new { poll_loop }
      end

      def stop
        @running = false
        SmartBot.logger.info "Stopping Telegram bot..."
      end

      def send(message)
        return unless @token

        chat_id = message.chat_id.to_i
        text = message.content
        
        # Truncate if too long
        text = text[0..4096] if text.length > 4096

        api_request("sendMessage", {
          chat_id: chat_id,
          text: text,
          parse_mode: "HTML"
        })
      rescue => e
        SmartBot.logger.error "Error sending Telegram message: #{e.message}"
        # Fallback to plain text
        begin
          api_request("sendMessage", {
            chat_id: chat_id,
            text: message.content[0..4096]
          })
        rescue => e2
          SmartBot.logger.error "Fallback failed: #{e2.message}"
        end
      end

      private

      def poll_loop
        while @running
          begin
            updates = api_request("getUpdates", { offset: @offset, limit: 100 })
            
            if updates&.dig("ok")
              updates["result"].each do |update|
                process_update(update)
                @offset = update["update_id"] + 1
              end
            end

            sleep(1)
          rescue => e
            SmartBot.logger.error "Telegram poll error: #{e.message}"
            sleep(5)
          end
        end
      end

      def process_update(update)
        message = update["message"]
        return unless message

        user = message["from"]
        chat = message["chat"]
        return unless user && chat

        sender_id = user["id"].to_s
        sender_id += "|#{user['username']}" if user["username"]
        chat_id = chat["id"]

        @mutex.synchronize { @chat_ids[sender_id] = chat_id }

        # Build content from text and/or media
        content_parts = []
        media_paths = []

        content_parts << message["text"] if message["text"]
        content_parts << message["caption"] if message["caption"]

        # Handle voice transcription
        if message["voice"] && @groq_api_key
          begin
            transcription = transcribe_voice(message["voice"])
            if transcription
              content_parts << "[transcription: #{transcription}]"
            end
          rescue => e
            SmartBot.logger.error "Voice transcription failed: #{e.message}"
          end
        end

        content = content_parts.compact.join("\n")
        content = "[empty message]" if content.empty?

        handle_message(
          sender_id: sender_id,
          chat_id: chat_id,
          content: content,
          media: media_paths,
          metadata: {
            message_id: message["message_id"],
            user_id: user["id"],
            username: user["username"],
            first_name: user["first_name"]
          }
        )
      end

      def api_request(method, params = {})
        uri = URI.parse("#{API_BASE}#{@token}/#{method}")
        
        if params.empty?
          response = Net::HTTP.get_response(uri)
        else
          req = Net::HTTP::Post.new(uri)
          req.content_type = "application/json"
          req.body = params.to_json
          response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: true) do |http|
            http.request(req)
          end
        end

        JSON.parse(response.body) if response.is_a?(Net::HTTPSuccess)
      end

      def transcribe_voice(voice)
        # Simplified - would download voice file and call Groq API
        nil
      end
    end
  end
end
