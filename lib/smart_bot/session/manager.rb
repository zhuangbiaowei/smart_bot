# frozen_string_literal: true

require "json"
require "date"
require "time"
require "pathname"

module SmartBot
  module Session
    class Session
      attr_reader :key, :messages, :created_at, :updated_at, :metadata

      def initialize(key, attrs = {})
        @key = key
        @messages = attrs[:messages] || []
        @created_at = attrs[:created_at] || Time.now
        @updated_at = attrs[:updated_at] || Time.now
        @metadata = attrs[:metadata] || {}
      end

      def add_message(role, content, **kwargs)
        msg = {
          role: role,
          content: content,
          timestamp: Time.now.iso8601
        }.merge(kwargs)
        @messages << msg
        @updated_at = Time.now
      end

      def get_history(max_messages: 50)
        recent = @messages.last(max_messages)
        recent.map { |m| { role: m[:role], content: m[:content] } }
      end

      def clear
        @messages = []
        @updated_at = Time.now
      end
    end

    class Manager
      def initialize(workspace)
        @workspace = workspace
        @sessions_dir = Utils::Helpers.sessions_path
        @cache = {}
      end

      def get_or_create(key)
        return @cache[key] if @cache[key]

        session = load(key) || Session.new(key)
        @cache[key] = session
        session
      end

      def save(session)
        path = session_path(session.key)
        data = {
          _type: "metadata",
          created_at: session.created_at.iso8601,
          updated_at: session.updated_at.iso8601,
          metadata: session.metadata
        }

        File.open(path, "w") do |f|
          f.puts JSON.generate(data)
          session.messages.each do |msg|
            f.puts JSON.generate(msg)
          end
        end
        @cache[session.key] = session
      end

      def delete(key)
        @cache.delete(key)
        path = session_path(key)
        File.delete(path) if File.exist?(path)
      end

      def list_sessions
        sessions = []
        Dir.glob(File.join(@sessions_dir, "*.jsonl")).each do |path|
          begin
            File.open(path) do |f|
              first_line = f.gets
              next unless first_line

              data = JSON.parse(first_line, symbolize_names: true)
              if data[:_type] == "metadata"
                key = File.basename(path, ".jsonl").gsub("_", ":")
                sessions << {
                  key: key,
                  created_at: data[:created_at],
                  updated_at: data[:updated_at]
                }
              end
            end
          rescue => e
            SmartBot.logger.warn "Failed to read session #{path}: #{e.message}"
          end
        end
        sessions.sort_by { |s| s[:updated_at] || "" }.reverse
      end

      private

      def session_path(key)
        safe_key = Utils::Helpers.safe_filename(key.gsub(":", "_"))
        File.join(@sessions_dir, "#{safe_key}.jsonl")
      end

      def load(key)
        path = session_path(key)
        return nil unless File.exist?(path)

        messages = []
        metadata = {}
        created_at = nil

        File.foreach(path) do |line|
          begin
            data = JSON.parse(line, symbolize_names: true)
            if data[:_type] == "metadata"
              metadata = data[:metadata] || {}
              created_at = Time.parse(data[:created_at]) if data[:created_at]
            else
              messages << data
            end
          rescue JSON::ParserError
            next
          end
        end

        Session.new(key, 
          messages: messages, 
          metadata: metadata,
          created_at: created_at || Time.now
        )
      rescue => e
        SmartBot.logger.warn "Failed to load session #{key}: #{e.message}"
        nil
      end
    end
  end
end
