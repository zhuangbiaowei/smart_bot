# frozen_string_literal: true

require "pathname"
require "fileutils"
require "date"

module SmartBot
  module Utils
    module Helpers
      extend self

      def ensure_dir(path)
        path = Pathname.new(path)
        FileUtils.mkdir_p(path) unless path.exist?
        path
      end

      def data_path
        ensure_dir(File.expand_path("~/.smart_bot"))
      end

      def workspace_path(workspace = nil)
        path = workspace ? File.expand_path(workspace) : File.expand_path("~/.smart_bot/workspace")
        ensure_dir(path)
      end

      def sessions_path
        ensure_dir(File.join(data_path, "sessions"))
      end

      def memory_path(workspace = nil)
        ws = workspace || workspace_path
        ensure_dir(File.join(ws, "memory"))
      end

      def skills_path(workspace = nil)
        ws = workspace || workspace_path
        ensure_dir(File.join(ws, "skills"))
      end

      def today_date
        Date.today.strftime("%Y-%m-%d")
      end

      def timestamp
        Time.now.iso8601
      end

      def truncate_string(str, max_len = 100, suffix = "...")
        return str if str.length <= max_len
        str[0...max_len - suffix.length] + suffix
      end

      def safe_filename(name)
        name.gsub(/[<>:"\/\\|?*]/, "_").strip
      end

      def parse_session_key(key)
        parts = key.split(":", 2)
        raise ArgumentError, "Invalid session key: #{key}" unless parts.length == 2
        parts
      end

      def camel_to_snake(name)
        name.gsub(/([A-Z])/, '_\1').downcase.sub(/^_/, '')
      end

      def snake_to_camel(name)
        parts = name.split("_")
        parts[0] + parts[1..].map(&:capitalize).join
      end
    end
  end
end
