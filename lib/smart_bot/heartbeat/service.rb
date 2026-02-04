# frozen_string_literal: true

require "pathname"

module SmartBot
  module Heartbeat
    HEARTBEAT_INTERVAL_S = 30 * 60 # 30 minutes
    HEARTBEAT_PROMPT = "Read HEARTBEAT.md in your workspace (if it exists).\n" \
                       "Follow any instructions or tasks listed there.\n" \
                       "If nothing needs attention, reply with just: HEARTBEAT_OK"
    HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"

    class Service
      def initialize(workspace:, on_heartbeat: nil, interval_s: HEARTBEAT_INTERVAL_S, enabled: true)
        @workspace = Pathname.new(workspace)
        @on_heartbeat = on_heartbeat
        @interval_s = interval_s
        @enabled = enabled
        @running = false
        @timer_thread = nil
      end

      def heartbeat_file
        @workspace / "HEARTBEAT.md"
      end

      def read_heartbeat_file
        heartbeat_file.exist? ? heartbeat_file.read(encoding: "UTF-8") : nil
      end

      def heartbeat_empty?(content)
        return true unless content
        
        skip_patterns = ["- [ ]", "* [ ]", "- [x]", "* [x]"]
        
        content.each_line do |line|
          stripped = line.strip
          next if stripped.empty?
          next if stripped.start_with?("#", "<!--")
          next if skip_patterns.include?(stripped)
          return false
        end
        true
      end

      def start
        return unless @enabled

        @running = true
        @timer_thread = Thread.new { run_loop }
        SmartBot.logger.info "Heartbeat started (every #{@interval_s}s)"
      end

      def stop
        @running = false
        @timer_thread&.kill
      end

      def trigger_now
        return nil unless @on_heartbeat
        @on_heartbeat.call(HEARTBEAT_PROMPT)
      end

      private

      def run_loop
        while @running
          sleep(@interval_s)
          tick if @running
        end
      end

      def tick
        content = read_heartbeat_file
        
        if heartbeat_empty?(content)
          SmartBot.logger.debug "Heartbeat: no tasks (HEARTBEAT.md empty)"
          return
        end

        SmartBot.logger.info "Heartbeat: checking for tasks..."

        return unless @on_heartbeat

        begin
          response = @on_heartbeat.call(HEARTBEAT_PROMPT)
          
          if response&.upcase&.gsub("_", "")&.include?(HEARTBEAT_OK_TOKEN)
            SmartBot.logger.info "Heartbeat: OK (no action needed)"
          else
            SmartBot.logger.info "Heartbeat: completed task"
          end
        rescue => e
          SmartBot.logger.error "Heartbeat execution failed: #{e.message}"
        end
      end
    end
  end
end
