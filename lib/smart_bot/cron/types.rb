# frozen_string_literal: true

module SmartBot
  module Cron
    class Schedule
      attr_reader :kind, :at_ms, :every_ms, :expr, :tz

      def initialize(kind:, at_ms: nil, every_ms: nil, expr: nil, tz: nil)
        @kind = kind
        @at_ms = at_ms
        @every_ms = every_ms
        @expr = expr
        @tz = tz
      end

      def to_h
        {
          kind: @kind,
          atMs: @at_ms,
          everyMs: @every_ms,
          expr: @expr,
          tz: @tz
        }
      end
    end

    class Payload
      attr_reader :kind, :message, :deliver, :channel, :to

      def initialize(kind: "agent_turn", message: "", deliver: false, channel: nil, to: nil)
        @kind = kind
        @message = message
        @deliver = deliver
        @channel = channel
        @to = to
      end

      def to_h
        {
          kind: @kind,
          message: @message,
          deliver: @deliver,
          channel: @channel,
          to: @to
        }
      end
    end

    class JobState
      attr_accessor :next_run_at_ms, :last_run_at_ms, :last_status, :last_error

      def initialize(next_run_at_ms: nil, last_run_at_ms: nil, last_status: nil, last_error: nil)
        @next_run_at_ms = next_run_at_ms
        @last_run_at_ms = last_run_at_ms
        @last_status = last_status
        @last_error = last_error
      end

      def to_h
        {
          nextRunAtMs: @next_run_at_ms,
          lastRunAtMs: @last_run_at_ms,
          lastStatus: @last_status,
          lastError: @last_error
        }
      end
    end

    class Job
      attr_reader :id, :name, :created_at_ms, :updated_at_ms, :delete_after_run
      attr_accessor :enabled, :schedule, :payload, :state

      def initialize(id:, name:, enabled: true, schedule: nil, payload: nil, 
                     state: nil, created_at_ms: nil, updated_at_ms: nil, 
                     delete_after_run: false)
        @id = id
        @name = name
        @enabled = enabled
        @schedule = schedule
        @payload = payload
        @state = state || JobState.new
        @created_at_ms = created_at_ms || (Time.now.to_f * 1000).to_i
        @updated_at_ms = updated_at_ms || @created_at_ms
        @delete_after_run = delete_after_run
      end

      def to_h
        {
          id: @id,
          name: @name,
          enabled: @enabled,
          schedule: @schedule&.to_h,
          payload: @payload&.to_h,
          state: @state&.to_h,
          createdAtMs: @created_at_ms,
          updatedAtMs: @updated_at_ms,
          deleteAfterRun: @delete_after_run
        }
      end
    end
  end
end
