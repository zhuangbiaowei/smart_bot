# frozen_string_literal: true

require "json"
require "fileutils"
require "securerandom"

module SmartBot
  module Cron
    class Service
      def initialize(store_path, on_job: nil)
        @store_path = store_path
        @on_job = on_job
        @store = nil
        @mutex = Mutex.new
        @running = false
        @timer_thread = nil
      end

      def start
        @mutex.synchronize do
          @running = true
          load_store
          recompute_next_runs
          save_store
          arm_timer
        end
        SmartBot.logger.info "Cron service started with #{@store[:jobs]&.size || 0} jobs"
      end

      def stop
        @mutex.synchronize do
          @running = false
          @timer_thread&.kill
        end
      end

      def list_jobs(include_disabled: false)
        load_store
        jobs = @store[:jobs] || []
        jobs = jobs.select(&:enabled) unless include_disabled
        jobs.sort_by { |j| j.state.next_run_at_ms || Float::INFINITY }
      end

      def add_job(name:, schedule:, message:, deliver: false, channel: nil, to: nil, delete_after_run: false)
        load_store
        now = (Time.now.to_f * 1000).to_i

        job = Job.new(
          id: SecureRandom.hex(4),
          name: name,
          enabled: true,
          schedule: schedule,
          payload: Payload.new(
            kind: "agent_turn",
            message: message,
            deliver: deliver,
            channel: channel,
            to: to
          ),
          state: JobState.new(next_run_at_ms: compute_next_run(schedule, now)),
          created_at_ms: now,
          updated_at_ms: now,
          delete_after_run: delete_after_run
        )

        @store[:jobs] ||= []
        @store[:jobs] << job
        save_store
        arm_timer

        SmartBot.logger.info "Cron: added job '#{name}' (#{job.id})"
        job
      end

      def remove_job(job_id)
        load_store
        before = @store[:jobs]&.size || 0
        @store[:jobs]&.reject! { |j| j.id == job_id }
        removed = (@store[:jobs]&.size || 0) < before

        if removed
          save_store
          arm_timer
          SmartBot.logger.info "Cron: removed job #{job_id}"
        end
        removed
      end

      def enable_job(job_id, enabled: true)
        load_store
        job = @store[:jobs]&.find { |j| j.id == job_id }
        return nil unless job

        job.enabled = enabled
        job.updated_at_ms = (Time.now.to_f * 1000).to_i
        
        if enabled
          job.state.next_run_at_ms = compute_next_run(job.schedule, job.updated_at_ms)
        else
          job.state.next_run_at_ms = nil
        end
        
        save_store
        arm_timer
        job
      end

      def run_job(job_id, force: false)
        load_store
        job = @store[:jobs]&.find { |j| j.id == job_id }
        return false unless job
        return false if !job.enabled && !force

        execute_job(job)
        save_store
        arm_timer
        true
      end

      def status
        load_store
        {
          enabled: @running,
          jobs: @store[:jobs]&.size || 0,
          next_wake_at_ms: get_next_wake_ms
        }
      end

      private

      def load_store
        @mutex.synchronize do
          return @store if @store

          if File.exist?(@store_path)
            begin
              data = JSON.parse(File.read(@store_path), symbolize_names: true)
              @store = {
                version: data[:version] || 1,
                jobs: (data[:jobs] || []).map { |j| parse_job(j) }
              }
            rescue JSON::ParserError
              @store = { version: 1, jobs: [] }
            end
          else
            @store = { version: 1, jobs: [] }
          end
        end
        @store
      end

      def parse_job(data)
        # Convert camelCase schedule keys to snake_case
        schedule_data = data[:schedule] || {}
        schedule_kwargs = {
          kind: schedule_data[:kind],
          at_ms: schedule_data[:atMs],
          every_ms: schedule_data[:everyMs],
          expr: schedule_data[:expr],
          tz: schedule_data[:tz]
        }.compact

        # Convert camelCase state keys to snake_case
        state_data = data[:state] || {}
        state_kwargs = {
          next_run_at_ms: state_data[:nextRunAtMs],
          last_run_at_ms: state_data[:lastRunAtMs],
          last_status: state_data[:lastStatus],
          last_error: state_data[:lastError]
        }.compact

        Job.new(
          id: data[:id],
          name: data[:name],
          enabled: data.fetch(:enabled, true),
          schedule: Schedule.new(**schedule_kwargs),
          payload: Payload.new(**data[:payload].transform_keys(&:to_sym)),
          state: JobState.new(**state_kwargs),
          created_at_ms: data[:createdAtMs],
          updated_at_ms: data[:updatedAtMs],
          delete_after_run: data.fetch(:deleteAfterRun, false)
        )
      end

      def save_store
        @mutex.synchronize do
          return unless @store
          FileUtils.mkdir_p(File.dirname(@store_path))
          File.write(@store_path, JSON.pretty_generate(
            version: @store[:version],
            jobs: @store[:jobs].map(&:to_h)
          ))
        end
      end

      def recompute_next_runs
        now = (Time.now.to_f * 1000).to_i
        @store[:jobs]&.each do |job|
          job.state.next_run_at_ms = compute_next_run(job.schedule, now) if job.enabled
        end
      end

      def compute_next_run(schedule, now_ms)
        case schedule.kind
        when "at"
          schedule.at_ms && schedule.at_ms > now_ms ? schedule.at_ms : nil
        when "every"
          return nil unless schedule.every_ms && schedule.every_ms > 0
          now_ms + schedule.every_ms
        when "cron"
          # Simplified - would use a cron parser gem
          nil
        end
      end

      def get_next_wake_ms
        times = @store[:jobs]&.select(&:enabled)&.map { |j| j.state.next_run_at_ms }&.compact
        times&.min
      end

      def arm_timer
        return unless @running

        @timer_thread&.kill

        next_wake = get_next_wake_ms
        return unless next_wake

        delay_ms = [next_wake - (Time.now.to_f * 1000).to_i, 0].max
        delay_s = delay_ms.to_f / 1000

        @timer_thread = Thread.new do
          sleep(delay_s)
          on_timer if @running
        end
      end

      def on_timer
        now = (Time.now.to_f * 1000).to_i
        due_jobs = @store[:jobs]&.select { |j| 
          j.enabled && j.state.next_run_at_ms && now >= j.state.next_run_at_ms 
        }

        due_jobs&.each { |job| execute_job(job) }
        save_store
        arm_timer
      end

      def execute_job(job)
        start_ms = (Time.now.to_f * 1000).to_i
        SmartBot.logger.info "Cron: executing job '#{job.name}' (#{job.id})"

        begin
          if @on_job
            response = @on_job.call(job)
            SmartBot.logger.info "Cron: job '#{job.name}' completed"
          end
          job.state.last_status = "ok"
          job.state.last_error = nil
        rescue => e
          job.state.last_status = "error"
          job.state.last_error = e.message
          SmartBot.logger.error "Cron: job '#{job.name}' failed: #{e.message}"
        end

        job.state.last_run_at_ms = start_ms
        job.updated_at_ms = (Time.now.to_f * 1000).to_i

        # Handle one-shot jobs
        if job.schedule.kind == "at"
          if job.delete_after_run
            @store[:jobs].reject! { |j| j.id == job.id }
          else
            job.enabled = false
            job.state.next_run_at_ms = nil
          end
        else
          job.state.next_run_at_ms = compute_next_run(job.schedule, job.updated_at_ms)
        end
      end
    end
  end
end
