# frozen_string_literal: true

require "thor"

module SmartBot
  module CLI
    # CronCommands must be defined before Commands
    class CronCommands < Thor
      desc "list", "List scheduled jobs"
      option :all, aliases: "-a", type: :boolean, default: false, desc: "Include disabled jobs"
      def list
        store_path = File.join(Utils::Helpers.data_path, "cron", "jobs.json")
        service = Cron::Service.new(store_path)
        jobs = service.list_jobs(include_disabled: options[:all])

        if jobs.empty?
          say "No scheduled jobs."
          return
        end

        say "ID\t\tName\t\tSchedule\t\tStatus\t\tNext Run"
        jobs.each do |job|
          sched = case job.schedule.kind
                  when "every" then "every #{(job.schedule.every_ms || 0) / 1000}s"
                  when "cron" then job.schedule.expr || ""
                  else "one-time"
                  end
          
          next_run = job.state.next_run_at_ms ? Time.at(job.state.next_run_at_ms / 1000).strftime("%Y-%m-%d %H:%M") : ""
          status = job.enabled ? set_color("enabled", :green) : set_color("disabled", :dim)
          
          say "#{job.id}\t#{job.name}\t#{sched}\t#{status}\t#{next_run}"
        end
      end

      desc "add", "Add a scheduled job"
      option :name, required: true, desc: "Job name"
      option :message, required: true, desc: "Message for agent"
      option :every, type: :numeric, desc: "Run every N seconds"
      option :cron, desc: "Cron expression (e.g. '0 9 * * *')"
      option :at, desc: "Run once at time (ISO format)"
      option :deliver, type: :boolean, default: false, desc: "Deliver response to channel"
      option :to, desc: "Recipient for delivery"
      option :channel, desc: "Channel for delivery"
      def add
        schedule = if options[:every]
          Cron::Schedule.new(kind: "every", every_ms: options[:every] * 1000)
        elsif options[:cron]
          Cron::Schedule.new(kind: "cron", expr: options[:cron])
        elsif options[:at]
          time = Time.parse(options[:at])
          Cron::Schedule.new(kind: "at", at_ms: (time.to_f * 1000).to_i)
        else
          say "Error: Must specify --every, --cron, or --at", :red
          exit 1
        end

        store_path = File.join(Utils::Helpers.data_path, "cron", "jobs.json")
        service = Cron::Service.new(store_path)
        
        job = service.add_job(
          name: options[:name],
          schedule: schedule,
          message: options[:message],
          deliver: options[:deliver],
          to: options[:to],
          channel: options[:channel]
        )

        say "✓ Added job '#{job.name}' (#{job.id})", :green
      end

      desc "remove JOB_ID", "Remove a scheduled job"
      def remove(job_id)
        store_path = File.join(Utils::Helpers.data_path, "cron", "jobs.json")
        service = Cron::Service.new(store_path)

        if service.remove_job(job_id)
          say "✓ Removed job #{job_id}", :green
        else
          say "Job #{job_id} not found", :red
        end
      end

      desc "enable JOB_ID", "Enable a job"
      option :disable, type: :boolean, default: false, desc: "Disable instead of enable"
      def enable(job_id)
        store_path = File.join(Utils::Helpers.data_path, "cron", "jobs.json")
        service = Cron::Service.new(store_path)

        job = service.enable_job(job_id, enabled: !options[:disable])
        if job
          status = options[:disable] ? "disabled" : "enabled"
          say "✓ Job '#{job.name}' #{status}", :green
        else
          say "Job #{job_id} not found", :red
        end
      end

      desc "execute JOB_ID", "Manually execute a job"
      option :force, aliases: "-f", type: :boolean, default: false, desc: "Execute even if disabled"
      def execute(job_id)
        store_path = File.join(Utils::Helpers.data_path, "cron", "jobs.json")
        service = Cron::Service.new(store_path)

        if service.run_job(job_id, force: options[:force])
          say "✓ Job executed", :green
        else
          say "Failed to run job #{job_id}", :red
        end
      end
    end

    class Commands < Thor
      desc "onboard", "Initialize SmartBot configuration and workspace"
      def onboard
        config_path = Config.default_config_path
        
        if File.exist?(config_path)
          say "Config already exists at #{config_path}", :yellow
          return unless yes?("Overwrite?")
        end

        # Create default config
        config = Config.new
        config.save
        say "✓ Created config at #{config_path}", :green

        # Create workspace
        workspace = Utils::Helpers.workspace_path
        say "✓ Created workspace at #{workspace}", :green

        # Create bootstrap files
        create_workspace_templates(workspace)

        say "\n🤖 SmartBot is ready!", :cyan
        say "\nNext steps:"
        say "  1. Add your API key to ~/.smart_bot/config.json"
        say "     Get one at: https://openrouter.ai/keys"
        say "  2. Chat: smart_bot agent -m \"Hello!\""
      end

      desc "gateway", "Start the SmartBot gateway"
      option :port, type: :numeric, default: 18790, desc: "Gateway port"
      option :verbose, type: :boolean, default: false, desc: "Verbose output"
      def gateway
        SmartBot.logger.level = Logger::DEBUG if options[:verbose]
        
        say "🤖 Starting SmartBot gateway on port #{options[:port]}..."

        config = Config.load
        
        unless config.api_key
          say "Error: No API key configured.", :red
          say "Set one in ~/.smart_bot/config.json under providers.openrouter.apiKey"
          exit 1
        end

        # Create components
        bus = Bus::Queue.new

        provider = Providers::OpenRouterProvider.new(
          api_key: config.api_key,
          api_base: config.api_base,
          default_model: config.model
        )

        agent = Agent::Loop.new(
          bus: bus,
          provider: provider,
          workspace: config.workspace_path,
          model: config.model,
          max_iterations: config.max_tool_iterations,
          brave_api_key: config.tools[:web_search]&.dig(:api_key)
        )

        # Cron service
        cron_store = File.join(Utils::Helpers.data_path, "cron", "jobs.json")
        cron = Cron::Service.new(cron_store, on_job: lambda { |job|
          response = agent.process_direct(job.payload.message, session_key: "cron:#{job.id}")
          if job.payload.deliver && job.payload.to
            bus.publish_outbound(Bus::OutboundMessage.new(
              channel: job.payload.channel || "telegram",
              chat_id: job.payload.to,
              content: response
            ))
          end
          response
        })

        # Heartbeat service
        heartbeat = Heartbeat::Service.new(
          workspace: config.workspace_path,
          on_heartbeat: lambda { |prompt| agent.process_direct(prompt, session_key: "heartbeat") },
          interval_s: 30 * 60,
          enabled: true
        )

        # Channel manager
        channels = Channels::Manager.new(config, bus)

        say "✓ Channels enabled: #{channels.enabled_channels.join(', ')}" if channels.enabled_channels.any?
        cron_status = cron.status
        say "✓ Cron: #{cron_status[:jobs]} scheduled jobs" if cron_status[:jobs] > 0
        say "✓ Heartbeat: every 30m"

        # Start services
        cron.start
        heartbeat.start

        # Trap signals
        trap("INT") do
          say "\nShutting down..."
          heartbeat.stop
          cron.stop
          agent.stop
          channels.stop_all
          exit 0
        end

        # Start agent and channels
        agent_thread = Thread.new { agent.run }
        channels_thread = Thread.new { channels.start_all }

        [agent_thread, channels_thread].each(&:join)
      end

      desc "agent", "Interact with the agent directly"
      option :message, aliases: "-m", desc: "Message to send to the agent"
      option :session, aliases: "-s", default: "cli:default", desc: "Session ID"
      def agent
        config = Config.load

        unless config.api_key
          say "Error: No API key configured.", :red
          exit 1
        end

        bus = Bus::Queue.new
        provider = Providers::OpenRouterProvider.new(
          api_key: config.api_key,
          api_base: config.api_base,
          default_model: config.model
        )

        agent_loop = Agent::Loop.new(
          bus: bus,
          provider: provider,
          workspace: config.workspace_path,
          brave_api_key: config.tools[:web_search]&.dig(:api_key)
        )

        if options[:message]
          # Single message mode
          response = agent_loop.process_direct(options[:message], session_key: options[:session])
          say "\n🤖 #{response}"
        else
          # Interactive mode
          say "🤖 Interactive mode (Ctrl+C to exit)\n"

          loop do
            begin
              user_input = ask("You:", :blue, bold: true)
              next if user_input.strip.empty?

              response = agent_loop.process_direct(user_input, session_key: options[:session])
              say "\n🤖 #{response}\n"
            rescue Interrupt
              say "\nGoodbye!"
              break
            end
          end
        end
      end

      desc "cron", "Manage scheduled tasks"
      subcommand "cron", CronCommands

      desc "status", "Show SmartBot status"
      def status
        config_path = Config.default_config_path
        workspace = Utils::Helpers.workspace_path

        say "🤖 SmartBot Status\n"

        config_ok = File.exist?(config_path)
        workspace_ok = File.exist?(workspace)
        
        say "Config: #{config_path} " + (config_ok ? set_color("✓", :green) : set_color("✗", :red))
        say "Workspace: #{workspace} " + (workspace_ok ? set_color("✓", :green) : set_color("✗", :red))

        if config_ok
          config = Config.load
          say "Model: #{config.model}"
          say "\nAPI Providers:"
          
          providers = {
            "OpenRouter" => :openrouter,
            "Anthropic" => :anthropic,
            "OpenAI" => :openai,
            "Gemini" => :gemini,
            "SiliconFlow" => :siliconflow,
            "DeepSeek" => :deepseek,
            "Aliyun" => :aliyun,
            "Kimi Coding" => :kimi_coding
          }
          
          providers.each do |name, key|
            value = config.providers[key]
            status = value && value[:api_key].to_s.strip.empty? ? set_color("not set", :dim) : set_color("✓", :green)
            say "  #{name}: #{status}"
          end
        end
      end

      private

      def create_workspace_templates(workspace)
        templates = {
          "AGENTS.md" => "# Agent Instructions\n\n" \
                        "You are a helpful AI assistant. Be concise, accurate, and friendly.\n\n" \
                        "## Guidelines\n\n" \
                        "- Always explain what you're doing before taking actions\n" \
                        "- Ask for clarification when the request is ambiguous\n" \
                        "- Use tools to help accomplish tasks\n" \
                        "- Remember important information in your memory files\n",
          
          "SOUL.md" => "# Soul\n\n" \
                      "I am SmartBot, a Ruby-based AI assistant.\n\n" \
                      "## Personality\n" \
                      "- Helpful and friendly\n" \
                      "- Concise and to the point\n" \
                      "- Curious and eager to learn\n\n" \
                      "## Values\n" \
                      "- Accuracy over speed\n" \
                      "- User privacy and safety\n" \
                      "- Transparency in actions\n",
          
          "USER.md" => "# User\n\n" \
                      "Information about the user goes here.\n\n" \
                      "## Preferences\n" \
                      "- Communication style: (casual/formal)\n" \
                      "- Timezone: (your timezone)\n" \
                      "- Language: (your preferred language)\n"
        }

        templates.each do |filename, content|
          file_path = File.join(workspace, filename)
          unless File.exist?(file_path)
            File.write(file_path, content)
            say "  Created #{filename}", :dim
          end
        end

        # Create memory directory and MEMORY.md
        memory_dir = File.join(workspace, "memory")
        FileUtils.mkdir_p(memory_dir)
        memory_file = File.join(memory_dir, "MEMORY.md")
        unless File.exist?(memory_file)
          File.write(memory_file, "# Long-term Memory\n\n" \
            "This file stores important information that should persist across sessions.\n\n" \
            "## User Information\n\n" \
            "(Important facts about the user)\n\n" \
            "## Preferences\n\n" \
            "(User preferences learned over time)\n\n" \
            "## Important Notes\n\n" \
            "(Things to remember)\n")
          say "  Created memory/MEMORY.md", :dim
        end
      end
    end
  end
end
