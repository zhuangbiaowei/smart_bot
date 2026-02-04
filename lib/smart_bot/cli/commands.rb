# frozen_string_literal: true

require "thor"
require "yaml"

module SmartBot
  module CLI
    class CronCommands < Thor
      desc "list", "List scheduled jobs"
      def list
        store_path = File.join(Utils::Helpers.data_path, "cron", "jobs.json")
        service = Cron::Service.new(store_path)
        jobs = service.list_jobs

        if jobs.empty?
          say "No scheduled jobs."
          return
        end

        say "ID\t\tName\t\tSchedule\t\tStatus"
        jobs.each do |job|
          sched = case job.schedule.kind
                  when "every" then "every #{(job.schedule.every_ms || 0) / 1000}s"
                  when "cron" then job.schedule.expr || ""
                  else "one-time"
                  end
          status = job.enabled ? set_color("enabled", :green) : set_color("disabled", :dim)
          say "#{job.id}\t#{job.name}\t#{sched}\t#{status}"
        end
      end

      desc "add", "Add a scheduled job"
      option :name, required: true, desc: "Job name"
      option :message, required: true, desc: "Message for agent"
      option :every, type: :numeric, desc: "Run every N seconds"
      option :cron, desc: "Cron expression"
      def add
        # ... keep existing cron implementation
      end
    end

    class Commands < Thor
      desc "onboard", "Initialize SmartBot configuration"
      def onboard
        config_path = File.expand_path("~/.smart_bot/config.json")
        
        if File.exist?(config_path)
          say "Config already exists at #{config_path}", :yellow
          return unless yes?("Overwrite?")
        end

        # Create config directory
        FileUtils.mkdir_p(File.dirname(config_path))
        
        # Copy default config
        default_config = File.join(File.dirname(__FILE__), "../../../config/smart_bot.yml")
        if File.exist?(default_config)
          FileUtils.cp(default_config, File.expand_path("~/.smart_bot/smart_bot.yml"))
        end
        
        # Create workspace
        workspace = Utils::Helpers.workspace_path
        say "✓ Created workspace at #{workspace}", :green

        # Create bootstrap files
        create_workspace_templates(workspace)

        say "\n🤖 SmartBot is ready!", :cyan
        say "\nNext steps:"
        say "  1. Add your API keys to ~/.smart_bot/smart_bot.yml"
        say "  2. Chat: smart_bot agent -m \"Hello!\""
      end

      desc "agent", "Interact with the agent"
      option :message, aliases: "-m", desc: "Message to send"
      option :session, aliases: "-s", default: "cli:default", desc: "Session ID"
      option :llm, aliases: "-l", desc: "LLM to use (deepseek, siliconflow, aliyun, kimi)"
      def agent
        # Load config and init SmartAgent engine
        config_path = File.expand_path("~/.smart_bot/smart_bot.yml")
        
        unless File.exist?(config_path)
          say "Config not found. Run 'smart_bot onboard' first.", :red
          exit 1
        end

        # Copy config to working directory if needed
        config = load_config_with_env(config_path)
        
        # Check if we have any API keys
        has_key = config["llms"].any? { |_, v| v["api_key"]&.to_s.strip.length > 0 }
        unless has_key
          say "Error: No API keys configured.", :red
          say "Add your API keys to ~/.smart_bot/smart_bot.yml"
          exit 1
        end

        # Initialize SmartAgent
        require "smart_agent"
        require "smart_prompt"
        
        # Initialize loggers first
        FileUtils.mkdir_p(File.expand_path("~/.smart_bot/logs"))
        
        # Initialize SmartAgent Engine (this also sets up SmartPrompt::Engine)
        agent_config = File.expand_path("~/.smart_bot/agent.yml")
        @agent_engine = SmartAgent::Engine.new(agent_config)
        
        # Load additional SmartBot workers and tools
        load_agents_and_tools
        
        # Get the agent
        agent = @agent_engine.agents[:smart_bot]
        
        current_llm = options[:llm] || config["default_llm"] || "deepseek"
        current_model = config["llms"][current_llm]&.[]("default_model")

        if options[:message]
          # Single message mode
          result = agent.please(options[:message])
          say "\n🤖 #{result}"
        else
          # Interactive mode
          say "🤖 SmartBot (powered by SmartAgent)"
          say "   Commands: /models, /model <name>, /llm <name>, /help\n"

          loop do
            begin
              user_input = ask("You:", :blue, bold: true)
              break if user_input.nil?
              next if user_input.strip.empty?

              # Handle slash commands
              if user_input.start_with?("/")
                handle_command(user_input, config, agent, current_llm, current_model)
                next
              end

              result = agent.please(user_input)
              say "\n🤖 #{result}\n"
              
            rescue Interrupt
              say "\nGoodbye!"
              break
            rescue => e
              say "\n❌ Error: #{e.message}\n", :red
            end
          end
        end
      end

      desc "status", "Show SmartBot status"
      def status
        config_path = File.expand_path("~/.smart_bot/smart_bot.yml")
        
        say "🤖 SmartBot Status\n"
        
        if File.exist?(config_path)
          say "Config: #{config_path} " + set_color("✓", :green)
          config = YAML.load_file(config_path)
          say "Default LLM: #{config['default_llm'] || 'Not set'}"
          
          say "\nConfigured Providers:"
          config["llms"]&.each do |name, settings|
            has_key = settings["api_key"].to_s.strip.length > 0
            status = has_key ? set_color("✓", :green) : set_color("not set", :dim)
            say "  #{name}: #{status} (#{settings['default_model']})"
          end
        else
          say "Config: not found. Run 'smart_bot onboard'", :red
        end
        
        workspace = Utils::Helpers.workspace_path
        say "\nWorkspace: #{workspace} " + (File.exist?(workspace) ? set_color("✓", :green) : set_color("✗", :red))
      end

      desc "cron", "Manage scheduled tasks"
      subcommand "cron", CronCommands

      private

      def load_config_with_env(config_path)
        content = File.read(config_path)
        # Replace environment variable references
        content = content.gsub(/\$\{(\w+)\}/) { ENV[$1] || "" }
        YAML.load(content)
      end

      def load_agents_and_tools
        # Load built-in workers from gem
        workers_dir = File.expand_path("../../../agents/workers", __dir__)
        if File.directory?(workers_dir)
          Dir.glob(File.join(workers_dir, "*.rb")).sort.each { |f| require f }
        end
        
        # Load built-in tools from gem
        tools_dir = File.expand_path("../../../agents/tools", __dir__)
        if File.directory?(tools_dir)
          Dir.glob(File.join(tools_dir, "*.rb")).sort.each { |f| require f }
        end
        
        # Load user custom workers
        user_workers = File.expand_path("~/.smart_bot/workers")
        if File.directory?(user_workers)
          Dir.glob(File.join(user_workers, "*.rb")).sort.each { |f| require f }
        end
        
        # Load main agent definition
        agent_file = File.expand_path("../../../agents/smart_bot.rb", __dir__)
        require agent_file if File.exist?(agent_file)
      end

      def handle_command(input, config, agent, current_llm, current_model)
        cmd, *args = input.split
        
        case cmd
        when "/help"
          say "\n📖 Commands:"
          say "  /models        - List available models"
          say "  /llm <name>   - Switch LLM provider"
          say "  /help          - Show this help"
          say "  Ctrl+C         - Exit\n"
          
        when "/models"
          say "\n📋 Available LLMs:"
          config["llms"]&.each do |name, settings|
            marker = (name == current_llm) ? set_color("→", :green) : " "
            say "  #{marker} #{name}: #{settings['default_model']}"
          end
          say ""
          
        when "/llm"
          if args.empty?
            say "Usage: /llm <name>", :yellow
            say "Current: #{current_llm}"
            return
          end
          
          new_llm = args.first
          if config["llms"]&.key?(new_llm)
            current_llm = new_llm
            current_model = config["llms"][new_llm]["default_model"]
            say "✓ Switched to LLM: #{set_color(current_llm, :green)} (#{current_model})"
          else
            say "❌ Unknown LLM: #{new_llm}", :red
          end
          
        else
          say "Unknown command: #{cmd}. Type /help for available commands.", :yellow
        end
      end

      def create_workspace_templates(workspace)
        # ... keep existing template creation
        templates = {
          "AGENTS.md" => "# Agent Instructions\n\nYou are a helpful AI assistant...",
          "SOUL.md" => "# Soul\n\nI am SmartBot, a helpful AI assistant...",
          "USER.md" => "# User\n\nInformation about the user..."
        }

        templates.each do |filename, content|
          file_path = File.join(workspace, filename)
          unless File.exist?(file_path)
            File.write(file_path, content)
            say "  Created #{filename}", :dim
          end
        end

        memory_dir = File.join(workspace, "memory")
        FileUtils.mkdir_p(memory_dir)
      end
    end
  end
end
