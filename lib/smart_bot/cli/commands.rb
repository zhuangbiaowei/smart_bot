# frozen_string_literal: true

require "thor"
require "yaml"

module SmartBot
  module CLI
    class Commands < Thor
      desc "agent", "Interact with the agent"
      option :message, aliases: "-m", desc: "Message to send"
      option :session, aliases: "-s", default: "cli:default", desc: "Session ID"
      option :llm, aliases: "-l", desc: "LLM to use"
      def agent
        # 初始化 SmartAgent
        require "smart_agent"
        require "smart_prompt"
        
        # 初始化日志
        FileUtils.mkdir_p(File.expand_path("~/.smart_bot/logs"))
        SmartAgent.logger = Logger.new(File.expand_path("~/.smart_bot/logs/smart_agent.log"))
        SmartAgent.logger.level = Logger::INFO
        
        # 加载配置和工具
        agent_config = File.expand_path("~/.smart_bot/agent.yml")
        @agent_engine = SmartAgent::Engine.new(agent_config)
        load_smartbot_tools
        
        # 获取当前配置
        smart_prompt_config = YAML.load_file(File.expand_path("~/.smart_bot/smart_prompt.yml"))
        current_llm = options[:llm] || smart_prompt_config["default_llm"] || "deepseek"
        
        if options[:message]
          # 单次对话模式
          response = chat_with_tools(options[:message], current_llm)
          say "\n🤖 #{response}"
        else
          # 交互模式
          say "🤖 SmartBot (powered by SmartAgent)"
          say "   Commands: /models, /llm <name>, /help\n"

          loop do
            begin
              user_input = ask("You:", :blue, bold: true)
              break if user_input.nil?
              next if user_input.strip.empty?

              # 处理斜杠命令
              if user_input.start_with?("/")
                handle_command(user_input, smart_prompt_config, current_llm)
                next
              end

              response = chat_with_tools(user_input, current_llm)
              say "\n🤖 #{response}\n"
              
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
        config_path = File.expand_path("~/.smart_bot/smart_prompt.yml")
        
        say "🤖 SmartBot Status\n"
        
        if File.exist?(config_path)
          say "Config: #{config_path} " + set_color("✓", :green)
          config = YAML.load_file(config_path)
          say "Default LLM: #{config['default_llm'] || 'Not set'}"
          
          say "\nConfigured Providers:"
          config["llms"]&.each do |name, settings|
            has_key = settings["api_key"].to_s.strip.length > 0
            status = has_key ? set_color("✓", :green) : set_color("not set", :dim)
            say "  #{name}: #{status} (#{settings['model']})"
          end
        else
          say "Config: not found. Run 'smart_bot onboard'", :red
        end
      end

      desc "onboard", "Initialize SmartBot configuration"
      def onboard
        say "🤖 SmartBot Setup\n"
        
        # 创建目录
        FileUtils.mkdir_p(File.expand_path("~/.smart_bot/logs"))
        FileUtils.mkdir_p(File.expand_path("~/.smart_bot/workspace"))
        FileUtils.mkdir_p(File.expand_path("~/.smart_bot/workers"))
        
        say "✓ Created directories", :green
        
        # 复制默认配置
        config_source = File.join(File.dirname(__FILE__), "../../../config/smart_bot.yml")
        if File.exist?(config_source)
          FileUtils.cp(config_source, File.expand_path("~/.smart_bot/smart_bot.yml"))
        end
        
        say "\n请编辑配置文件添加 API Key："
        say "  ~/.smart_bot/smart_prompt.yml"
        say "\n然后运行: smart_bot agent"
      end

      private

      # 主要的对话逻辑 - 手动处理工具调用
      def chat_with_tools(message, llm_name)
        # 检查是否需要调用工具
        url_pattern = %r{https?://[^\s]+}
        urls = message.scan(url_pattern)
        
        # 如果消息包含 URL，直接调用 web_fetch
        if urls.any?
          url = urls.first
          say "🔍 正在抓取网页: #{url}", :cyan
          
          tool_result = call_tool(:web_fetch, {
            "url" => url,
            "extract_mode" => "markdown"
          })
          
          if tool_result[:error]
            return "抓取失败: #{tool_result[:error]}"
          end
          
          # 构建包含抓取结果的提示
          prompt = <<~PROMPT
            用户问题: #{message}
            
            网页标题: #{tool_result[:title]}
            
            网页内容:
            #{tool_result[:content][0..3000]}
            #{tool_result[:truncated] ? "...(内容已截断)" : ""}
            
            请根据以上内容回答用户的问题。
          PROMPT
          
          return call_llm(prompt, llm_name)
        end
        
        # 检查文件操作请求
        if message =~ /读取?文件|read file/i
          # 尝试提取文件路径
          path_match = message.match(/["']?([\w\-\.\/\\]+\.[\w]+)["']?/)
          if path_match
            path = path_match[1]
            say "📖 正在读取文件: #{path}", :cyan
            
            tool_result = call_tool(:read_file, { "path" => path })
            
            if tool_result[:error]
              return "读取失败: #{tool_result[:error]}"
            end
            
            return "文件内容:\n```\n#{tool_result[:content][0..2000]}\n```#{tool_result[:content].length > 2000 ? '\n...(已截断)' : ''}"
          end
        end
        
        # 默认：直接调用 LLM
        call_llm(message, llm_name)
      end

      # 调用 LLM
      def call_llm(prompt, llm_name)
        engine = SmartPrompt::Engine.new(File.expand_path("~/.smart_bot/smart_prompt.yml"))
        
        # 创建临时 worker
        SmartPrompt.define_worker :temp_chat do
          use llm_name
          sys_msg "You are SmartBot, a helpful AI assistant."
          prompt params[:text]
          send_msg
        end
        
        result = engine.call_worker(:temp_chat, { text: prompt })
        result
      end

      # 调用工具
      def call_tool(tool_name, params)
        tool = SmartAgent::Tool.find_tool(tool_name)
        return { error: "Tool not found: #{tool_name}" } unless tool
        
        tool.call(params)
      end

      # 加载 SmartBot 自定义工具
      def load_smartbot_tools
        tools_dir = File.expand_path("~/smart_ai/smart_bot/agents/tools")
        if File.directory?(tools_dir)
          Dir.glob(File.join(tools_dir, "*.rb")).each { |f| require f }
        end
      end

      # 处理斜杠命令
      def handle_command(input, config, current_llm)
        cmd, *args = input.split
        
        case cmd
        when "/help"
          say "\n📖 Commands:"
          say "  /models        - List available LLMs"
          say "  /llm <name>   - Switch LLM provider"
          say "  /help          - Show this help"
          say "  Ctrl+C         - Exit\n"
          
        when "/models"
          say "\n📋 Available LLMs:"
          config["llms"]&.each do |name, settings|
            marker = (name == current_llm) ? set_color("→", :green) : " "
            say "  #{marker} #{name}: #{settings['model']}"
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
            say "✓ Switched to LLM: #{set_color(current_llm, :green)}"
          else
            say "❌ Unknown LLM: #{new_llm}", :red
          end
        else
          say "Unknown command: #{cmd}. Type /help for available commands.", :yellow
        end
      end
    end
  end
end
