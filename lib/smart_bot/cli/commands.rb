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
        load_mcp_clients
        
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
        
        # 检查是否是搜索请求
        search_patterns = [
          /^搜索[：:]?\s*(.+)/i,
          /搜索\s+(.+)/i,
          /search\s+for\s+(.+)/i,
          /google\s+(.+)/i,
          /bing\s+(.+)/i,
          /baidu\s+(.+)/i,
          /查找\s+(.+)/i
        ]
        
        search_query = nil
        search_patterns.each do |pattern|
          if match = message.match(pattern)
            search_query = match[1].strip
            break
          end
        end
        
        # 检测特定搜索引擎
        serp_engine = "google"
        if message =~ /^baidu\s+/i
          serp_engine = "baidu"
        elsif message =~ /^bing\s+/i
          serp_engine = "bing"
        end
        
        # 优先使用 MCP 搜索（如果可用）
        if search_query
          mcp_result = try_mcp_search(search_query, llm_name)
          return mcp_result if mcp_result
        end
        
        # 回退到 SerpAPI（如果配置了）
        if search_query && ENV["SERP_API_KEY"]
          say "🔍 正在使用 SerpAPI(#{serp_engine}) 搜索: #{search_query}", :cyan
          
          tool_result = call_tool(:serp_search, {
            "query" => search_query,
            "engine" => serp_engine,
            "count" => 5
          })
          
          if tool_result[:error]
            return "搜索失败: #{tool_result[:error]}"
          end
          
          # 构建搜索结果摘要
          results_text = tool_result[:results].map.with_index(1) do |r, i|
            "#{i}. #{r[:title]}\n   #{r[:url]}\n   #{r[:description]}"
          end.join("\n\n")
          
          kg_text = ""
          if tool_result[:knowledge_graph]
            kg = tool_result[:knowledge_graph]
            kg_text = "\n\n📚 知识图谱: #{kg[:title]}\n#{kg[:description]}"
          end
          
          prompt = <<~PROMPT
            用户搜索: #{search_query}
            
            搜索结果:
            #{results_text}#{kg_text}
            
            请根据以上搜索结果，为用户提供简洁有用的回答。
          PROMPT
          
          return call_llm(prompt, llm_name)
        end
        
        # 回退到 Brave Search
        if search_query
          say "🔍 正在使用 Brave Search 搜索: #{search_query}", :cyan
          
          tool_result = call_tool(:web_search, {
            "query" => search_query,
            "count" => 5
          })
          
          if tool_result[:error]
            if tool_result[:error].include?("Brave API key not configured")
              return "搜索功能需要 Brave API Key。请设置环境变量:\n  export BRAVE_API_KEY=\"BSA-your-key\"\n\n获取方式: https://brave.com/search/api/"
            end
            return "搜索失败: #{tool_result[:error]}"
          end
          
          # 构建搜索结果摘要
          results_text = tool_result[:results].map.with_index(1) do |r, i|
            "#{i}. #{r[:title]}\n   #{r[:url]}\n   #{r[:description]}"
          end.join("\n\n")
          
          prompt = <<~PROMPT
            用户搜索: #{search_query}
            
            搜索结果:
            #{results_text}
            
            请根据以上搜索结果，为用户提供简洁有用的回答。
          PROMPT
          
          return call_llm(prompt, llm_name)
        end
        
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
        
        # 使用唯一的 worker 名称避免冲突
        worker_name = :"temp_chat_#{llm_name}"
        
        # 只在未定义时创建 worker
        unless SmartPrompt::Worker.workers.key?(worker_name)
          SmartPrompt.define_worker worker_name do
            use llm_name
            sys_msg "You are SmartBot, a helpful AI assistant."
            prompt params[:text]
            send_msg
          end
        end
        
        result = engine.call_worker(worker_name, { text: prompt })
        result
      end

      # 调用工具
      def call_tool(tool_name, params)
        tool = SmartAgent::Tool.find_tool(tool_name)
        return { error: "Tool not found: #{tool_name}" } unless tool
        
        tool.call(params)
      end

      # 尝试使用 MCP 搜索
      def try_mcp_search(query, llm_name)
        # 先检查 MCP 是否可用（通过查找 search 工具）
        server_name = find_mcp_server_for_tool(:search)
        return nil unless server_name
        
        say "🔍 正在使用 MCP(#{server_name}) 搜索: #{query}", :cyan
        
        # 调用 MCP 搜索工具
        result = call_mcp_tool(server_name, "search", { "query" => query })
        
        # 如果 MCP 不可用或失败，返回 nil 让上层回退
        return nil if result.nil?
        
        # MCP 返回的是 hash，处理不同格式
        result_hash = result.is_a?(Hash) ? result : { "content" => result.to_s }
        
        # 检查是否有错误
        if result_hash["error"] || result_hash["isError"]
          say "MCP 搜索返回错误: #{result_hash["error"]}"
          return nil
        end
        
        # 提取内容
        content = result_hash["content"] || result_hash["text"] || result_hash.to_s
        
        # 尝试解析 JSON 格式的搜索结果
        begin
          # 如果 content 是 Array，直接处理
          results = content.is_a?(Array) ? content : nil
          
          # 如果是字符串，尝试解析
          if content.is_a?(String) && (content.start_with?("[") || content.include?("formattedUrl"))
            results = JSON.parse(content)
          end
          
          if results
            # 解析 MCP 返回的嵌套 JSON 格式
            if results.is_a?(Array) && results.first.is_a?(Hash)
              if results.first["type"] == "text"
                # 嵌套的 JSON 字符串
                inner_text = results.first["text"]
                results = JSON.parse(inner_text) if inner_text
              end
            end
            
            # 格式化为易读的文本
            if results.is_a?(Array) && results.first.is_a?(Hash)
              formatted = results.first(5).map.with_index(1) do |r, i|
                title = r["title"] || "结果 #{i}"
                url = r["link"] || r["url"] || r["formattedUrl"] || ""
                snippet = r["snippet"] || r["description"] || ""
                "#{i}. **#{title}**\n   #{url}\n   #{snippet[0..150]}"
              end.join("\n\n")
              
              prompt = <<~PROMPT
                用户搜索: #{query}
                
                搜索结果:
                #{formatted}
                
                请根据以上搜索结果，为用户提供简洁有用的中文回答。
              PROMPT
              
              return call_llm(prompt, llm_name)
            end
          end
        rescue JSON::ParserError
          # 不是 JSON，继续处理
        end
        
        # 如果内容看起来像搜索结果列表
        if content.include?("http") || content.include?("标题") || content.include?("链接")
          prompt = <<~PROMPT
            用户搜索: #{query}
            
            MCP 搜索结果:
            #{content[0..4000]}
            
            请根据以上搜索结果，为用户提供简洁有用的中文回答。
          PROMPT
          
          return call_llm(prompt, llm_name)
        else
          # 直接返回 MCP 的结果
          return content
        end
        
      rescue => e
        say "MCP 搜索失败: #{e.message}", :yellow
        nil
      end

      # 加载 SmartBot 自定义工具
      def load_smartbot_tools
        tools_dir = File.expand_path("~/smart_ai/smart_bot/agents/tools")
        if File.directory?(tools_dir)
          Dir.glob(File.join(tools_dir, "*.rb")).each { |f| require f }
        end
      end

      # 加载 MCP 客户端
      def load_mcp_clients
        mcp_dir = File.expand_path("~/smart_ai/smart_bot/agents/mcp_clients")
        if File.directory?(mcp_dir)
          Dir.glob(File.join(mcp_dir, "*.rb")).each { |f| require f }
        end
      rescue => e
        say "⚠️  MCP 客户端加载失败: #{e.message}", :yellow if @agent_engine
      end

      # 调用 MCP 工具
      def call_mcp_tool(server_name, tool_name, params)
        # 获取已定义的服务器
        servers = SmartAgent::MCPClient.servers
        return nil unless servers.key?(server_name)
        
        # 创建客户端并调用工具
        client = SmartAgent::MCPClient.new(server_name)
        client.call(tool_name, params)
      rescue => e
        say "MCP 调用失败: #{e.message}", :yellow if @agent_engine
        nil
      end

      # 查找 MCP 服务器（通过工具名）
      def find_mcp_server_for_tool(tool_name)
        server_name = SmartAgent::MCPClient.find_server_by_tool_name(tool_name.to_sym)
        return nil unless server_name
        
        # 返回可用的服务器名称
        servers = SmartAgent::MCPClient.servers
        servers.key?(server_name) ? server_name : nil
      rescue
        nil
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
