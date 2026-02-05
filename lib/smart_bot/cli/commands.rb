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
        
        # 加载 Skill 系统
        load_skill_system
        load_smartbot_tools
        load_mcp_clients
        
        # 加载并激活所有 skills
        load_and_activate_skills
        
        # 获取当前配置
        smart_prompt_config = YAML.load_file(File.expand_path("~/.smart_bot/smart_prompt.yml"))
        current_llm = options[:llm] || smart_prompt_config["default_llm"] || "deepseek"
        
        if options[:message]
          # 单次对话模式
          message = options[:message]
          
          # 处理斜杠命令
          if message.start_with?("/")
            handle_command(message, smart_prompt_config, current_llm)
          else
            response = chat_with_tools(message, current_llm)
            say "\n🤖 #{response}"
          end
        else
          # 交互模式 - 使用 Conversation 维护对话历史
          say "🤖 SmartBot (powered by SmartAgent)"
          say "   Commands: /models, /llm <name>, /skills, /help"
          say "   Use '/new' to start a new conversation\n"

          # 创建 SmartPrompt Engine
          sp_engine = SmartPrompt::Engine.new(File.expand_path("~/.smart_bot/smart_prompt.yml"))
          
          # 创建 Conversation 实例来维护对话历史
          conversation = SmartPrompt::Conversation.new(sp_engine)
          conversation.use(current_llm)
          # 使用 with_history: true 确保系统消息也进入历史记录
          conversation.sys_msg("You are SmartBot, a helpful AI assistant. Remember information the user shares with you during this conversation.", { with_history: true })

          loop do
            begin
              user_input = ask("You:", :blue, bold: true)
              break if user_input.nil?
              next if user_input.strip.empty?

              # 处理斜杠命令
              if user_input.start_with?("/")
                if user_input.strip == "/new"
                  # 新建对话
                  conversation = SmartPrompt::Conversation.new(sp_engine)
                  conversation.use(current_llm)
                  conversation.sys_msg("You are SmartBot, a helpful AI assistant. Remember information the user shares with you during this conversation.", { with_history: true })
                  say "\n🆕 New conversation started!\n", :green
                else
                  handle_command(user_input, smart_prompt_config, current_llm)
                end
                next
              end

              # 使用对话历史进行多轮对话
              response = chat_with_conversation(user_input, conversation, current_llm)
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
        FileUtils.mkdir_p(File.expand_path("~/smart_ai/smart_bot/skills"))
        
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

      desc "skill NAME", "Create a new skill"
      option :description, aliases: "-d", default: "A new skill", desc: "Skill description"
      option :author, aliases: "-a", default: "SmartBot User", desc: "Author name"
      def skill(name)
        # 验证名称
        unless name =~ /^[a-z][a-z0-9_]*$/
          say "❌ Invalid skill name. Use lowercase letters, numbers, and underscores only.", :red
          return
        end
        
        # 创建目录
        skills_dir = File.expand_path("~/smart_ai/smart_bot/skills")
        skill_dir = File.join(skills_dir, name)
        
        if File.exist?(skill_dir)
          say "❌ Skill '#{name}' already exists!", :red
          return
        end
        
        FileUtils.mkdir_p(skill_dir)
        
        # 创建 skill.rb
        skill_rb = File.join(skill_dir, "skill.rb")
        File.write(skill_rb, skill_template(name, options))
        
        # 创建 SKILL.md
        skill_md = File.join(skill_dir, "SKILL.md")
        File.write(skill_md, skill_md_template(name, options))
        
        say "✅ Created skill '#{name}'", :green
        say "   Location: #{skill_dir}"
        say "   Files:"
        say "     - skill.rb"
        say "     - SKILL.md"
        say "\n📝 To activate your skill:"
        say "   The skill will be automatically loaded when you run smart_bot agent"
      end

      private

      # 主要的对话逻辑 - 手动处理工具调用
      def chat_with_tools(message, llm_name)
        # 检查是否需要调用工具
        url_pattern = %r{https?://[^\s]+}
        urls = message.scan(url_pattern)
        
        # ========== 1. 优先检查是否明确指定了 Skill ==========
        # 检查用户是否明确提到了某个 skill 名称
        explicit_skill = detect_explicit_skill(message)
        if explicit_skill
          skill_result = call_skill_by_name(explicit_skill, message, urls, llm_name)
          return skill_result if skill_result
        end
        
        # ========== 2. 智能 Skill 推荐 ==========
        # 使用模糊匹配 + LLM 选择来找到最佳技能
        suggestions = smart_skill_suggest(message, llm_name, 3)
        
        if suggestions.any?
          best = suggestions.first
          
          case best[:confidence]
          when :explicit
            # 已经在上面的显式检测中处理了
            nil
          when :high
            # 高置信度模糊匹配，直接执行
            say "🎯 找到匹配技能: #{best[:name]}", :green
            skill_result = call_skill_by_name(best[:name], message, urls, llm_name)
            return skill_result if skill_result
          when :llm_selected
            # LLM 选择的技能
            say "🤖 推荐使用技能: #{best[:name]}", :cyan
            skill_result = call_skill_by_name(best[:name], message, urls, llm_name)
            return skill_result if skill_result
          when :fuzzy
            # 多个模糊匹配，询问用户或选择最佳
            if suggestions.length == 1
              say "🔍 找到可能匹配的技能: #{best[:name]} (置信度: #{best[:score]})", :yellow
              skill_result = call_skill_by_name(best[:name], message, urls, llm_name)
              return skill_result if skill_result
            else
              # 多个候选，列出供参考
              list = suggestions.map { |s| "#{s[:name]}(#{s[:score]})" }.join(", ")
              say "🔍 找到多个可能匹配的技能: #{list}", :yellow
              say "   尝试使用第一个: #{best[:name]}..."
              skill_result = call_skill_by_name(best[:name], message, urls, llm_name)
              return skill_result if skill_result
            end
          end
        end
        
        # ========== 3. 天气查询 ==========
        weather_match = message.match(/(.+?)(?:的)?天气/i) || message.match(/weather\s+(?:in|for)?\s+(.+)/i)
        if weather_match
          location = weather_match[1].strip
          # 移除常见后缀
          location = location.gsub(/今天|明天|后天|现在|怎么样|如何/, '').strip
          
          say "🌤️  正在查询 #{location} 的天气...", :cyan
          
          tool = SmartAgent::Tool.find_tool(:get_weather)
          if tool
            result = tool.call({ "location" => location, "unit" => "c" })
            
            if result[:error]
              return "查询天气失败: #{result[:error]}"
            end
            
            return <<~WEATHER
              #{result[:location]}, #{result[:country]} 当前天气:
              
              🌡️  温度: #{result[:temperature]}
              📝  状况: #{result[:condition]}
              💧  湿度: #{result[:humidity]}
              💨  风速: #{result[:wind]}
              🤔  体感: #{result[:feels_like]}
            WEATHER
          end
        end
        
        # ========== 4. 搜索请求 ==========
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
        
        # ========== 5. URL 抓取 ==========
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

        # 尝试使用已加载的 Markdown Skills
        skill_result = try_markdown_skills(message, llm_name)
        return skill_result if skill_result
        
        # 默认：直接调用 LLM
        call_llm(message, llm_name)
      end

      # 调用 LLM (单次对话模式)
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
      
      # 使用 Conversation 进行多轮对话
      def chat_with_conversation(message, conversation, llm_name)
        # 先检查是否需要调用工具
        url_pattern = %r{https?://[^\s]+}
        urls = message.scan(url_pattern)
        
        # 检查是否是特殊命令（搜索、天气等）
        # 这些仍然使用即时工具调用，不进入对话历史
        
        # ========== 1. 显式 Skill 调用 ==========
        explicit_skill = detect_explicit_skill(message)
        if explicit_skill
          skill_result = call_skill_by_name(explicit_skill, message, urls, llm_name)
          return skill_result if skill_result
        end
        
        # ========== 2. 天气查询 ==========
        weather_match = message.match(/(.+?)(?:的)?天气/i) || message.match(/weather\s+(?:in|for)?\s+(.+)/i)
        if weather_match
          location = weather_match[1].strip
          location = location.gsub(/今天|明天|后天|现在|怎么样|如何/, '').strip
          
          tool = SmartAgent::Tool.find_tool(:get_weather)
          if tool
            result = tool.call({ "location" => location, "unit" => "c" })
            if result[:error]
              return "查询天气失败: #{result[:error]}"
            end
            weather_info = <<~WEATHER
              #{result[:location]}, #{result[:country]} 当前天气:
              🌡️ 温度: #{result[:temperature]}
              📝 状况: #{result[:condition]}
              💧 湿度: #{result[:humidity]}
              💨 风速: #{result[:wind]}
            WEATHER
            # 将天气信息加入对话历史（使用 with_history: true）
            conversation.add_message({ role: "assistant", content: weather_info }, true)
            return weather_info
          end
        end
        
        # ========== 3. 搜索请求 ==========
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
        
        if search_query
          # 尝试 MCP 搜索
          server_name = find_mcp_server_for_tool(:search)
          if server_name
            result = call_mcp_tool(server_name, "search", { "query" => search_query })
            if result
              search_result = format_search_result(result, search_query, llm_name)
              # 将搜索结果加入对话历史
              conversation.add_message({ role: "assistant", content: "搜索结果:\n#{search_result}" }, true)
              return search_result
            end
          end
        end
        
        # ========== 4. URL 抓取 ==========
        if urls.any?
          url = urls.first
          tool_result = call_tool(:web_fetch, { "url" => url, "extract_mode" => "markdown" })
          if tool_result && !tool_result[:error]
            # 将网页内容作为上下文发送给 LLM
            context = "网页标题: #{tool_result[:title]}\n\n网页内容:\n#{tool_result[:content][0..2000]}"
            conversation.add_message({ role: "user", content: "#{message}\n\n[网页内容]\n#{context}" }, true)
            response = conversation.send_msg(with_history: true)
            # 将助手回复也加入历史
            conversation.add_message({ role: "assistant", content: response }, true)
            return response
          end
        end
        
        # ========== 5. 普通对话（使用 Conversation 维护历史）==========
        # 添加用户消息到历史（使用 with_history: true）
        conversation.add_message({ role: "user", content: message }, true)
        # 发送消息时使用 with_history: true 保留历史
        response = conversation.send_msg(with_history: true)
        # 将助手回复也加入历史
        conversation.add_message({ role: "assistant", content: response }, true)
        response
      rescue => e
        "Error: #{e.message}"
      end
      
      # 格式化搜索结果
      def format_search_result(result, query, llm_name)
        result_hash = result.is_a?(Hash) ? result : { "content" => result.to_s }
        content = result_hash["content"] || result_hash["text"] || result_hash.to_s
        
        # 简化返回结果
        if content.is_a?(Array) && content.first.is_a?(Hash)
          results = content
          results = JSON.parse(content.first["text"]) if content.first["type"] == "text"
          
          if results.is_a?(Array) && results.first.is_a?(Hash)
            formatted = results.first(5).map.with_index(1) do |r, i|
              "#{i}. #{r["title"] || r["name"]}\n   #{r["link"] || r["url"]}\n   #{r["snippet"] || r["description"]}"
            end.join("\n\n")
            return "搜索结果:\n#{formatted}"
          end
        end
        
        content.to_s
      rescue
        "搜索完成，但无法解析结果"
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

      # 加载 Skill 系统
      def load_skill_system
        require_relative "../skill"
      rescue => e
        say "⚠️  Failed to load skill system: #{e.message}", :yellow
      end

      # 加载并激活所有 skills
      def load_and_activate_skills
        skills_dir = File.expand_path("~/smart_ai/smart_bot/skills")
        
        # 加载所有 skill 文件（原生 Ruby + Markdown Skills）
        SmartBot::Skill.load_all(skills_dir)
        
        # 激活所有已注册的 skills
        SmartBot::Skill.activate_all!
        
        # 简明显示加载数量
        loaded_count = SmartBot::Skill.list.length
        say "   Skills loaded: #{loaded_count}", :green if loaded_count > 0
      rescue => e
        say "⚠️  Failed to load skills: #{e.message}", :yellow
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

      # 检测用户是否明确指定了某个 skill
      # 例如："用youtube_downloader下载视频" 或 "使用 invoice_organizer 整理发票"
      def detect_explicit_skill(message)
        # 匹配模式：
        # - 用xxx skill
        # - 使用xxx
        # - 调用xxx
        # - 通过xxx
        patterns = [
          /(?:用|使用|调用)\s*([a-z_][a-z0-9_-]*)/i,
          /(?:用|使用|调用)\s*([a-z_][a-z0-9_-]*)\s*skill/i,
          /([a-z_][a-z0-9_-]*)\s+skill/i,
          /(?:via|using|with)\s+([a-z_][a-z0-9_-]*)/i
        ]
        
        patterns.each do |pattern|
          if match = message.match(pattern)
            skill_name = match[1].downcase.strip
            # 验证是否存在于已注册的 skills 中（支持 Symbol 和 String 两种 key）
            return skill_name if SmartBot::Skill.find(skill_name.to_sym) || SmartBot::Skill.find(skill_name)
          end
        end
        
        nil
      end

      # 模糊查找技能 - 基于关键词匹配描述、名称和标签
      def fuzzy_find_skill(query, limit = 5)
        return [] if query.nil? || query.strip.empty?
        
        query = query.downcase.strip
        query_words = query.split(/[\s,，。！？?]+/).reject { |w| w.empty? }
        
        # 识别关键动作词 - 中英文映射
        action_keywords = {
          "download" => ["download", "下载", "save", "保存", "get", "获取"],
          "search" => ["search", "搜索", "find", "查找", "query", "查询"],
          "weather" => ["weather", "天气", "temperature", "温度"],
          "video" => ["video", "视频", "youtube", "bilibili", "tiktok", "抖音"],
          "audio" => ["audio", "音频", "music", "音乐", "sound", "声音", "mp3"],
          "image" => ["image", "图片", "photo", "照片", "picture", "图"],
          "transcribe" => ["transcribe", "转录", "transcript", "字幕", "transcription"],
          "analyze" => ["analyze", "分析", "analysis", "统计", "analytics"],
          "convert" => ["convert", "转换", "transform", "格式化", "format"],
          "send" => ["send", "发送", "email", "邮件", "message", "消息"]
        }
        
        # 扩展查询词 - 添加语义相关词
        expanded_words = query_words.dup
        query_words.each do |word|
          action_keywords.each do |action, keywords|
            if keywords.include?(word)
              expanded_words << action unless expanded_words.include?(action)
              # 添加同义词组中的其他词
              expanded_words.concat(keywords.reject { |k| k == word })
            end
          end
        end
        expanded_words.uniq!
        
        skills = SmartBot::Skill.registry
        matches = []
        
        skills.each do |name, skill|
          name_str = name.to_s.downcase
          desc = skill.description.to_s.downcase
          
          # 计算匹配分数
          score = 0
          
          # 1. 名称完全匹配 (最高优先级)
          score += 200 if name_str == query
          
          # 2. 名称包含完整查询词
          score += 100 if name_str.include?(query)
          
          # 3. 多个查询词都匹配名称（重要！）
          name_matches = query_words.count { |w| w.length >= 2 && name_str.include?(w) }
          score += name_matches * 60
          
          # 4. 查询词包含名称（短名称匹配）
          score += 50 if query.include?(name_str) && name_str.length > 2
          
          # 5. 描述包含完整查询
          score += 40 if desc.include?(query)
          
          # 6. 扩展词匹配（处理中英文语义）
          expanded_words.each do |word|
            next if word.length < 2
            
            # 名称匹配权重更高
            if name_str.include?(word)
              score += 35
            end
            
            # 描述匹配 - 加权
            if desc.include?(word)
              score += 20
              # 描述开头的匹配权重更高
              score += 25 if desc.start_with?(word)
              # 在 "Use when" 或 "Triggers on" 语句中的匹配
              score += 30 if desc =~ /use when.*#{word}/ || desc =~ /triggers on.*#{word}/
            end
          end
          
          # 7. 原始查询词匹配（基础分）
          query_words.each do |word|
            next if word.length < 2
            score += 10 if name_str.include?(word)
            score += 5 if desc.include?(word)
          end
          
          # 8. 关键词提取匹配（从描述中提取的关键词）
          keywords = extract_keywords(desc)
          query_keywords = extract_keywords(query)
          common = keywords & query_keywords
          score += common.length * 25
          
          # 9. 工具名称匹配
          skill.tools.each do |tool|
            tool_name = tool[:name].to_s.downcase
            score += 35 if tool_name.include?(query)
            query_words.each do |word|
              next if word.length < 2
              score += 15 if tool_name.include?(word)
            end
            # 扩展词匹配
            expanded_words.each do |word|
              next if word.length < 2
              score += 20 if tool_name.include?(word)
            end
          end
          
          # 10. 动作语义匹配 - 检测查询中的动作意图
          action_keywords.each do |action, keywords|
            if keywords.any? { |k| query.include?(k) }
              # skill 名称或描述包含相关动作
              if name_str.include?(action) || desc.include?(action)
                score += 60
              end
              # 检查工具名称
              skill.tools.each do |tool|
                if tool[:name].to_s.downcase.include?(action)
                  score += 40
                end
              end
            end
          end
          
          matches << { name: name, skill: skill, score: score } if score > 0
        end
        
        # 按分数排序并返回前 N 个
        matches.sort_by { |m| -m[:score] }.first(limit)
      end
      
      # 提取关键词（简单的 TF-IDF 近似）
      def extract_keywords(text)
        # 常见停用词
        stopwords = %w[a an and are as at be by for from has he in is it its of on that the to was will with 的 是 在 和 了 有 我 他 她 它 你 这 那 个 上 下 中 就 都 而 及 与 或 等]
        
        # 提取单词（包括中文）
        words = text.downcase.scan(/[a-z]+|[\u4e00-\u9fa5]/)
        words.reject { |w| stopwords.include?(w) || w.length < 2 }
      end
      
      # 智能技能推荐 - 结合模糊匹配和 LLM 选择
      def smart_skill_suggest(message, llm_name, limit = 3)
        # 首先尝试显式指定
        explicit = detect_explicit_skill(message)
        return [{ name: explicit, confidence: :explicit }] if explicit
        
        # 模糊匹配获取候选
        candidates = fuzzy_find_skill(message, 10)
        return [] if candidates.empty?
        
        # 如果只有一个高置信度匹配，直接返回
        return [{ name: candidates.first[:name], confidence: :high }] if candidates.first[:score] >= 80
        
        # 如果有多个候选，使用 LLM 进行选择
        if candidates.length > 1 && candidates.first[:score] >= 30
          # 构建候选列表
          candidate_list = candidates.first(limit).map do |c|
            "- #{c[:name]}: #{c[:skill].description}"
          end.join("\n")
          
          selection_prompt = <<~PROMPT
            用户请求: #{message}

            候选技能（按相关度排序）:
            #{candidate_list}

            请判断哪个技能最适合处理用户的请求。
            如果没有任何技能匹配，请回复 "none"。
            如果有匹配的技能，请只回复技能名称。
            只输出技能名称，不要解释。
          PROMPT

          begin
            engine = SmartPrompt::Engine.new(File.expand_path("~/.smart_bot/smart_prompt.yml"))
            
            worker_name = :"skill_selector_#{Time.now.to_i}"
            SmartPrompt.define_worker worker_name do
              use llm_name
              sys_msg "You are a skill selector. Choose the best skill for the user's request."
              prompt params[:text]
              send_msg
            end

            selected = engine.call_worker(worker_name, { text: selection_prompt }).strip.downcase
            
            if selected != "none" && !selected.empty?
              # 标准化名称
              selected = selected.gsub(/[^a-z0-9_]/, "_").gsub(/_+/, "_").gsub(/^_+|_$/, "")
              # 验证存在
              if SmartBot::Skill.find(selected.to_sym) || SmartBot::Skill.find(selected)
                return [{ name: selected, confidence: :llm_selected }]
              end
            end
          rescue => e
            SmartBot.logger&.debug "LLM skill selection failed: #{e.message}"
          end
        end
        
        # 返回最佳模糊匹配
        candidates.first(limit).map { |c| { name: c[:name], confidence: :fuzzy, score: c[:score] } }
      end
      
      # 根据 skill 名称直接调用
      def call_skill_by_name(skill_name, message, urls, llm_name)
        # 支持 Symbol 和 String 两种 key
        skill = SmartBot::Skill.find(skill_name.to_sym) || SmartBot::Skill.find(skill_name)
        unless skill
          SmartBot.logger&.debug "Skill not found: #{skill_name}"
          return nil
        end

        say "🛠️  正在使用技能: #{skill_name}", :cyan

        # 首先尝试查找真正的脚本工具（有对应的脚本文件在 scripts/ 目录）
        config = skill.config rescue {}
        skill_path = config[:skill_path]
        scripts_dir = skill_path ? File.join(skill_path, "scripts") : nil
        
        script_tools = skill.tools.reject { |t| t[:name].to_s.end_with?('_agent') }
        
        # 检查是否是真正的脚本工具（有对应的脚本文件）
        real_script_tool = script_tools.find do |t|
          tool_name = t[:name].to_s
          # 检查 scripts 目录下是否有对应的脚本文件
          if scripts_dir && Dir.exist?(scripts_dir)
            # 提取基础名称（去掉 skill 前缀）
            base_name = tool_name.to_s.sub(/^#{skill_name}_/, "")
            Dir.glob(File.join(scripts_dir, "*")).any? { |f| File.basename(f, ".*") == base_name }
          else
            false
          end
        end
        
        if real_script_tool
          # 有真正的脚本工具，构建参数并执行
          tool_name = real_script_tool[:name]
          
          tool = SmartAgent::Tool.find_tool(tool_name)
          unless tool
            SmartBot.logger&.debug "Script tool not found: #{tool_name}"
            return nil
          end

          # 构建脚本参数
          url = urls.first || ""
          args = url
          
          say "📜 执行脚本: #{tool_name}", :cyan
          
          result = tool.call({ "args" => args })
          
          if result.is_a?(Hash)
            if result[:success]
              return "✅ 执行成功\n\n#{result[:output]}"
            else
              return "❌ 执行失败 (exit code: #{result[:exit_code]})\n\n#{result[:error]}"
            end
          else
            return result.to_s
          end
        else
          # 没有真正的脚本工具，尝试调用第一个非 _agent 工具或 _agent 工具
          # 优先尝试非 _agent 工具（如 smart_search）
          target_tool = script_tools.first
          
          # 如果没有非 _agent 工具，尝试 _agent 工具
          unless target_tool
            agent_tool_name = :"#{skill_name}_agent"
            target_tool = skill.tools.find { |t| t[:name] == agent_tool_name || t[:name].to_s == agent_tool_name.to_s }
          end
          
          unless target_tool
            SmartBot.logger&.debug "No suitable tool found for skill: #{skill_name}"
            return nil
          end

          tool = SmartAgent::Tool.find_tool(target_tool[:name])
          unless tool
            SmartBot.logger&.debug "SmartAgent tool not found: #{target_tool[:name]}"
            return nil
          end

          # 构建调用参数
          tool_name = target_tool[:name].to_s
          
          if tool_name.end_with?('_agent')
            # 调用 agent 工具
            context = urls.any? ? "包含的URL: #{urls.join(', ')}" : ""
            result = tool.call({ 
              "task" => message,
              "context" => context
            })
          else
            # 调用普通工具（如 smart_search）
            # 提取搜索关键词或任务
            query = message.gsub(/用#{skill_name}/, "").gsub(/#{skill_name}/, "").strip
            query = urls.first if urls.any? && query.empty?
            
            say "🔍 执行: #{tool_name}", :cyan
            
            result = tool.call({ 
              "query" => query,
              "count" => 5
            })
          end

          if result.is_a?(Hash)
            if result[:error]
              return "❌ 执行失败: #{result[:error]}"
            elsif result[:results]
              # 格式化搜索结果
              results_text = result[:results].map.with_index(1) do |r, i|
                "#{i}. #{r[:title]}\n   #{r[:url]}\n   #{r[:description]}"
              end.join("\n\n")
              return "搜索结果:\n#{results_text}"
            else
              return result.to_s
            end
          else
            return result.to_s
          end
        end
      rescue => e
        SmartBot.logger&.warn "Skill execution failed: #{e.message}"
        SmartBot.logger&.warn e.backtrace.first(5).join("\n")
        nil
      end

      # 尝试使用 Markdown Skills
      # 根据用户输入匹配合适的 skill 并调用
      def try_markdown_skills(message, llm_name)
        # 获取所有已注册的 skills
        skills = SmartBot::Skill.registry
        return nil if skills.empty?

        # 构建技能列表和描述
        skill_descriptions = skills.map do |name, skill|
          "- #{name}: #{skill.description}"
        end.join("\n")

        # 创建一个简单的匹配提示词
        selection_prompt = <<~PROMPT
          用户输入: #{message}

          可用技能:
          #{skill_descriptions}

          请判断哪个技能最适合处理用户的请求。
          如果没有任何技能匹配，请回复 "none"。
          如果有匹配的技能，请只回复技能名称（如：search, weather, invoice_organizer）。
          只输出技能名称，不要解释。
        PROMPT

        # 调用 LLM 选择技能
        engine = SmartPrompt::Engine.new(File.expand_path("~/.smart_bot/smart_prompt.yml"))
        
        worker_name = :"skill_selector_#{Time.now.to_i}"
        SmartPrompt.define_worker worker_name do
          use llm_name
          sys_msg "You are a skill selector. Choose the best skill for the user's request."
          prompt params[:text]
          send_msg
        end

        selected_skill = engine.call_worker(worker_name, { text: selection_prompt }).strip.downcase
        
        # 如果没有匹配，返回 nil
        return nil if selected_skill == "none" || selected_skill.empty?
        
        # 标准化技能名称
        selected_skill = selected_skill.gsub(/[^a-z0-9_]/, "_").gsub(/_+/, "_").gsub(/^_+|_$/, "")
        
        # 查找对应的 skill（支持 Symbol 和 String 两种 key）
        skill = skills[selected_skill.to_sym] || skills[selected_skill]
        return nil unless skill

        # 查找该 skill 的 _agent 工具
        agent_tool_name = :"#{selected_skill}_agent"
        agent_tool = skill.tools.find { |t| t[:name] == agent_tool_name || t[:name].to_s == agent_tool_name.to_s }
        
        return nil unless agent_tool

        # 调用 skill 的 agent 工具
        say "🛠️  正在使用技能: #{selected_skill}", :cyan
        
        tool = SmartAgent::Tool.find_tool(agent_tool_name)
        return nil unless tool

        result = tool.call({ 
          "task" => message,
          "context" => ""
        })

        result[:result] if result.is_a?(Hash) && result[:result]
      rescue => e
        SmartBot.logger&.warn "Markdown skill execution failed: #{e.message}"
        nil
      end

      # 处理斜杠命令
      def handle_command(input, config, current_llm)
        cmd, *args = input.split
        
        case cmd
        when "/help"
          say "\n📖 Commands:"
          say "  /models              - List available LLMs"
          say "  /llm <name>          - Switch LLM provider"
          say "  /skills [offset]     - List skills (default: first 40)"
          say "  /find <keyword>      - Search skills by keyword"
          say "  /skill_help <name>   - Show detailed help for a skill"
          say "  /new                 - Start a new conversation (clear history)"
          say "  /help                - Show this help"
          say "  Ctrl+C               - Exit\n"
          
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
          
        when "/skills"
          # 解析分页参数: /skills [offset]
          offset = args.first.to_i
          offset = 0 if offset < 0

          all_skills = SmartBot::Skill.registry.to_a
          total = all_skills.length

          if total == 0
            say "\n🛠️  No skills loaded", :yellow
          else
            per_page = 40
            start_idx = offset
            end_idx = [offset + per_page, total].min

            say "\n🛠️  Skills (#{start_idx + 1}-#{end_idx} of #{total}):\n"

            all_skills[start_idx...end_idx].each do |name, skill|
              desc = skill.description.to_s[0..60]
              desc += "..." if skill.description.to_s.length > 60
              say "  • #{set_color(name.to_s, :green)} - #{desc}"
            end

            # 显示分页提示
            if end_idx < total
              say "\n  ... and #{total - end_idx} more"
              say "  Use /skills #{end_idx} to see more"
            end
            say ""
          end

        when "/find"
          if args.empty?
            say "Usage: /find <keyword>", :yellow
            say "Example: /find download   # 搜索下载相关技能"
            say "         /find youtube    # 搜索 YouTube 相关技能"
            say "         /find 天气        # 搜索天气相关技能"
            return
          end

          keyword = args.join(" ")
          matches = fuzzy_find_skill(keyword, 10)

          if matches.empty?
            say "\n🔍 No skills found matching '#{keyword}'", :yellow
            say "Try different keywords or use /skills to browse all"
          else
            say "\n🔍 Skills matching '#{keyword}' (top #{matches.length}):\n"
            matches.each_with_index do |match, idx|
              name = match[:name]
              skill = match[:skill]
              score = match[:score]
              desc = skill.description.to_s[0..70]
              desc += "..." if skill.description.to_s.length > 70

              # 根据分数显示不同颜色
              color = score >= 80 ? :green : (score >= 40 ? :yellow : :dim)
              confidence = score >= 80 ? "★★★" : (score >= 40 ? "★★" : "★")

              say "  #{confidence} #{set_color(name.to_s, color, :bold)}"
              say "     #{desc}"
              say ""
            end
            say "Use '#{set_color("用 <skill_name> ", :cyan)}<your task>' to use a skill"
          end
          say ""

        when "/skill_help"
          if args.empty?
            say "Usage: /skill_help <skill_name>", :yellow
            say "Example: /skill_help youtube_downloader"
            return
          end
          
          skill_name = args.first
          # 支持 Symbol 和 String 两种 key
          skill = SmartBot::Skill.find(skill_name.to_sym) || SmartBot::Skill.find(skill_name)
          
          unless skill
            say "❌ Skill '#{skill_name}' not found", :red
            say "Use /skills to list available skills"
            return
          end
          
          say "\n📚 Skill: #{set_color(skill_name.to_s, :green, :bold)}\n"
          say "Description: #{skill.description}"
          say "Version: #{skill.version}"
          say "Author: #{skill.author}"
          
          if skill.tools.any?
            say "\nTools:"
            skill.tools.each do |tool|
              tool_desc = tool[:desc] || tool[:description] || "No description"
              say "  • #{tool[:name]} - #{tool_desc}"
            end
          end
          
          # 尝试读取 SKILL.md 文件
          config = skill.config rescue {}
          skill_path = config[:skill_path]
          
          if skill_path
            skill_md = File.join(skill_path, "SKILL.md")
            if File.exist?(skill_md)
              say "\n📖 SKILL.md Content:\n"
              content = File.read(skill_md)
              # 跳过 YAML frontmatter
              if content =~ /\A---\s*\n.*\n---\s*\n(.*)/m
                body = $1
                # 显示前 1000 字符
                preview = body[0..1000].strip
                say preview
                say "\n... (truncated)" if body.length > 1000
              else
                preview = content[0..1000].strip
                say preview
                say "\n... (truncated)" if content.length > 1000
              end
            end
          end
          
          say ""
          
        else
          say "Unknown command: #{cmd}. Type /help for available commands.", :yellow
        end
      end

      # Skill 模板
      def skill_template(name, options)
        class_name = name.split('_').map(&:capitalize).join
        <<~TEMPLATE
# frozen_string_literal: true

# #{class_name} Skill - #{options[:description]}

SmartBot::Skill.register :#{name} do
  desc "#{options[:description]}"
  ver "0.1.0"
  author_name "#{options[:author]}"

  # 注册工具示例
  # register_tool :#{name}_tool do
  #   desc "Description of what this tool does"
  #   param_define :param1, "Parameter description", :string
  #   
  #   tool_proc do
  #     # Tool implementation
  #     { result: "success" }
  #   end
  # end

  # 激活时的配置
  on_activate do
    SmartAgent.logger&.info "#{name} skill activated!"
  end
end
        TEMPLATE
      end

      def skill_md_template(name, options)
        class_name = name.split('_').map(&:capitalize).join
        <<~TEMPLATE
# #{class_name} Skill

#{options[:description]}

## Usage

```ruby
# Add usage examples here
SmartAgent::Tool.call(:your_tool_name, { "param" => "value" })
```

## CLI Usage

```bash
smart_bot agent -m "your command here"
```

## Configuration

Add configuration instructions here.

## Author

#{options[:author]}
        TEMPLATE
      end
    end
  end
end
