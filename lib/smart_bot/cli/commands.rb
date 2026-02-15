# frozen_string_literal: true

require "thor"
require "yaml"
require "open3"
require "timeout"
require "shellwords"
require "json"

# Load enhanced command execution system
require_relative "../skill_system/execution/enhanced_command_runner"

module SmartBot
  module CLI
    class Commands < Thor
      DEFAULT_SYSTEM_LANGUAGE = "简体中文"

      desc "agent", "Interact with the agent"
      option :message, aliases: "-m", desc: "Message to send"
      option :session, aliases: "-s", default: "cli:default", desc: "Session ID"
      option :llm, aliases: "-l", desc: "LLM to use"
      def agent
        @interactive_agent_mode = options[:message].nil?
        @smart_prompt_config_path = File.expand_path("~/.smart_bot/smart_prompt.yml")

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
        
        # 初始化新的 Skill System
        load_new_skill_system
        
        # 获取当前配置
        smart_prompt_config = load_smart_prompt_config
        current_llm = options[:llm] || smart_prompt_config["default_llm"] || "deepseek"
        @system_language = configured_system_language(smart_prompt_config)
        
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
          conversation.sys_msg(default_system_prompt(@system_language), { with_history: true })

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
                  conversation.sys_msg(default_system_prompt(@system_language), { with_history: true })
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
          say "System Language: #{config['system_language'] || DEFAULT_SYSTEM_LANGUAGE}"
          
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

      desc "language [LANG]", "Show or set preferred conversation language"
      def language(lang = nil)
        config = load_smart_prompt_config

        if lang.nil? || lang.strip.empty?
          say "Current system language: #{set_color(configured_system_language(config), :green)}"
          say "Usage: smart_bot language <LANG>"
          return
        end

        language_value = normalize_language(lang)
        unless valid_language?(language_value)
          say "❌ Invalid language. Use letters, numbers, spaces, '-' or '_'.", :red
          return
        end

        config["system_language"] = language_value
        save_smart_prompt_config(config)
        @system_language = language_value

        say "✓ System language set to: #{set_color(language_value, :green)}"
      rescue => e
        say "❌ Failed to update language: #{e.message}", :red
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

      desc "skill create NAME", "Create a new skill"
      option :description, aliases: "-d", default: "A new skill", desc: "Skill description"
      option :author, aliases: "-a", default: "SmartBot User", desc: "Author name"
      def skill_create(name)
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

      # Register SkillCommands as subcommand
      require_relative "skill_commands"
      register SkillCommands, "skill", "skill [COMMAND]", "Manage skills"

      private

      # 主要的对话逻辑 - 手动处理工具调用
      def chat_with_tools(message, llm_name)
        # 检查是否需要调用工具
        url_pattern = %r{https?://[^\s]+}
        urls = message.scan(url_pattern)

        # ========== 0. 新的 Skill System 路由 ==========
        skill_system_result = try_skill_system_route(message, llm_name)
        return skill_system_result if skill_system_result

        # ========== 1. 显式 run_skill 委派 ==========
        delegated = try_run_skill_delegation(message, llm_name)
        return delegated if delegated

        # ========== 2. 优先检查是否明确指定了 Skill ==========
        # 检查用户是否明确提到了某个 skill 名称
        explicit_skill = detect_explicit_skill(message)
        if explicit_skill
          skill_result = call_skill_by_name(explicit_skill, message, urls, llm_name)
          return skill_result if skill_result
        end

        # ========== 3. 智能 Skill 推荐 ==========
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
        
        # ========== 3. 搜索请求 ==========
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
        language_key = current_system_language.downcase.gsub(/[^a-z0-9]+/, "_").gsub(/^_+|_+$/, "")
        language_key = "lang" if language_key.empty?
        worker_name = :"temp_chat_#{llm_name}_#{language_key}"
        
        # 只在未定义时创建 worker
        unless SmartPrompt::Worker.workers.key?(worker_name)
          SmartPrompt.define_worker worker_name do
            use llm_name
            sys_msg default_system_prompt(current_system_language)
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

        # ========== 0. 新的 Skill System 路由 ==========
        skill_system_result = try_skill_system_route(message, llm_name)
        return skill_system_result if skill_system_result

        # 显式 run_skill 委派（不进入会话历史）
        delegated = try_run_skill_delegation(message, llm_name)
        return delegated if delegated

        # 检查是否是特殊命令（搜索、天气等）
        # 这些仍然使用即时工具调用，不进入对话历史

        # ========== 1. 显式 Skill 调用 ==========
        explicit_skill = detect_explicit_skill(message)
        if explicit_skill
          skill_result = call_skill_by_name(explicit_skill, message, urls, llm_name)
          return skill_result if skill_result
        end
        
        # ========== 2. 搜索请求 ==========
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
            conversation.add_message({ role: "user", content: with_language_instruction("#{message}\n\n[网页内容]\n#{context}") }, true)
            response = conversation.send_msg(with_history: true)
            # 将助手回复也加入历史
            conversation.add_message({ role: "assistant", content: response }, true)
            return response
          end
        end
        
        # ========== 5. 普通对话（使用 Conversation 维护历史）==========
        # 添加用户消息到历史（使用 with_history: true）
        conversation.add_message({ role: "user", content: with_language_instruction(message) }, true)
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

      # 解析 run_skill 语法并执行委派
      # 支持:
      # - run_skill skill_name: task details
      # - run_skill skill_name task details
      # - 用run_skill 调用 skill_name: task details
      def try_run_skill_delegation(message, llm_name)
        payload = parse_run_skill_request(message)
        return nil unless payload

        execute_run_skill(
          skill_name: payload[:skill_name],
          task: payload[:task],
          max_depth: payload[:max_depth],
          chain: payload[:chain],
          parent_skill: payload[:parent_skill],
          llm_name: llm_name
        )
      end

      def parse_run_skill_request(message)
        text = message.to_s.strip
        return nil if text.empty?

        # run_skill <skill>[: ]<task>
        pattern = /(?:^|\s)(?:用\s*)?run_skill\s+([a-z_][a-z0-9_-]*)\s*(?::|\s)\s*(.+)\z/i
        match = text.match(pattern)
        return nil unless match

        skill_name = normalize_skill_name(match[1])
        task_text = match[2].to_s.strip
        return nil if skill_name.empty? || task_text.empty?

        max_depth = nil
        # 可选参数: "max_depth=3"
        if task_text =~ /\bmax_depth\s*=\s*(\d+)\b/i
          max_depth = Regexp.last_match(1).to_i
          task_text = task_text.sub(/\bmax_depth\s*=\s*\d+\b/i, "").strip
        end

        {
          skill_name: skill_name,
          task: task_text,
          max_depth: max_depth,
          chain: nil,
          parent_skill: nil
        }
      end

      def execute_run_skill(skill_name:, task:, llm_name:, parent_skill: nil, chain: nil, max_depth: nil)
        current_chain = parse_chain(chain)
        normalized_skill = normalize_skill_name(skill_name)
        return "run_skill error: invalid skill_name" if normalized_skill.empty?
        return "run_skill error: task is required" if task.to_s.strip.empty?

        current_chain << normalize_skill_name(parent_skill) unless parent_skill.to_s.strip.empty?
        current_chain = current_chain.reject(&:empty?)

        effective_max_depth = max_depth.to_i > 0 ? max_depth.to_i : 2

        if current_chain.include?(normalized_skill)
          cycle = (current_chain + [normalized_skill]).join(" -> ")
          return "run_skill error: delegation cycle detected (#{cycle})"
        end

        if current_chain.length >= effective_max_depth
          return "run_skill error: delegation depth limit reached (max_depth=#{effective_max_depth})"
        end

        # 验证 skill 存在
        skill = SmartBot::Skill.find(normalized_skill.to_sym) || SmartBot::Skill.find(normalized_skill)
        return "run_skill error: skill not found: #{normalized_skill}" unless skill

        say "🔁 run_skill -> #{normalized_skill}", :cyan

        grounded_task = build_grounding_guarded_task(task)
        task_urls = grounded_task.scan(%r{https?://[^\s]+})
        result = call_skill_by_name(
          normalized_skill,
          grounded_task,
          task_urls,
          llm_name,
          require_evidence: true
        )
        return "run_skill error: delegated skill execution failed: #{normalized_skill}" if result.nil?
        if result.to_s.start_with?("❌ 该技能")
          return "run_skill error: #{result}"
        end

        # Generic anti-hallucination guard:
        # If output is too assertive without required evidence structure,
        # force one corrective retry with stricter constraints.
        if grounding_structure_missing?(result.to_s) && grounding_risky_claims?(result.to_s)
          retry_task = build_grounding_retry_task(original_task: task, previous_output: result.to_s)
          retry_urls = retry_task.scan(%r{https?://[^\s]+})
          retried = call_skill_by_name(
            normalized_skill,
            retry_task,
            retry_urls,
            llm_name,
            require_evidence: true
          )
          if retried
            if retried.to_s.start_with?("❌ 该技能")
              return "run_skill error: #{retried}"
            end
            result = "⚠️ 首次结果缺少可验证依据，已自动触发一次防幻觉重试。\n\n#{retried}"
          end
        end

        next_chain = current_chain + [normalized_skill]
        <<~TEXT.strip
          run_skill delegated: #{normalized_skill}
          chain: #{next_chain.join(" -> ")}

          #{result}
        TEXT
      end

      def parse_chain(chain)
        return [] if chain.nil?
        return chain.map { |item| normalize_skill_name(item) } if chain.is_a?(Array)
        return [] unless chain.is_a?(String)

        chain.split(/\s*(?:->|>)\s*/).map { |item| normalize_skill_name(item) }
      end

      def normalize_skill_name(name)
        name.to_s.strip.downcase.gsub(/[^a-z0-9_]/, "_").gsub(/_+/, "_").gsub(/^_+|_+$/, "")
      end

      def execute_skill_via_markdown(skill_name:, skill:, task:, urls:, llm_name:)
        config = skill.config rescue {}
        skill_path = config[:skill_path]
        return nil unless skill_path

        skill_md = File.join(skill_path, "SKILL.md")
        return nil unless File.exist?(skill_md)

        content = File.read(skill_md, encoding: "UTF-8")
        commands = extract_bash_commands(content)
        return nil if commands.empty?

        selected = select_relevant_commands(commands, task: task, urls: urls).first(3)
        return nil if selected.empty?

        # Use enhanced command runner for validation, adaptation, and execution
        runner = SkillSystem::Execution::EnhancedCommandRunner.new(
          require_confirmation: false,
          timeout: 30
        )

        executed = []
        blocked = []

        selected.each do |cmd|
          prepared_cmd = prepare_command_for_task(cmd, urls: urls)

          unless command_allowed_for_evidence?(prepared_cmd)
            blocked << { command: cmd, reason: "blocked by safety filter" }
            next
          end

          # Use enhanced execution with validation and retry
          context = { urls: urls, task: task, interactive: false }
          result = runner.run(prepared_cmd, context)

          if result[:success]
            executed << {
              ok: true,
              exit_code: 0,
              stdout: result[:stdout].to_s,
              stderr: result[:stderr].to_s,
              command: result[:command] || prepared_cmd,
              original_command: cmd,
              adaptations: result[:adaptations]
            }
          else
            executed << {
              ok: false,
              exit_code: -1,
              stdout: "",
              stderr: result[:error].to_s,
              command: result[:command] || prepared_cmd,
              original_command: cmd,
              error_stage: result[:stage]
            }
          end
        end

        return nil if executed.empty?

        summarize_evidence_execution(
          skill_name: skill_name,
          task: task,
          extracted_commands: commands,
          selected_commands: selected,
          blocked_commands: blocked,
          executed: executed,
          llm_name: llm_name
        )
      rescue => e
        "❌ 证据执行流程失败: #{e.message}"
      end

      def extract_bash_commands(skill_md_content)
        skill_md_content.scan(/```bash\s*\n(.*?)```/m).flatten.map(&:strip).reject(&:empty?)
      end

      def select_relevant_commands(commands, task:, urls:)
        url_present = urls.any?
        keywords = task.to_s.downcase.scan(/[a-z0-9_]+|[\u4e00-\u9fa5]+/)

        scored = commands.map do |cmd|
          lower = cmd.downcase
          score = 0
          score += 4 if url_present && (lower.include?("video_url") || lower.include?("youtube") || lower.include?("youtu.be"))
          score += 3 if lower.include?("dump-json") || lower.include?("--list-subs") || lower.include?("--write-auto-sub")
          score += 2 if lower.include?("python3") || lower.include?("sed ")
          score += keywords.count { |k| k.length > 1 && lower.include?(k) }
          [cmd, score]
        end

        scored.sort_by { |(_, s)| -s }.map(&:first)
      end

      def command_allowed_for_evidence?(cmd)
        blocked_patterns = [
          /\brm\b/i,
          /\bsudo\b/i,
          /\bapt(-get)?\b/i,
          /\byum\b/i,
          /\bbrew\b/i,
          /\bchoco\b/i,
          /\bpip\s+install\b/i,
          /\bnpm\s+install\b/i,
          /\bgit\s+clone\b/i,
          /\bcurl\b.*\|\s*(sh|bash)/i,
          /\bwget\b.*\|\s*(sh|bash)/i
        ]
        blocked_patterns.none? { |p| cmd.match?(p) }
      end

      def prepare_command_for_task(cmd, urls:)
        prepared = cmd.dup
        if urls.any?
          escaped_url = Shellwords.escape(urls.first)
          prepared = prepared.gsub("VIDEO_URL", escaped_url)

          video_id = extract_youtube_video_id(urls.first)
          prepared = prepared.gsub("VIDEO_ID", video_id) if video_id
        end
        prepared
      end

      def extract_youtube_video_id(url)
        return nil if url.to_s.strip.empty?
        u = url.to_s

        if (m = u.match(/[?&]v=([A-Za-z0-9_-]{11})/))
          return m[1]
        end
        if (m = u.match(%r{youtu\.be/([A-Za-z0-9_-]{11})}))
          return m[1]
        end
        if (m = u.match(%r{/shorts/([A-Za-z0-9_-]{11})}))
          return m[1]
        end
        nil
      end

      def run_evidence_command(command, timeout_sec: 30)
        stdout = ""
        stderr = ""
        status = nil

        Timeout.timeout(timeout_sec) do
          stdout, stderr, status = Open3.capture3("bash", "-lc", command)
        end

        {
          ok: status&.success? || false,
          exit_code: status&.exitstatus,
          stdout: stdout.to_s,
          stderr: stderr.to_s
        }
      rescue Timeout::Error
        {
          ok: false,
          exit_code: nil,
          stdout: "",
          stderr: "Command timed out after #{timeout_sec}s"
        }
      end

      def summarize_evidence_execution(skill_name:, task:, extracted_commands:, selected_commands:, blocked_commands:, executed:, llm_name:)
        facts = extract_key_value_facts(executed)
        successful_commands = executed.count { |e| e[:ok] }
        verified_facts_count = facts.length

        verified = if facts.empty?
                     "- No structured facts extracted from command output."
                   else
                     facts.map { |k, v| "- #{k}: #{v}" }.join("\n")
                   end

        unknown = if facts.empty?
                    "- Unable to verify key facts from command output."
                  else
                    "- Any fact not listed above remains Unverified."
                  end

        extracted = extracted_commands.map.with_index(1) do |cmd, idx|
          "#{idx}. #{cmd}"
        end.join("\n")

        selected = selected_commands.map.with_index(1) do |cmd, idx|
          "#{idx}. #{cmd}"
        end.join("\n")

        blocked = if blocked_commands.empty?
                    "(none)"
                  else
                    blocked_commands.map.with_index(1) do |item, idx|
                      "#{idx}. #{item[:command]}\n   reason: #{item[:reason]}"
                    end.join("\n")
                  end

        steps = executed.map.with_index(1) do |e, idx|
          <<~STEP
            #{idx}. original_command: #{e[:original_command]}
               prepared_command: #{e[:command]}
               exit_code: #{e[:exit_code]} (ok=#{e[:ok]})
               stdout:
            #{indent_multiline(e[:stdout].to_s, 6)}
               stderr:
            #{indent_multiline(e[:stderr].to_s, 6)}
          STEP
        end.join("\n")

        <<~TEXT.strip
          Skill: #{skill_name}
          Task: #{task}

          Extracted Commands From SKILL.md
          #{extracted.empty? ? "(none)" : extracted}

          Selected Commands
          #{selected.empty? ? "(none)" : selected}

          Blocked Commands
          #{blocked}

          Evidence Quality
          - successful_commands: #{successful_commands}/#{executed.length}
          - verified_facts_count: #{verified_facts_count}
          - grounded: #{successful_commands > 0 && verified_facts_count > 0 ? "yes" : "no"}

          Verified Facts
          #{verified}

          Unverified / Unknown
          #{unknown}

          What I Actually Executed
          #{steps}
        TEXT
      end

      def indent_multiline(text, spaces)
        pad = " " * spaces
        body = text.to_s
        return "#{pad}(empty)" if body.strip.empty?

        body.lines.map { |line| "#{pad}#{line}" }.join
      end

      def extract_key_value_facts(executed)
        pairs = []
        executed.each do |e|
          [e[:stdout], e[:stderr]].each do |text|
            json_facts = extract_facts_from_json(text)
            pairs.concat(json_facts) if json_facts.any?

            text.to_s.each_line do |line|
              m = line.match(/^\s*([A-Za-z][A-Za-z0-9 _\-\/]{1,50})\s*:\s*(.+?)\s*$/)
              next unless m

              key = m[1].strip
              value = m[2].strip
              next if key.empty? || value.empty?
              next if value.length > 300
              pairs << [key, value]
            end
          end
        end

        uniq = {}
        pairs.each do |k, v|
          uniq[k] ||= v
        end
        uniq.to_a.first(20)
      end

      def extract_facts_from_json(text)
        body = text.to_s.strip
        return [] if body.empty?

        parsed = JSON.parse(body)
        return [] unless parsed.is_a?(Hash)

        keys = %w[title uploader channel view_count upload_date duration id webpage_url]
        keys.filter_map do |k|
          value = parsed[k]
          next if value.nil? || value.to_s.strip.empty?
          [k, value.to_s]
        end
      rescue JSON::ParserError, TypeError
        []
      end

      def evidence_grounded_enough?(evidence_text)
        text = evidence_text.to_s
        successful = text[/successful_commands:\s*(\d+)\/\d+/, 1].to_i
        facts = text[/verified_facts_count:\s*(\d+)/, 1].to_i
        grounded_flag = text.match?(/grounded:\s*yes/i)
        (successful > 0 && facts > 0) || grounded_flag
      end

      def build_grounding_guarded_task(task)
        <<~TASK.strip
          #{task}

          Grounding requirements (must follow):
          1. Do not claim facts as verified unless they come from concrete tool output.
          2. Separate output into:
             - Verified Facts (with concrete evidence/source)
             - Unverified / Unknown
             - What I Actually Executed
          3. If evidence is missing, explicitly say "Unknown" rather than guessing.
        TASK
      end

      def build_grounding_retry_task(original_task:, previous_output:)
        <<~TASK.strip
          #{original_task}

          Your previous answer did not provide sufficient grounding and may contain unverified claims.

          Previous answer:
          #{previous_output}

          Rewrite the answer with strict grounding:
          - Only keep claims backed by concrete tool outputs.
          - Mark everything else as Unverified / Unknown.
          - Include a short "What I Actually Executed" section.
          - Do not invent titles, numbers, names, dates, or events.
        TASK
      end

      def grounding_structure_missing?(text)
        normalized = text.to_s
        has_verified = normalized.match?(/verified facts|已验证事实|可验证事实/i)
        has_unknown = normalized.match?(/unverified|unknown|未验证|未知|不确定/i)
        has_executed = normalized.match?(/what i actually executed|实际执行|执行步骤|commands/i)
        !(has_verified && has_unknown && has_executed)
      end

      def grounding_risky_claims?(text)
        normalized = text.to_s
        risky_patterns = [
          /我已(?:获取|确认|验证)/,
          /根据我(?:获取|提取|分析)到/,
          /\bI (?:fetched|verified|confirmed|extracted)\b/i,
          /视频标题\s*[:：]/,
          /metadata|元数据/i
        ]
        risky_patterns.any? { |pattern| normalized.match?(pattern) }
      end

      # run_skill 会为防幻觉在任务末尾附加 grounding 约束。
      # 对脚本工具来说这会污染命令参数，需要在执行前去掉该后缀。
      def strip_grounding_suffix(task_text)
        text = task_text.to_s
        marker = /\n\nGrounding requirements \(must follow\):\n/i
        parts = text.split(marker, 2)
        parts.first.to_s.strip
      end

      # 纯文本改写类技能（如 humanizer/rewrite/translate）可在闭环输入下跳过证据命令阶段
      def allow_text_only_agent_without_evidence?(skill:, message:, urls:)
        return false if urls.any?

        task = message.to_s
        return false if task.strip.empty?

        skill_text = [
          skill.name.to_s,
          skill.respond_to?(:description) ? skill.description.to_s : "",
          (skill.config[:description].to_s rescue "")
        ].join(" ").downcase
        text = task.downcase

        transform_patterns = [
          /humaniz|rewrite|paraphrase|polish|edit|proofread|translate|rephrase/,
          /润色|改写|人性化|去ai|优化文案|校对|翻译|重写/
        ]
        external_patterns = [
          /search|fetch|crawl|scrape|download|youtube|weather|news|stock|price|metadata|transcript|api/,
          /搜索|抓取|下载|视频|天气|新闻|股价|价格|元数据|转录|接口/
        ]

        has_transform_intent = transform_patterns.any? { |p| text.match?(p) || skill_text.match?(p) }
        needs_external_data = external_patterns.any? { |p| text.match?(p) || skill_text.match?(p) }

        payload = task.gsub(%r{https?://[^\s]+}, "").strip
        has_substantial_input = payload.length >= 40 || payload.match?(/[，。！？,.!?].+[，。！？,.!?]/m)

        has_transform_intent && !needs_external_data && has_substantial_input
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

        SmartBot::Skill.load_all(skills_dir)
        SmartBot::Skill.activate_all!
      rescue => e
        # Legacy skill loading failed, continue with new system
      end

      def load_new_skill_system
        require_relative "../skill_system"

        stats = SmartBot::SkillSystem.load_all
        say "   Skill System: #{stats[:available]} skills available", :green if stats[:available] > 0

        router = SmartBot::SkillSystem.router
        if router.semantic_index
          semantic_stats = router.semantic_stats
          say "   Semantic index: #{semantic_stats[:unique_terms]} terms", :blue
        end
      rescue => e
        say "⚠️  Skill system not available: #{e.message}", :yellow
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

      def try_skill_system_route(message, llm_name)
        return nil unless defined?(SmartBot::SkillSystem)
        return nil if SmartBot::SkillSystem.registry.empty?

        begin
          plan = SmartBot::SkillSystem.route(message)
          say "🎯 Skill System routing: plan.empty=#{plan.empty?}", :cyan
          return nil if plan.empty?

          primary_skill = plan.primary_skill
          say "🎯 Primary skill: #{primary_skill&.name || 'nil'}", :cyan
          return nil unless primary_skill

          say "🎯 Skill System matched: #{primary_skill.name}", :cyan

          result = SmartBot::SkillSystem.execute(
            plan,
            context: { llm: llm_name },
            repair_confirmation_callback: skill_repair_confirmation_callback
          )

          if result.success?
            # Format the output nicely
            value = result.value
            if value.is_a?(Hash) && value[:success] && value[:output]
              format_skill_output(value[:output], primary_skill.name)
            elsif value.is_a?(Hash)
              value[:output] || value.to_s
            else
              value.to_s
            end
          else
            say "⚠️ Skill execution failed: #{result.error}", :yellow
            nil
          end
        rescue => e
          say "⚠️ Skill System routing error: #{e.message}", :yellow
          SmartBot.logger&.debug "Skill System routing failed: #{e.message}"
          nil
        end
      end

      def format_skill_output(output, skill_name)
        # Clean up the output for better display
        lines = output.to_s.split("\n")
        
        # Remove progress bar lines (lines with \r)
        lines = lines.reject { |line| line.include?("\r") }
        
        # Remove empty lines at the beginning and end
        lines = lines.drop_while(&:empty?)
        lines = lines.reverse.drop_while(&:empty?).reverse
        
        # Format the output
        formatted = lines.join("\n")
        
        # Add a header
        "📥 Download started by #{skill_name}\n\n#{formatted}"
      end

      def skill_repair_confirmation_callback
        return nil unless @interactive_agent_mode
        return nil unless $stdin.tty? && $stdout.tty?

        method(:confirm_skill_repair)
      end

      def confirm_skill_repair(payload)
        skill_name = payload[:skill]&.name || "unknown"
        attempt = payload[:attempt]
        diagnosis = payload[:diagnosis] || {}
        repair_plan = payload[:repair_plan] || {}
        patches = repair_plan[:patches] || []

        say "\n🩹 Skill '#{skill_name}' 执行失败，准备进行第 #{attempt} 次自动修复。", :yellow
        say "错误类型: #{diagnosis[:error_type] || 'unknown'}", :yellow
        say "错误信息: #{diagnosis[:error_message]}", :yellow if diagnosis[:error_message]
        say "计划补丁:", :cyan
        patches.each_with_index do |patch, index|
          say "  #{index + 1}. #{patch[:file]} (#{patch[:action]}): #{patch[:description]}"
        end

        answer = ask("是否应用以上修复？(y=应用 / n=跳过 / s=提供修复建议)", :yellow).to_s.strip.downcase
        case answer
        when "y", "yes"
          { approved: true }
        when "s", "suggest"
          suggestion = ask("请输入你的修复建议（将追加到 SKILL.md 后重试）:", :yellow).to_s.strip
          { approved: true, suggestion: suggestion }
        else
          { approved: false }
        end
      rescue => e
        say "⚠️ 修复确认失败: #{e.message}", :yellow
        { approved: false }
      end

      def load_smart_prompt_config
        config_path = @smart_prompt_config_path || File.expand_path("~/.smart_bot/smart_prompt.yml")
        return {} unless File.exist?(config_path)

        data = YAML.load_file(config_path)
        data.is_a?(Hash) ? data : {}
      rescue
        {}
      end

      def save_smart_prompt_config(config)
        config_path = @smart_prompt_config_path || File.expand_path("~/.smart_bot/smart_prompt.yml")
        FileUtils.mkdir_p(File.dirname(config_path))
        File.write(config_path, YAML.dump(config))
      end

      def configured_system_language(config = nil)
        source = config || load_smart_prompt_config
        language = source["system_language"].to_s.strip
        language.empty? ? DEFAULT_SYSTEM_LANGUAGE : language
      end

      def current_system_language
        @system_language ||= configured_system_language
      end

      def normalize_language(value)
        value.to_s.strip
      end

      def valid_language?(value)
        return false if value.nil? || value.empty? || value.length > 50
        !!(value =~ /\A[\p{L}\p{N}\s\-_]+\z/u)
      end

      def default_system_prompt(language)
        <<~PROMPT.strip
          You are SmartBot, a helpful AI assistant.
          Remember information the user shares during this conversation.
          Always respond in #{language}, unless the user explicitly asks for a different language.
        PROMPT
      end

      def with_language_instruction(user_text)
        language = current_system_language
        "Please reply in #{language} unless I explicitly request another language.\n\n#{user_text}"
      end

      def render_skill_system_list
        return say("\n⚠️ Skill System not available", :yellow) unless defined?(SmartBot::SkillSystem)

        SmartBot::SkillSystem.load_all if SmartBot::SkillSystem.registry.empty?
        registry = SmartBot::SkillSystem.registry

        say "🛠️  Available Skills\n\n"

        if registry.empty?
          say "No skills found.", :yellow
          return
        end

        available = registry.list_available
        unavailable = registry.reject(&:available?)

        if available.any?
          say "Available (#{available.size}):", :green
          available.each { |skill| display_skill_system_item(skill) }
          say ""
        end

        if unavailable.any?
          say "Unavailable (#{unavailable.size}):", :yellow
          unavailable.each { |skill| display_skill_system_item(skill, available: false) }
        end

        say "\nStats: #{registry.stats}"
      end

      def display_skill_system_item(skill, available: true)
        status = available ? "✓" : "✗"
        color = available ? :green : :yellow
        say "  #{status} #{skill.name} - #{skill.description}", color
      end

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
        query_words = extract_keywords(query)
        skills = SmartBot::Skill.registry
        matches = []

        skills.each do |name, skill|
          name_str = name.to_s.downcase
          desc = skill.description.to_s.downcase
          sys_skill = skill_system_skill(name_str)
          triggers = sys_skill&.metadata&.triggers || []
          sys_desc = sys_skill&.description.to_s.downcase
          tool_names = skill.tools.map { |t| t[:name].to_s.downcase }

          searchable_text = [name_str, desc, sys_desc, triggers.join(" "), tool_names.join(" ")].join(" ")
          searchable_terms = extract_keywords(searchable_text)
          overlap = (query_words & searchable_terms)

          score = 0

          # Exact and near-exact name matches.
          score += 200 if name_str == query
          score += 100 if name_str.include?(query)
          score += query_words.count { |w| name_str.include?(w) } * 50
          score += 50 if query.include?(name_str) && name_str.length > 2

          # Textual overlap from SKILL metadata / description / tool names.
          score += overlap.size * 20
          score += 40 if desc.include?(query) || sys_desc.include?(query)

          # Explicit trigger phrase hit has strong signal.
          trigger_hits = triggers.count { |t| query.include?(t.to_s.downcase) }
          score += trigger_hits * 30

          matches << { name: name, skill: skill, score: score } if score > 0
        end

        # 按分数排序并返回前 N 个
        matches.sort_by { |m| -m[:score] }.first(limit)
      end

      def skill_system_skill(skill_name)
        return nil unless defined?(SmartBot::SkillSystem)
        return nil unless SmartBot::SkillSystem.respond_to?(:registry)

        registry = SmartBot::SkillSystem.registry
        return nil if registry.nil? || registry.empty?

        registry.find(skill_name)
      rescue
        nil
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
      def call_skill_by_name(skill_name, message, urls, llm_name, require_evidence: false)
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

          # 构建脚本参数：优先使用任务文本（如 "init" / "search xxx"），
          # 并去掉 run_skill 注入的 grounding 后缀；为空时回退到首个 URL。
          script_task = strip_grounding_suffix(message)
          args = script_task.empty? ? (urls.first || "") : script_task
          
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
            # 首先执行 SKILL.md 中的命令获取实际数据
            evidence_result = execute_skill_via_markdown(
              skill_name: skill_name,
              skill: skill,
              task: message,
              urls: urls,
              llm_name: llm_name
            )

            if evidence_result
              if require_evidence && !evidence_grounded_enough?(evidence_result)
                return "❌ run_skill 证据不足：未能从实际命令输出中提取到可验证视频信息（如 title/uploader/view_count）。已拒绝生成总结以避免幻觉。"
              end

              # 将执行结果作为上下文传递给 agent
              context = urls.any? ? "包含的URL: #{urls.join(', ')}\n\n" : ""
              context += "命令执行结果:\n#{evidence_result}"
              
              say "📊 已将执行结果传递给 skill agent 进行分析", :cyan
              
              result = tool.call({ 
                "task" => message,
                "context" => context
              })
            elsif require_evidence
              if allow_text_only_agent_without_evidence?(skill: skill, message: message, urls: urls)
                context = +""
                context << (urls.any? ? "包含的URL: #{urls.join(', ')}\n\n" : "")
                context << "任务类型: 文本改写/润色（闭环输入），无外部证据命令。\n"
                context << "约束: 仅基于用户提供文本改写，不得添加外部事实。"

                say "📝 该技能为闭环文本任务，跳过证据命令阶段", :cyan
                result = tool.call({
                  "task" => message,
                  "context" => context
                })
              else
                return "❌ 该技能仅提供说明型 `_agent`，且无法从 SKILL.md 生成可执行证据流程；为避免幻觉，run_skill 已拒绝本次调用。"
              end
            else
              # 调用 agent 工具（无证据模式）
              context = urls.any? ? "包含的URL: #{urls.join(', ')}" : ""
              result = tool.call({ 
                "task" => message,
                "context" => context
              })
            end
          else
            # 调用普通工具（参数根据工具定义动态构建）
            params = build_tool_call_params(
              tool: tool,
              skill_name: skill_name,
              message: message,
              urls: urls
            )

            say "🔍 执行: #{tool_name}", :cyan
            result = tool.call(params)
          end

          if result.is_a?(Hash)
            if result[:error]
              return "❌ 执行失败: #{result[:error]}"
            elsif result[:result]
              # Claude-style skill agent usually returns { result: "...", skill: "..." }
              return result[:result].to_s
            elsif result["result"]
              return result["result"].to_s
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

      def build_tool_call_params(tool:, skill_name:, message:, urls:)
        defined_params = tool.context&.params&.keys&.map(&:to_s) || []
        raw_text = strip_grounding_suffix(message.to_s)
        cleaned_text = raw_text.gsub(/用\s*#{Regexp.escape(skill_name.to_s)}/i, "").gsub(/#{Regexp.escape(skill_name.to_s)}/i, "").strip
        cleaned_text = urls.first.to_s if cleaned_text.empty? && urls.any?

        params = {}

        params["args"] = cleaned_text if defined_params.include?("args")
        params["task"] = cleaned_text if defined_params.include?("task")
        params["query"] = cleaned_text if defined_params.include?("query")
        params["count"] = 5 if defined_params.include?("count")

        if defined_params.include?("url")
          params["url"] = urls.first.to_s.empty? ? cleaned_text : urls.first.to_s
        end

        if defined_params.include?("location")
          params["location"] = cleaned_text
        end

        if defined_params.include?("days")
          days = raw_text[/\b(\d+)\b/, 1]&.to_i
          days ||= 1
          params["days"] = [days, 1].max
        end

        if params.empty?
          params["query"] = cleaned_text
          params["count"] = 5
        end

        params
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
          如果有匹配的技能，请只回复技能名称（必须来自上面的可用技能列表）。
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
          say "  /language [name]     - Show or set response language"
          say "  /skills              - List all available skills"
          say "  /find <keyword>      - Search skills by keyword"
          say "  /skill_help <name>   - Show detailed help for a skill"
          say "  /run_skill <skill> <task> - Delegate task to a specific skill"
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

        when "/language"
          if args.empty?
            say "Current language: #{set_color(current_system_language, :green)}"
            say "Usage: /language <name>"
            return
          end

          language_value = normalize_language(args.join(" "))
          unless valid_language?(language_value)
            say "❌ Invalid language. Use letters, numbers, spaces, '-' or '_'.", :red
            return
          end

          config["system_language"] = language_value
          save_smart_prompt_config(config)
          @system_language = language_value
          say "✓ Language updated: #{set_color(language_value, :green)}"
          
        when "/skills"
          render_skill_system_list

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

        when "/run_skill"
          if args.length < 2
            say "Usage: /run_skill <skill_name> <task>", :yellow
            say "Example: /run_skill invoice_organizer 整理 ./receipts 下的发票并输出CSV"
            return
          end

          skill_name = args.shift
          task = args.join(" ").strip
          output = execute_run_skill(skill_name: skill_name, task: task, llm_name: current_llm)
          say "\n#{output}\n"
          
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
