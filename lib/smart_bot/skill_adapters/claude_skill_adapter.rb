# frozen_string_literal: true

module SmartBot
  module SkillAdapters
    # 适配 awesome-claude-skills / ClawdHub 格式的 Skill
    # 将 Markdown + YAML 格式的 Skill 转换为 SmartBot 工具
    class ClaudeSkillAdapter
      class << self
        # 加载单个 Skill
        # @param skill_path [String] Skill 目录路径
        # @return [Boolean] 是否成功加载
        def load(skill_path)
          skill_md = File.join(skill_path, "SKILL.md")
          return false unless File.exist?(skill_md)

          content = File.read(skill_md)
          parser = SkillMdParser.new(content, skill_path)

          unless parser.valid?
            # 静默跳过无效的 SKILL.md
            return false
          end

          # Keep first-loaded skill when names collide (local workspace skills
          # are loaded before external dirs), and avoid silent override.
          existing = SmartBot::Skill.find(parser.name.to_sym) || SmartBot::Skill.find(parser.name)
          return false if existing

          # 注册为 SmartBot Skill
          SmartBot::Skill.register parser.name do
            desc parser.description
            ver parser.version
            author_name parser.author

            # 配置
            configure do |config|
              config[:skill_path] = skill_path
              config[:metadata] = parser.metadata
            end

            # 注册主 Agent 工具 - 使用 SKILL.md 作为系统提示词
            # 捕获变量到局部变量，避免块中的闭包问题
            skill_name = parser.name
            skill_description = parser.description
            skill_content = parser.content
            agent_desc = "#{skill_description} (Powered by #{skill_name} skill)"
            
            register_tool :"#{skill_name}_agent", agent_desc do
              desc agent_desc
              param_define :task, "Task or question for the skill", :string
              param_define :context, "Additional context", :string

              tool_proc do
                task = input_params["task"]
                context = input_params["context"] || ""

                # 构建系统提示词
                system_prompt = <<~PROMPT
                  You are an AI assistant specialized in: #{skill_description}

                  # Skill Instructions

                  #{skill_content}

                  # Guidelines

                  1. Follow the instructions above carefully
                  2. Use available tools when needed
                  3. Be thorough but concise in your responses
                  4. If you need to execute commands or scripts, mention them clearly
                PROMPT

                # 构建用户提示词
                user_prompt = "Task: #{task}"
                user_prompt += "\n\nContext: #{context}" unless context.empty?

                # 直接内联 LLM 调用
                config_path = File.expand_path("~/.smart_bot/smart_prompt.yml")
                if !File.exist?(config_path)
                  { result: "Error: SmartPrompt config not found", skill: skill_name }
                else
                  begin
                    config = YAML.load_file(config_path)
                    llm_name = config["default_llm"] || "deepseek"

                    engine = SmartPrompt::Engine.new(config_path)
                    worker_name = :"skill_#{skill_name}_#{Time.now.to_i}"

                    # 捕获 user_prompt 到局部变量，避免在 worker 块中引用外部变量
                    captured_prompt = user_prompt
                    SmartPrompt.define_worker worker_name do
                      use llm_name
                      sys_msg system_prompt
                      prompt captured_prompt
                      send_msg
                    end

                    llm_result = engine.call_worker(worker_name, {})

                    { result: llm_result, skill: skill_name }
                  rescue => e
                    SmartBot.logger&.error "Error in #{skill_name}_agent: #{e.message}"
                    { result: "Error: #{e.message}", skill: skill_name }
                  end
                end
              end
            end

            # 注册 scripts/ 中的脚本为工具
            scripts_dir = File.join(skill_path, "scripts")
            if Dir.exist?(scripts_dir)
              Dir.glob(File.join(scripts_dir, "*")).each do |script|
                next if File.directory?(script)

                script_name = File.basename(script, ".*")
                tool_name = :"#{parser.name}_#{script_name}"

                script_desc = "Execute #{File.basename(script)} from #{parser.name} skill"
                register_tool tool_name, script_desc do
                  desc script_desc
                  param_define :args, "Arguments to pass to the script", :string

                  tool_proc do
                    args = input_params["args"] || ""
                    ClaudeSkillAdapter.send(:execute_script, script, args)
                  end
                end
              end
            end

            # 激活回调 - 静默处理
            on_activate do
              # 不再输出激活日志
            end
          end

          true
        rescue => e
          # 静默处理加载失败的 skill
          false
        end

        # 批量加载目录下的所有 Skills
        # 支持两种目录结构：
        # 1. {skills_dir}/{skill_name}/SKILL.md (awesome-claude-skills 格式)
        # 2. {skills_dir}/{author}/{skill_name}/SKILL.md (ClawdHub/OpenClaw 格式)
        # @param skills_dir [String] Skills 根目录
        # @return [Array<String>] 成功加载的 skill 名称列表
        def load_all(skills_dir)
          return [] unless File.directory?(skills_dir)

          loaded = []
          # 使用 **/SKILL.md 递归查找所有 SKILL.md 文件
          Dir.glob(File.join(skills_dir, "**/SKILL.md")).each do |skill_md|
            skill_path = File.dirname(skill_md)
            skill_name = File.basename(skill_path)
            
            # 避免重复加载同名 skill
            next if loaded.include?(skill_name)
            
            if load(skill_path)
              loaded << skill_name
            end
          end

          loaded
        end

        # 获取加载的 skill 数量（用于显示）
        def loaded_count(skills_dir)
          return 0 unless File.directory?(skills_dir)
          Dir.glob(File.join(skills_dir, "**/SKILL.md")).count
        end

        private

        # 构建系统提示词
        def build_system_prompt(parser)
          <<~PROMPT
            You are an AI assistant specialized in: #{parser.description}

            # Skill Instructions

            #{parser.content}

            # Guidelines

            1. Follow the instructions above carefully
            2. Use available tools when needed
            3. Be thorough but concise in your responses
            4. If you need to execute commands or scripts, mention them clearly
          PROMPT
        end

        # 构建用户提示词
        def build_user_prompt(task, context)
          prompt = "Task: #{task}"
          prompt += "\n\nContext: #{context}" unless context.empty?
          prompt
        end

        # 调用 LLM 执行 Skill
        def call_skill_llm(system_prompt, user_prompt, skill_name)
          # 获取默认 LLM 配置
          config_path = File.expand_path("~/.smart_bot/smart_prompt.yml")
          return "Error: SmartPrompt config not found" unless File.exist?(config_path)

          config = YAML.load_file(config_path)
          llm_name = config["default_llm"] || "deepseek"

          # 创建临时 worker
          engine = SmartPrompt::Engine.new(config_path)
          worker_name = :"skill_#{skill_name}_#{Time.now.to_i}"

          SmartPrompt.define_worker worker_name do
            use llm_name
            sys_msg system_prompt
            prompt params[:user_prompt]
            send_msg
          end

          result = engine.call_worker(worker_name, { user_prompt: user_prompt })
          result
        rescue => e
          "Error executing skill: #{e.message}"
        end

        # 执行脚本
        def execute_script(script_path, args)
          ext = File.extname(script_path).downcase

          # 确保 deno 在 PATH 中（yt-dlp 需要）
          deno_path = File.expand_path("~/.deno/bin")
          env_path = "#{deno_path}:#{ENV['PATH']}"

          command = case ext
                    when ".py"
      "export PATH=\"#{env_path}\" && python3 #{script_path} #{args}"
                    when ".rb"
      "export PATH=\"#{env_path}\" && ruby #{script_path} #{args}"
                    when ".sh"
      "export PATH=\"#{env_path}\" && bash #{script_path} #{args}"
                    when ".js", ".mjs"
      "export PATH=\"#{env_path}\" && node #{script_path} #{args}"
                    else
      "export PATH=\"#{env_path}\" && #{script_path} #{args}"
                    end

          # 执行命令
          output = `#{command} 2>&1`
          exit_status = $?.exitstatus

          if exit_status == 0
            { success: true, output: output }
          else
            { success: false, error: output, exit_code: exit_status }
          end
        rescue => e
          { success: false, error: e.message }
        end
      end

      # SKILL.md 解析器
      class SkillMdParser
        attr_reader :name, :description, :version, :author, :metadata, :content, :skill_path

        def initialize(raw_content, skill_path = nil)
          @raw_content = raw_content
          @skill_path = skill_path
          @name = nil
          @description = nil
          @version = "0.1.0"
          @author = "Unknown"
          @metadata = {}
          @content = ""
          parse!
        end

        def valid?
          !@name.nil? && !@name.empty? && !@description.nil? && !@description.empty?
        end

        private

        def parse!
          # 解析 YAML frontmatter
          # 支持两种格式：
          # 1. ---\nname: xxx\ndescription: xxx\n---\ncontent
          # 2. ---\nname: xxx\n---\ndescription: xxx\ncontent (Anthropic 格式)

          if @raw_content =~ /\A---\s*\n(.+?)\n---\s*\n(.*)\z/m
            yaml_content = $1
            remaining = $2

            begin
              yaml = YAML.safe_load(yaml_content, permitted_classes: [Date, Time], aliases: true) || {}

              @name = normalize_name(yaml["name"])
              @description = yaml["description"]
              @version = yaml["version"] || yaml.dig("metadata", "version") || "0.1.0"
              @author = yaml["author"] || yaml.dig("metadata", "author") || "Unknown"
              @metadata = yaml["metadata"] || {}

              # 如果 description 在 YAML 里，剩余部分全是 content
              # 如果 description 不在 YAML 里，尝试从 remaining 提取
              if @description.nil? || @description.empty?
                # Anthropic 格式：description 在正文第一行
                lines = remaining.lines
                @description = lines.first&.strip&.gsub(/^#+\s*/, "")
                @content = lines[1..]&.join || ""
              else
                @content = remaining
              end
            rescue Psych::SyntaxError
              # YAML 解析错误，静默处理
              @content = @raw_content
            end
          else
            # 没有 YAML frontmatter，整个内容作为 content
            @content = @raw_content
          end
        end

        # 规范化 skill 名称
        def normalize_name(name)
          return nil if name.nil?

          # 转换为符号安全的格式
          name.downcase
              .gsub(/[^a-z0-9]+/, "_")  # 非字母数字转为下划线
              .gsub(/^_+|_+$/, "")      # 移除首尾下划线
              .gsub(/_+/, "_")          # 多个下划线合并
        end
      end
    end
  end
end
