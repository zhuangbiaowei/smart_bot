# frozen_string_literal: true

module SmartBot
  # Skill 系统 - 类似于 OpenClaw 的插件机制
  # 每个 skill 是一个独立的模块，可以注册工具、命令、MCP 客户端等
  module Skill
    class << self
      def registry
        @registry ||= {}
      end

      # 注册一个 skill
      def register(name, &block)
        registry[name] = SkillDefinition.new(name, &block)
        # 静默注册，不再输出日志
      end

      # 查找 skill
      def find(name)
        registry[name]
      end

      # 列出所有已注册的 skills
      def list
        registry.keys
      end

      # 加载所有 skills
      # 支持原生 Ruby Skills 和 Markdown Skills (awesome-claude-skills 格式)
      def load_all(skills_dir = nil)
        skills_dir ||= File.expand_path("~/smart_ai/smart_bot/skills")
        
        # 使用 UnifiedSkillLoader 加载所有类型的 skills
        require_relative "skill_adapters" unless defined?(SmartBot::SkillAdapters)
        SmartBot::SkillAdapters::UnifiedSkillLoader.load_all(skills_dir)
      end

      # 激活所有已注册的 skills
      def activate_all!
        registry.each do |name, skill|
          begin
            skill.activate!
            # 静默激活，不再输出日志
          rescue
            # 静默跳过激活失败的 skill
          end
        end
      end
    end

    # Skill 定义类
    class SkillDefinition
      attr_reader :name, :description, :version, :author, :tools, :commands, :mcp_clients, :config

      def initialize(name, &block)
        @name = name
        @description = ""
        @version = "0.1.0"
        @author = ""
        @tools = []
        @commands = []
        @mcp_clients = []
        @config = {}
        @activation_block = nil
        
        instance_eval(&block) if block_given?
      end

      # DSL 方法
      def desc(text)
        @description = text
      end

      def ver(v)
        @version = v
      end

      def author_name(name)
        @author = name
      end

      # 注册工具
      def register_tool(tool_name, desc = nil, &block)
        @tools << { name: tool_name, desc: desc, block: block }
      end

      # 注册命令
      def register_command(cmd_name, desc = "", &block)
        @commands << { name: cmd_name, desc: desc, block: block }
      end

      # 注册 MCP 客户端
      def register_mcp(client_name, &block)
        @mcp_clients << { name: client_name, block: block }
      end

      # 配置
      def configure(&block)
        @config_block = block
      end

      # 激活回调
      def on_activate(&block)
        @activation_block = block
      end

      # 激活 skill
      def activate!
        # 确保 SmartAgent.logger 已设置
        SmartAgent.logger ||= SmartBot.logger
        
        # 注册工具
        @tools.each do |tool_def|
          # 工具将在 SmartAgent 中注册
          SmartAgent::Tool.define(tool_def[:name], &tool_def[:block])
        end

        # 注册 MCP 客户端
        @mcp_clients.each do |mcp_def|
          SmartAgent::MCPClient.define(mcp_def[:name], &mcp_def[:block])
        end

        # 执行配置
        @config_block&.call(@config)

        # 执行激活回调
        @activation_block&.call
      end

      def to_h
        {
          name: @name,
          description: @description,
          version: @version,
          author: @author,
          tools: @tools.map { |t| t[:name] },
          commands: @commands.map { |c| { name: c[:name], desc: c[:desc] } },
          mcp_clients: @mcp_clients.map { |m| m[:name] }
        }
      end
    end
  end
end
