# frozen_string_literal: true

require "json"
require "fileutils"
require "pathname"
require "yaml"

module SmartBot
  class Config
    DEFAULTS = {
      workspace: "~/.smart_bot/workspace",
      model: "deepseek-chat",
      max_tokens: 8192,
      temperature: 0.7,
      max_tool_iterations: 20
    }.freeze

    attr_accessor :workspace, :model, :max_tokens, :temperature, :max_tool_iterations

    def initialize(attrs = {})
      DEFAULTS.each do |key, value|
        instance_variable_set("@#{key}", attrs[key] || attrs[key.to_s] || value)
      end
    end

    def self.load(config_path = nil)
      # Try YAML config first (new format)
      yaml_path = config_path || File.expand_path("~/.smart_bot/smart_bot.yml")
      
      if File.exist?(yaml_path)
        begin
          content = File.read(yaml_path)
          # Replace environment variables
          content = content.gsub(/\\$\\{(\\w+)\\}/) { ENV[$1] || "" }
          data = YAML.load(content)
          
          return new(
            workspace: data["workspace"],
            model: data["default_llm"],
            max_tokens: data["max_tokens"],
            temperature: data["temperature"]
          )
        rescue => e
          SmartBot.logger.error "Failed to parse config: #{e.message}"
        end
      end
      
      # Fall back to JSON config (legacy)
      json_path = File.expand_path("~/.smart_bot/config.json")
      if File.exist?(json_path)
        begin
          data = JSON.parse(File.read(json_path), symbolize_names: true)
          return new(data)
        rescue JSON::ParserError => e
          SmartBot.logger.error "Failed to parse config: #{e.message}"
        end
      end
      
      new
    end

    def save(config_path = nil)
      path = config_path || self.class.default_config_path
      FileUtils.mkdir_p(File.dirname(path))
      File.write(path, JSON.pretty_generate(to_h))
    end

    def self.default_config_path
      File.expand_path("~/.smart_bot/config.json")
    end

    def self.data_dir
      path = File.expand_path("~/.smart_bot")
      FileUtils.mkdir_p(path)
      path
    end

    def workspace_path
      Pathname.new(File.expand_path(@workspace))
    end

    def to_h
      {
        workspace: @workspace,
        model: @model,
        max_tokens: @max_tokens,
        temperature: @temperature,
        max_tool_iterations: @max_tool_iterations
      }
    end
  end
end
