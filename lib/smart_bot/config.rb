# frozen_string_literal: true

require "json"
require "fileutils"
require "pathname"

module SmartBot
  class Config
    DEFAULTS = {
      workspace: "~/.smart_bot/workspace",
      model: "anthropic/claude-opus-4-5",
      max_tokens: 8192,
      temperature: 0.7,
      max_tool_iterations: 20,
      gateway_host: "0.0.0.0",
      gateway_port: 18790,
      providers: {
        openrouter: { api_key: "", api_base: "https://openrouter.ai/api/v1" },
        anthropic: { api_key: "", api_base: nil },
        openai: { api_key: "", api_base: nil },
        gemini: { api_key: "", api_base: nil },
        groq: { api_key: "", api_base: nil }
      },
      channels: {
        telegram: { enabled: false, token: "", allow_from: [] },
        whatsapp: { enabled: false, allow_from: [] }
      },
      tools: {
        web_search: { api_key: "", max_results: 5 }
      }
    }.freeze

    attr_accessor :workspace, :model, :max_tokens, :temperature, :max_tool_iterations
    attr_accessor :gateway_host, :gateway_port
    attr_accessor :providers, :channels, :tools

    def initialize(attrs = {})
      DEFAULTS.each do |key, value|
        instance_variable_set("@#{key}", attrs[key] || attrs[key.to_s] || deep_copy(value))
      end
    end

    def self.load(config_path = nil)
      path = config_path || default_config_path
      
      if File.exist?(path)
        begin
          data = JSON.parse(File.read(path), symbolize_names: true)
          new(data)
        rescue JSON::ParserError => e
          SmartBot.logger.error "Failed to parse config: #{e.message}"
          new
        end
      else
        new
      end
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

    def api_key
      key = providers[:openrouter][:api_key]
      return key if key && !key.to_s.strip.empty?
      
      key = providers[:deepseek][:api_key]
      return key if key && !key.to_s.strip.empty?
      
      key = providers[:siliconflow][:api_key]
      return key if key && !key.to_s.strip.empty?
      
      key = providers[:aliyun][:api_key]
      return key if key && !key.to_s.strip.empty?
      
      key = providers[:kimi_coding][:api_key]
      return key if key && !key.to_s.strip.empty?
      
      key = providers[:anthropic][:api_key]
      return key if key && !key.to_s.strip.empty?
      
      key = providers[:openai][:api_key]
      return key if key && !key.to_s.strip.empty?
      
      key = providers[:gemini][:api_key]
      return key if key && !key.to_s.strip.empty?
      
      key = providers[:groq][:api_key]
      return key if key && !key.to_s.strip.empty?
      
      nil
    end

    def api_base
      key = providers[:openrouter][:api_key]
      return providers[:openrouter][:api_base] if key && !key.to_s.strip.empty?
      
      key = providers[:deepseek][:api_key]
      return providers[:deepseek][:api_base] if key && !key.to_s.strip.empty?
      
      key = providers[:siliconflow][:api_key]
      return providers[:siliconflow][:api_base] if key && !key.to_s.strip.empty?
      
      key = providers[:aliyun][:api_key]
      return providers[:aliyun][:api_base] if key && !key.to_s.strip.empty?
      
      key = providers[:kimi_coding][:api_key]
      return providers[:kimi_coding][:api_base] if key && !key.to_s.strip.empty?
      
      key = providers[:anthropic][:api_key]
      return providers[:anthropic][:api_base] if key && !key.to_s.strip.empty?
      
      key = providers[:openai][:api_key]
      return providers[:openai][:api_base] if key && !key.to_s.strip.empty?
      
      key = providers[:gemini][:api_key]
      return providers[:gemini][:api_base] if key && !key.to_s.strip.empty?
      
      key = providers[:groq][:api_key]
      return providers[:groq][:api_base] if key && !key.to_s.strip.empty?
      
      nil
    end

    def to_h
      {
        workspace: @workspace,
        model: @model,
        max_tokens: @max_tokens,
        temperature: @temperature,
        max_tool_iterations: @max_tool_iterations,
        gateway_host: @gateway_host,
        gateway_port: @gateway_port,
        providers: @providers,
        channels: @channels,
        tools: @tools
      }
    end

    private

    def deep_copy(obj)
      Marshal.load(Marshal.dump(obj))
    end
  end
end
