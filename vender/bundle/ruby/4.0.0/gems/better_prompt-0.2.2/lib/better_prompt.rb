require "sqlite3"
require "fileutils"
require "time"
require "smart_prompt"

require_relative 'better_prompt/cli'
require_relative 'better_prompt/orm'
Dir["#{__dir__}/better_prompt/components/*.rb"].each { |f| require_relative f }

module BetterPrompt
  class << self
    def setup(db_path:)
      BetterPrompt.logger = Logger.new("./log/better.log")
      @db_path = db_path
      # Return true if database already exists
      if File.exist?(db_path)
        ORM.setup("sqlite://"+db_path)
        require_relative 'better_prompt/models'
        return true 
      end

      # Ensure directory exists
      FileUtils.mkdir_p(File.dirname(db_path))

      # Initialize database
      SQLite3::Database.new(db_path) do |db|
        # Execute each statement from init.sql
        File.read(File.join(__dir__, "../db/init.sql")).split(/;\s*\n/).each do |statement|
          db.execute(statement.strip) unless statement.strip.empty?
        end
      end
      ORM.setup("sqlite://"+db_path)
      require_relative 'better_prompt/models'
      true
    rescue => e
      raise "Failed to initialize database: #{e.message}"
    end

    def add_model(provider, model_name)
      model_version = if model_name.include?(':')
                        model_name.split(':', 2).last
                      else
                        'latest'
                      end
      model_name = model_name.split(":")[0]
      
      ORM::Model.find_or_create(
        model_provider: provider,
        model_name: model_name,
        model_version: model_version
      )
    rescue => e
      raise "Failed to add model: #{e.message}"
    end

    def add_prompt(template_name, role, prompt_content)
      prompt_length = prompt_content.length
      
      ORM::Prompt.find_or_create(
        prompt_template_name: template_name,
        role: role,
        prompt_content: prompt_content
      ) do |prompt|
        prompt.prompt_length = prompt_length
      end
    rescue => e
      raise "Failed to add prompt: #{e.message}"
    end

    def add_model_call(provider, model_name, messages, streaming=false, temperature=0.7, max_tokens=0, top_p=0.0, top_k=0, params={})
      model_version = if model_name.include?(':')
                        model_name.split(':', 2).last
                      else
                        'latest'
                      end
      model_base_name = model_name.split(":")[0]
      
      # 查找或创建model记录
      model = ORM::Model.first(
        model_provider: provider,
        model_name: model_base_name,
        model_version: model_version
      )
      raise "Model not found: #{provider}/#{model_name}" unless model

      # 从messages构建prompt_list
      prompt_list = messages.map do |msg|
        role = msg[:role] || msg["role"]
        content = msg[:content] || msg["content"]
        if role == "assistant" || role == "tool"
          add_prompt("NULL", role, content)
        end
        # 查找prompt_id
        prompt = ORM::Prompt.first(role: role, prompt_content: content)
        prompt&.prompt_id || 0
      end.to_json

      # 创建model_call记录
      model_call = ORM::ModelCall.create(
        prompt_list: prompt_list,
        model_id: model.model_id,
        is_streaming: streaming,
        temperature: temperature,
        max_tokens: max_tokens,
        top_p: top_p,
        top_k: top_k,
        additional_parameters: params.to_json
      )

      model_call.call_id
    rescue => e
      raise "Failed to add model call: #{e.message}"
    end

    def add_response(call_id, response_content, is_streaming)
      response_content = response_content.to_s
      response_length = response_content.length
      current_time = Time.now
      
      # 查找model_call记录
      model_call = ORM::ModelCall[call_id]
      raise "Model call not found: #{call_id}" unless model_call

      # 计算响应时间(毫秒)
      response_time_ms = ((current_time - model_call.call_timestamp) * 1000).to_i

      # 创建response记录
      response = ORM::Response.create(
        call_id: call_id,
        response_content: response_content,
        response_length: response_length,
        response_time_ms: response_time_ms,
        is_streaming: is_streaming
      )

      response.response_id
    rescue => e
      raise "Failed to add response: #{e.message}"
    end
    def engine
      if @engine == nil
        file_path = "./config/config.yml"
        file_path = "./config/llm_config.yml" unless File.exist?(file_path)
        @engine = SmartPrompt::Engine.new(file_path)
      end
      return @engine
    end
    def logger=(logger)
      @logger = logger
    end
    def logger
      @logger ||= Logger.new($stdout).tap do |log|
        log.progname = "Better Prompt"
      end
    end
  end
end