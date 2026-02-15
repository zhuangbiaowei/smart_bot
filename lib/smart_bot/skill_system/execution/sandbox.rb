# frozen_string_literal: true

require "timeout"
require "shellwords"
require "uri"
require_relative "openclaw_executor"

module SmartBot
  module SkillSystem
    # Sandbox for permission enforcement and safe execution
    class Sandbox
      def check_permissions(permissions, skill_type = nil)
        # For OpenClaw skills, be more permissive by default
        if skill_type == :openclaw_instruction
          return true if permissions.network[:outbound] && network_allowed?
          # Allow if basic permissions are set
          return true if permissions.filesystem[:read].any? || permissions.filesystem[:write].any?
        end

        # Check filesystem permissions
        permissions.filesystem[:read].each do |path|
          return false unless readable?(path)
        end

        permissions.filesystem[:write].each do |path|
          return false unless writable?(path)
        end

        # Check network permissions
        if permissions.network[:outbound]
          return false unless network_allowed?
        end

        # Check environment variables
        permissions.environment[:allow].each do |var|
          return false unless ENV.key?(var)
        end

        true
      end

      def execute(skill, parameters, _context)
        policy = skill.metadata.execution_policy

        case policy.sandbox
        when :none
          execute_unrestricted(skill, parameters)
        when :process
          execute_in_process(skill, parameters)
        when :container
          execute_in_container(skill, parameters)
        else
          execute_in_process(skill, parameters)
        end
      end

      private

      def execute_unrestricted(skill, parameters)
        invoke_skill(skill, parameters)
      end

      def execute_in_process(skill, parameters)
        policy = skill.metadata.execution_policy

        Timeout.timeout(policy.timeout) do
          invoke_skill(skill, parameters)
        end
      rescue Timeout::Error
        ExecutionResult.failure(
          skill: skill,
          error: "Execution timeout after #{policy.timeout}s"
        )
      end

      def execute_in_container(_skill, _parameters)
        ExecutionResult.failure(
          skill: nil,
          error: "Container sandbox not yet implemented"
        )
      end

      def invoke_skill(skill, parameters)
        case skill.type
        when :ruby_native
          invoke_ruby_skill(skill, parameters)
        when :instruction
          invoke_instruction_skill(skill, parameters)
        when :script
          invoke_script_skill(skill, parameters)
        when :openclaw_instruction
          invoke_openclaw_skill(skill, parameters)
        else
          ExecutionResult.failure(
            skill: skill,
            error: "Unknown skill type: #{skill.type}"
          )
        end
      end

      def invoke_ruby_skill(skill, parameters)
        definition = SmartBot::Skill.find(skill.name.to_sym)
        return ExecutionResult.failure(
          skill: skill,
          error: "Ruby skill not registered: #{skill.name}"
        ) unless definition

        tool_name = select_ruby_tool_name(skill, definition, parameters)
        return ExecutionResult.failure(
          skill: skill,
          error: "No executable tool found for Ruby skill: #{skill.name}"
        ) unless tool_name

        begin
          tool = SmartAgent::Tool.find_tool(tool_name.to_sym) || SmartAgent::Tool.find_tool(tool_name.to_s)
          return ExecutionResult.failure(
            skill: skill,
            error: "Tool not found: #{tool_name}"
          ) unless tool

          input = build_ruby_tool_input(skill, tool, parameters)
          result = tool.call(input)
          ExecutionResult.success(skill: skill, value: result)
        rescue => e
          ExecutionResult.failure(skill: skill, error: e.message)
        end
      end

      def select_ruby_tool_name(skill, definition, parameters)
        input = parameters.transform_keys(&:to_s)
        requested = input["tool"] || input["action"]
        candidates = []
        candidates << requested if requested && !requested.to_s.strip.empty?
        candidates << "#{skill.name}_agent"

        Array(definition.tools).each do |tool_def|
          tool_name = tool_def[:name] || tool_def["name"]
          candidates << tool_name.to_s unless tool_name.nil?
        end

        candidates.compact.uniq.find do |name|
          SmartAgent::Tool.find_tool(name.to_sym) || SmartAgent::Tool.find_tool(name.to_s)
        end
      end

      def build_ruby_tool_input(skill, tool, parameters)
        input = parameters.transform_keys(&:to_s)
        defined_keys = tool.context&.params&.keys&.map(&:to_s) || []
        return input if defined_keys.empty?

        task = input["task"].to_s
        query = input["query"].to_s
        text = [task, query].find { |v| !v.empty? }.to_s
        built = {}

        defined_keys.each do |key|
          if input.key?(key) && !input[key].to_s.empty?
            built[key] = input[key]
            next
          end

          case key
          when "task", "query", "args"
            built[key] = text unless text.empty?
          when "url"
            built[key] = extract_first_url(text)
          when "location"
            location = extract_location_candidate(text, skill.metadata.triggers)
            built[key] = location unless location.empty?
          when "days"
            days = text[/\b(\d+)\b/, 1]&.to_i
            built[key] = [days || 1, 1].max
          end
        end

        built.delete_if { |_k, v| v.nil? || v.to_s.empty? }
        built.empty? ? input : built
      end

      def extract_first_url(text)
        match = text.to_s.match(%r{https?://[^\s]+})
        match ? match[0] : nil
      end

      def extract_location_candidate(text, triggers)
        raw = text.to_s.dup
        return "" if raw.empty?

        Array(triggers).sort_by { |t| -t.to_s.length }.each do |trigger|
          t = trigger.to_s.strip
          next if t.empty?
          raw.gsub!(/#{Regexp.escape(t)}/i, " ")
        end

        cleaned = raw.gsub(/[^\p{Han}A-Za-z0-9\s]/, " ").gsub(/\s+/, " ").strip
        return "" if cleaned.empty?

        en_tokens = cleaned.split(/\s+/)
        en_stopwords = %w[today tomorrow now weather forecast whats what is are in for at how]
        place = en_tokens.find { |tok| tok.match?(/[A-Za-z]/) && !en_stopwords.include?(tok.downcase) }
        return place if place

        if (m = cleaned.match(/([\p{Han}]{2,})(?=今天|明天|后天|现在|的|$)/))
          return m[1]
        end

        cleaned
      end

      def invoke_instruction_skill(skill, parameters)
        skill.load_full_content

        system_prompt = skill.content
        user_prompt = parameters[:task] || parameters["task"]

        result = execute_via_llm(system_prompt, user_prompt)

        ExecutionResult.success(skill: skill, value: result)
      rescue => e
        ExecutionResult.failure(skill: skill, error: e.message)
      end

      def invoke_openclaw_skill(skill, parameters)
        executor = Execution::OpenClawExecutor.new
        executor.execute(skill, parameters, {})
      end

      def invoke_script_skill(skill, parameters)
        entrypoint = skill.entrypoint_for(parameters[:action] || "default")
        return ExecutionResult.failure(
          skill: skill,
          error: "No entrypoint found"
        ) unless entrypoint

        script_path = File.join(skill.source_path, entrypoint.command)
        return ExecutionResult.failure(
          skill: skill,
          error: "Script not found: #{script_path}"
        ) unless File.exist?(script_path)

        result = execute_script(script_path, parameters)

        if result[:success]
          ExecutionResult.success(skill: skill, value: result)
        else
          ExecutionResult.failure(skill: skill, error: result[:error])
        end
      end

      def execute_via_llm(system_prompt, user_prompt)
        config_path = File.expand_path("~/.smart_bot/smart_prompt.yml")
        return "Error: SmartPrompt config not found" unless File.exist?(config_path)

        config = YAML.load_file(config_path)
        engine = SmartPrompt::Engine.new(config_path)
        llm_name = config["default_llm"] || "deepseek"

        worker_name = :"skill_execution_#{Time.now.to_i}"

        SmartPrompt.define_worker worker_name do
          use llm_name
          sys_msg system_prompt
          prompt user_prompt
          send_msg
        end

        engine.call_worker(worker_name, {})
      rescue => e
        "Error executing via LLM: #{e.message}"
      end

      def execute_script(script_path, parameters)
        ext = File.extname(script_path).downcase

        command = case ext
                  when ".py" then "python3 #{script_path}"
                  when ".rb" then "ruby #{script_path}"
                  when ".sh" then "bash #{script_path}"
                  when ".js" then "node #{script_path}"
                  else script_path
                  end

        # Build arguments
        args = build_script_args(parameters)
        full_command = "#{command} #{args}"

        output = `#{full_command} 2>&1`
        exit_status = $?.exitstatus

        if exit_status == 0
          { success: true, output: output }
        else
          { success: false, error: output, exit_code: exit_status }
        end
      rescue => e
        { success: false, error: e.message }
      end

      def build_script_args(parameters)
        # Extract URL from task if present
        task = parameters[:task] || parameters["task"]
        if task
          url_match = task.match(%r{(https?://[^\s]+)})
          if url_match
            return Shellwords.escape(url_match[1])
          end
        end

        # Fallback to named parameters
        parameters.map { |k, v| "#{k}=#{Shellwords.escape(v.to_s)}" }.join(" ")
      end

      def readable?(path)
        File.readable?(File.expand_path(path))
      end

      def writable?(path)
        dir = File.expand_path(path)
        File.directory?(dir) ? File.writable?(dir) : File.writable?(File.dirname(dir))
      end

      def network_allowed?
        true
      end
    end
  end
end
