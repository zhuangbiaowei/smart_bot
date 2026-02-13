# frozen_string_literal: true

module SmartBot
  module SkillSystem
    # Value object representing permission settings for a skill
    class PermissionSet
      attr_reader :filesystem, :network, :environment

      def initialize(filesystem: {}, network: {}, environment: {})
        # Normalize to symbol keys
        fs = normalize_hash(filesystem)
        net = normalize_hash(network)
        env = normalize_hash(environment)

        @filesystem = {
          read: fs[:read] || [],
          write: fs[:write] || []
        }
        @network = {
          outbound: net[:outbound] || false
        }
        @environment = {
          allow: env[:allow] || []
        }
      end

      def can_read?(path)
        return true if @filesystem[:read].empty?
        @filesystem[:read].any? { |allowed| path.start_with?(allowed) }
      end

      def can_write?(path)
        return false if @filesystem[:write].empty?
        @filesystem[:write].any? { |allowed| path.start_with?(allowed) }
      end

      def can_access_network?
        @network[:outbound]
      end

      def can_access_env?(var)
        @environment[:allow].include?(var)
      end

      def to_h
        {
          filesystem: @filesystem,
          network: @network,
          environment: @environment
        }
      end

      private

      def normalize_hash(hash)
        return {} unless hash.is_a?(Hash)
        hash.transform_keys(&:to_sym)
      end
    end

    # Value object representing execution policy
    class ExecutionPolicy
      attr_reader :sandbox, :approval, :timeout

      def initialize(sandbox: :process, approval: :ask, timeout: 120)
        @sandbox = sandbox  # :none, :process, :container, :microvm
        @approval = approval  # :auto, :ask, :manual
        @timeout = timeout
      end

      def auto?
        @approval == :auto
      end

      def ask?
        @approval == :ask
      end

      def manual?
        @approval == :manual
      end

      def to_h
        {
          sandbox: @sandbox,
          approval: @approval,
          timeout: @timeout
        }
      end
    end

    # Value object representing a prerequisite
    class Prerequisite
      attr_reader :type, :name, :options

      def initialize(type:, name:, options: {})
        @type = type  # :bin, :env, :file, :gem, :any_bins, :openclaw_config, :openclaw_os
        @name = name
        @options = options
      end

      def satisfied?
        case @type
        when :bin
          !`which #{@name} 2>/dev/null`.empty?
        when :any_bins
          # name is an array of bins, any one satisfied is ok
          return true unless @name.is_a?(Array)
          @name.any? { |bin| !`which #{bin} 2>/dev/null`.empty? }
        when :env
          ENV.key?(@name)
        when :file
          File.exist?(@name)
        when :gem
          Gem::Specification.find_all_by_name(@name).any?
        when :openclaw_config
          # Check SmartBot config for path like "browser.enabled"
          check_openclaw_config(@name)
        when :openclaw_os
          # name is an array of supported OS
          return true unless @name.is_a?(Array)
          current_os = case RUBY_PLATFORM
                       when /linux/ then "linux"
                       when /darwin/ then "darwin"
                       when /win32|win64|mingw/ then "win32"
                       else "unknown"
                       end
          @name.include?(current_os)
        else
          true
        end
      end

      def to_h
        {
          type: @type,
          name: @name,
          satisfied: satisfied?
        }
      end

      private

      def check_openclaw_config(path)
        # Parse path like "browser.enabled"
        return false unless defined?(SmartBot) && SmartBot.respond_to?(:config)
        
        config = SmartBot.config
        parts = path.to_s.split(".")
        
        current = config
        parts.each do |part|
          return false unless current.is_a?(Hash)
          current = current[part] || current[part.to_sym]
          return false if current.nil?
        end
        
        # Check if truthy
        ![false, nil, "", 0, "false", "off", "no"].include?(current)
      end
    end

    # Value object representing an entrypoint
    class Entrypoint
      attr_reader :name, :runtime, :command, :inputs, :outputs

      def initialize(name:, runtime:, command:, inputs: {}, outputs: {})
        @name = name
        @runtime = runtime
        @command = command
        @inputs = inputs
        @outputs = outputs
      end

      def to_h
        {
          name: @name,
          runtime: @runtime,
          command: @command,
          inputs: @inputs,
          outputs: @outputs
        }
      end
    end
  end
end
