# frozen_string_literal: true

module SmartBot
  module Tools
    class Registry
      def initialize
        @tools = {}
      end

      def register(tool)
        @tools[tool.name] = tool
        SmartBot.logger.debug "Registered tool: #{tool.name}"
      end

      def unregister(name)
        @tools.delete(name)
      end

      def get(name)
        @tools[name.to_sym]
      end

      def has?(name)
        @tools.key?(name.to_sym)
      end

      def get_definitions
        @tools.values.map(&:to_schema)
      end

      def execute(name, params = {})
        tool = @tools[name.to_sym]
        return "Error: Tool '#{name}' not found" unless tool

        begin
          tool.execute(**params.transform_keys(&:to_sym))
        rescue => e
          "Error executing #{name}: #{e.message}"
        end
      end

      def tool_names
        @tools.keys
      end

      def empty?
        @tools.empty?
      end

      def size
        @tools.size
      end
    end
  end
end
