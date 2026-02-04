# frozen_string_literal: true

module SmartBot
  module Providers
    class ToolCallRequest
      attr_reader :id, :name, :arguments

      def initialize(id:, name:, arguments:)
        @id = id
        @name = name
        @arguments = arguments
      end
    end

    class LLMResponse
      attr_reader :content, :tool_calls, :finish_reason, :usage

      def initialize(content:, tool_calls: [], finish_reason: "stop", usage: {})
        @content = content
        @tool_calls = tool_calls
        @finish_reason = finish_reason
        @usage = usage
      end

      def has_tool_calls?
        !@tool_calls.empty?
      end
    end

    class Base
      def initialize(api_key: nil, api_base: nil, default_model: nil)
        @api_key = api_key
        @api_base = api_base
        @default_model = default_model
      end

      def chat(messages:, tools: nil, model: nil, max_tokens: 4096, temperature: 0.7)
        raise NotImplementedError
      end

      def default_model
        @default_model
      end

      protected

      def tool_definitions(tools)
        return nil unless tools
        tools.map do |tool|
          {
            type: "function",
            function: {
              name: tool[:name],
              description: tool[:description],
              parameters: tool[:parameters]
            }
          }
        end
      end
    end
  end
end
