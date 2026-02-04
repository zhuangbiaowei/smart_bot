# frozen_string_literal: true

module SmartBot
  module Tools
    class Base
      attr_reader :name, :description, :parameters

      def initialize(name:, description:, parameters:)
        @name = name
        @description = description
        @parameters = parameters
      end

      def execute(**kwargs)
        raise NotImplementedError
      end

      def to_schema
        {
          type: "function",
          function: {
            name: @name.to_s,
            description: @description,
            parameters: @parameters
          }
        }
      end

      def to_json(*args)
        to_schema.to_json(*args)
      end
    end
  end
end
