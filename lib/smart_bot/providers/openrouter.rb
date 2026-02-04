# frozen_string_literal: true

require "net/http"
require "json"
require "uri"

module SmartBot
  module Providers
    class OpenRouterProvider < Base
      attr_reader :api_key, :api_base

      def initialize(api_key:, api_base: "https://openrouter.ai/api/v1", default_model: nil)
        super(api_key: api_key, api_base: api_base, default_model: default_model)
        @api_key = api_key
        @api_base = api_base
        @default_model = default_model || "anthropic/claude-opus-4-5"
      end

      def chat(messages:, tools: nil, model: nil, max_tokens: 4096, temperature: 0.7)
        model ||= @default_model
        
        uri = URI.parse("#{@api_base}/chat/completions")
        
        body = {
          model: model,
          messages: messages,
          max_tokens: max_tokens,
          temperature: temperature
        }
        
        if tools && !tools.empty?
          body[:tools] = tools
          body[:tool_choice] = "auto"
        end

        request = Net::HTTP::Post.new(uri)
        request["Authorization"] = "Bearer #{@api_key}"
        request["Content-Type"] = "application/json"
        # OpenRouter-specific headers - only add for OpenRouter
        if @api_base.include?("openrouter")
          request["HTTP-Referer"] = "https://smartbot.local"
          request["X-Title"] = "SmartBot"
        end
        request.body = body.to_json

        # Debug: log the request body (full for debugging)
        File.write("/tmp/last_request.json", request.body)
        SmartBot.logger.debug "LLM Request saved to /tmp/last_request.json"

        response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: uri.scheme == "https") do |http|
          http.request(request)
        end

        if response.is_a?(Net::HTTPSuccess)
          parse_response(JSON.parse(response.body, symbolize_names: true))
        else
          LLMResponse.new(
            content: "Error calling LLM: #{response.code} - #{response.body}",
            finish_reason: "error"
          )
        end
      rescue => e
        LLMResponse.new(
          content: "Error calling LLM: #{e.message}",
          finish_reason: "error"
        )
      end

      private

      def parse_response(data)
        choice = data[:choices] && data[:choices][0]
        message = choice[:message] if choice
        
        tool_calls = []
        if message && message[:tool_calls]
          tool_calls = message[:tool_calls].map do |tc|
            args = tc[:function][:arguments]
            args = JSON.parse(args) if args.is_a?(String)
            ToolCallRequest.new(
              id: tc[:id],
              name: tc[:function][:name].to_sym,
              arguments: args
            )
          end
        end

        usage = data[:usage] ? {
          prompt_tokens: data[:usage][:prompt_tokens],
          completion_tokens: data[:usage][:completion_tokens],
          total_tokens: data[:usage][:total_tokens]
        } : {}

        LLMResponse.new(
          content: message[:content],
          tool_calls: tool_calls,
          finish_reason: choice[:finish_reason] || "stop",
          usage: usage
        )
      end
    end
  end
end
