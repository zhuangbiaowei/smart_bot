# frozen_string_literal: true

require "net/http"
require "json"
require "uri"
require "cgi"

module SmartBot
  module Tools
    class WebSearchTool < Base
      USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"

      def initialize(api_key: nil, max_results: 5)
        @api_key = api_key || ENV["BRAVE_API_KEY"]
        @max_results = max_results
        
        super(
          name: :web_search,
          description: "Search the web. Returns titles, URLs, and snippets.",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string", description: "Search query" },
              count: { type: "integer", description: "Results (1-10)", minimum: 1, maximum: 10 }
            },
            required: ["query"]
          }
        )
      end

      def execute(query:, count: nil)
        return "Error: BRAVE_API_KEY not configured" unless @api_key

        n = [([count || @max_results, 1].max), 10].min
        
        uri = URI.parse("https://api.search.brave.com/res/v1/web/search")
        uri.query = URI.encode_www_form(q: query, count: n)
        
        request = Net::HTTP::Get.new(uri)
        request["Accept"] = "application/json"
        request["X-Subscription-Token"] = @api_key
        
        response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: true) do |http|
          http.request(request)
        end

        if response.is_a?(Net::HTTPSuccess)
          data = JSON.parse(response.body, symbolize_names: true)
          results = data.dig(:web, :results) || []
          
          return "No results for: #{query}" if results.empty?

          lines = ["Results for: #{query}\n"]
          results.first(n).each_with_index do |item, i|
            lines << "#{i + 1}. #{item[:title]}"
            lines << "   #{item[:url]}"
            lines << "   #{item[:description]}" if item[:description]
          end
          lines.join("\n")
        else
          "Error: #{response.code} - #{response.body}"
        end
      rescue => e
        "Error: #{e.message}"
      end
    end

    class WebFetchTool < Base
      USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"

      def initialize(max_chars: 50000)
        @max_chars = max_chars
        
        super(
          name: :web_fetch,
          description: "Fetch URL and extract readable content (HTML → markdown/text).",
          parameters: {
            type: "object",
            properties: {
              url: { type: "string", description: "URL to fetch" },
              extractMode: { type: "string", enum: ["markdown", "text"], default: "markdown" },
              maxChars: { type: "integer", minimum: 100 }
            },
            required: ["url"]
          }
        )
      end

      def execute(url:, extractMode: "markdown", maxChars: nil)
        max_chars = maxChars || @max_chars
        
        uri = URI.parse(url)
        request = Net::HTTP::Get.new(uri)
        request["User-Agent"] = USER_AGENT
        
        response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: uri.scheme == "https") do |http|
          http.request(request)
        end

        return JSON.generate(error: "HTTP #{response.code}", url: url) unless response.is_a?(Net::HTTPSuccess)

        content_type = response["Content-Type"].to_s
        body = response.body
        
        if content_type.include?("application/json")
          text = JSON.pretty_generate(JSON.parse(body))
          extractor = "json"
        elsif content_type.include?("text/html") || body[0..255].match?(/\A\s*(<!DOCTYPE|\u003chtml)/i)
          # Simple HTML to text extraction
          text = extract_text(body, extractMode)
          extractor = "html"
        else
          text = body
          extractor = "raw"
        end

        truncated = text.length > max_chars
        text = text[0...max_chars] if truncated

        JSON.generate(
          url: url,
          finalUrl: response.uri.to_s,
          status: response.code,
          extractor: extractor,
          truncated: truncated,
          length: text.length,
          text: text
        )
      rescue => e
        JSON.generate(error: e.message, url: url)
      end

      private

      def extract_text(html, mode)
        # Simple HTML tag stripping
        text = html.gsub(/<script.*?\/<script\u003e/mi, "")
                   .gsub(/<style.*?\/<style\u003e/mi, "")
                   .gsub(/<[^\u003e]+>/, " ")
                   .gsub(/\u0026\w+;/, " ")
                   .gsub(/\u0026#\d+;/, " ")
                   .squeeze(" ")
                   .strip
        
        # Normalize whitespace
        text.gsub(/[ \t]+/, " ").gsub(/\n{3,}/, "\n\n")
      end
    end
  end
end
