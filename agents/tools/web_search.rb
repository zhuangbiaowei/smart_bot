# frozen_string_literal: true

# 网页搜索工具
SmartAgent::Tool.define :web_search do
  desc "Search the web for information"
  param_define :query, "Search query", :string
  param_define :count, "Number of results (1-10)", :integer
  
  tool_proc do
    require "net/http"
    require "json"
    require "uri"
    
    query = input_params["query"]
    count = input_params["count"] || 5
    
    api_key = ENV["BRAVE_API_KEY"]
    if api_key.nil? || api_key.strip.empty?
      next { error: "Brave API key not configured" }
    end
    
    begin
      uri = URI("https://api.search.brave.com/res/v1/web/search")
      uri.query = URI.encode_www_form({ q: query, count: count })
      
      request = Net::HTTP::Get.new(uri)
      request["X-Subscription-Token"] = api_key
      request["Accept"] = "application/json"
      
      response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: true) do |http|
        http.request(request)
      end
      
      if response.is_a?(Net::HTTPSuccess)
        data = JSON.parse(response.body)
        results = (data["web"]&.[]("results") || []).map do |r|
          {
            title: r["title"],
            url: r["url"],
            description: r["description"]
          }
        end
        
        { query: query, results: results, count: results.length }
      else
        { error: "Search failed: #{response.code} - #{response.message}" }
      end
    rescue => e
      { error: "Error searching: #{e.message}" }
    end
  end
end
