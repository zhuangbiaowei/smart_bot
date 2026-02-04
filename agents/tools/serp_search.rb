# frozen_string_literal: true

# SerpAPI 搜索工具 - 支持 Google、Bing、Baidu 等多种搜索引擎
SmartAgent::Tool.define :serp_search do
  desc "Search using SerpAPI (supports Google, Bing, Baidu, etc.)"
  param_define :query, "Search query", :string
  param_define :engine, "Search engine (google, bing, baidu, yahoo, duckduckgo)", :string
  param_define :count, "Number of results (1-10)", :integer
  
  tool_proc do
    require "net/http"
    require "json"
    require "uri"
    
    query = input_params["query"]
    engine = input_params["engine"] || "google"
    count = input_params["count"] || 5
    
    api_key = ENV["SERP_API_KEY"]
    if api_key.nil? || api_key.strip.empty?
      next { error: "SerpAPI key not configured. Get one at https://serpapi.com/" }
    end
    
    begin
      # 构建 SerpAPI 请求
      params = {
        "q" => query,
        "engine" => engine,
        "api_key" => api_key,
        "num" => [count, 10].min
      }
      
      # 根据引擎添加特定参数
      case engine
      when "google"
        params["hl"] = "zh-CN"
        params["gl"] = "cn"
      when "baidu"
        # Baidu 特定参数
      when "bing"
        params["cc"] = "CN"
      end
      
      uri = URI("https://serpapi.com/search")
      uri.query = URI.encode_www_form(params)
      
      request = Net::HTTP::Get.new(uri)
      request["Accept"] = "application/json"
      
      response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: true) do |http|
        http.request(request)
      end
      
      unless response.is_a?(Net::HTTPSuccess)
        next { error: "SerpAPI request failed: #{response.code}" }
      end
      
      data = JSON.parse(response.body)
      
      # 检查错误
      if data["error"]
        next { error: "SerpAPI error: #{data["error"]}" }
      end
      
      # 提取搜索结果
      results = []
      
      # Google/Bing 有机搜索结果
      if organic = data["organic_results"]
        organic.first(count).each do |r|
          results << {
            title: r["title"],
            url: r["link"],
            description: r["snippet"] || r["description"],
            source: engine
          }
        end
      end
      
      # Baidu 结果
      if baidu_results = data["results"]
        baidu_results.first(count).each do |r|
          results << {
            title: r["title"],
            url: r["url"],
            description: r["abstract"],
            source: "baidu"
          }
        end
      end
      
      # 提取知识图谱（如果有）
      knowledge_graph = nil
      if kg = data["knowledge_graph"]
        knowledge_graph = {
          title: kg["title"],
          description: kg["description"],
          source: kg.dig("source", "name")
        }
      end
      
      # 提取相关问题
      related_questions = []
      if rq = data["related_questions"]
        related_questions = rq.first(3).map { |q| q["question"] }
      end
      
      {
        query: query,
        engine: engine,
        results: results,
        count: results.length,
        knowledge_graph: knowledge_graph,
        related_questions: related_questions,
        total_results: data.dig("search_information", "total_results")
      }
      
    rescue => e
      { error: "SerpAPI search error: #{e.message}" }
    end
  end
end
