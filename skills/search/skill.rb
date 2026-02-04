# frozen_string_literal: true

# Search Skill - 搜索功能整合
# 提供多种搜索方式：MCP、SerpAPI、Brave Search

SmartBot::Skill.register :search do
  desc "多源搜索功能 - MCP、SerpAPI、Brave Search"
  ver "1.0.0"
  author_name "SmartBot Team"

  # 配置不同搜索源的优先级
  configure do |config|
    config[:priority] = [:mcp, :serpapi, :brave]
    config[:default_count] = 5
  end

  # 注册 MCP 客户端（如果可用）
  register_mcp :dephy_search do
    type :sse
    url "https://mesh-api.dephy.io/mcp/d766aab9-eefb-4c82-b132-959370a131d8/sse"
  end

  # 注册搜索工具
  register_tool :smart_search do
    desc "Smart search with automatic fallback"
    param_define :query, "Search query", :string
    param_define :count, "Number of results", :integer
    
    tool_proc do
      query = input_params["query"]
      count = input_params["count"] || 5
      
      # 尝试 MCP 搜索
      mcp_server = find_mcp_server(:search)
      if mcp_server
        result = call_mcp_tool(mcp_server, "search", { "query" => query })
        return format_mcp_result(result, query) if result
      end
      
      # 回退到 SerpAPI
      if ENV["SERP_API_KEY"]
        result = call_native_tool(:serp_search, { "query" => query, "count" => count })
        return result if result && !result[:error]
      end
      
      # 回退到 Brave Search
      if ENV["BRAVE_API_KEY"]
        result = call_native_tool(:web_search, { "query" => query, "count" => count })
        return result if result && !result[:error]
      end
      
      { error: "No search service available" }
    end
  end

  # 注册网页抓取工具
  register_tool :smart_scrape do
    desc "Smart web scraping with MCP fallback"
    param_define :url, "URL to scrape", :string
    
    tool_proc do
      url = input_params["url"]
      
      # 尝试 MCP scrape
      mcp_server = find_mcp_server(:scrape)
      if mcp_server
        result = call_mcp_tool(mcp_server, "scrape", { "url" => url })
        return { content: result["content"] } if result && result["content"]
      end
      
      # 回退到本地 web_fetch
      call_native_tool(:web_fetch, { "url" => url, "extract_mode" => "markdown" })
    end
  end

  on_activate do
    SmartAgent.logger&.info "Search skill activated with MCP, SerpAPI, and Brave support!"
  end

  # 辅助方法
  def find_mcp_server(tool_name)
    server_name = SmartAgent::MCPClient.find_server_by_tool_name(tool_name)
    return nil unless server_name
    SmartAgent::MCPClient.servers.key?(server_name) ? server_name : nil
  rescue
    nil
  end

  def call_mcp_tool(server_name, tool_name, params)
    client = SmartAgent::MCPClient.new(server_name)
    client.call(tool_name, params)
  rescue
    nil
  end

  def call_native_tool(tool_name, params)
    tool = SmartAgent::Tool.find_tool(tool_name)
    tool&.call(params)
  end

  def format_mcp_result(result, query)
    return nil unless result.is_a?(Hash)
    
    content = result["content"]
    return nil unless content
    
    # 解析 JSON 格式的搜索结果
    if content.is_a?(Array) && content.first.is_a?(Hash) && content.first["type"] == "text"
      text = content.first["text"]
      results = JSON.parse(text) rescue []
      
      {
        query: query,
        results: results.first(5).map do |r|
          {
            title: r["title"],
            url: r["link"] || r["url"],
            description: r["snippet"] || ""
          }
        end,
        source: "mcp"
      }
    else
      { query: query, results: [], raw: content }
    end
  rescue
    nil
  end
end
