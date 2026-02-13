# frozen_string_literal: true

# DePHY Mesh API MCP Client - 集成搜索、网页抓取等多功能服务
SmartAgent::MCPClient.define :all_in_one do
  type :sse
  url "https://mesh-api.dephy.io/mcp/d766aab9-eefb-4c82-b132-959370a131d8/sse"
  
  # 可选：添加认证头（如果需要）
  # headers "Authorization" => "Bearer #{ENV['DEPHY_API_KEY']}"
end
