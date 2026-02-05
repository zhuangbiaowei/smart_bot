# frozen_string_literal: true

# SmartBot Agent - 主代理定义
SmartAgent.define :smart_bot do
  question = params[:text]
  
  # 第一次调用，可能触发工具
  result = call_worker(:smart_bot, params.merge(
    with_tools: true,
    with_history: true
  ))
  
  # 如果调用了工具，循环处理直到完成
  while result.call_tools
    call_tools(result)
    # 工具执行后，让 LLM 继续处理
    result = call_worker(:smart_bot, params.merge(
      with_tools: true,
      with_history: true
    ))
  end
  
  result.content
end

# 构建 SmartBot Agent - 工具在这里定义
SmartAgent.build_agent(
  :smart_bot,
  mcp_servers: [:all_in_one],
  tools: [:read_file, :write_file, :edit_file, :list_dir, :exec, :web_fetch],
)
