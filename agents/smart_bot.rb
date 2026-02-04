# frozen_string_literal: true

# SmartBot Agent - 主代理定义
SmartAgent.define :smart_bot do
  question = params[:text]
  
  # 使用 call_worker 处理对话，工具由 build_agent 时定义
  loop do
    result = call_worker(:smart_bot, params.merge(
      with_history: true
    ))
    
    if result.call_tools
      # 执行工具调用
      call_tools(result)
      params[:text] = "请根据工具执行结果继续回答"
    else
      # 没有工具调用，返回最终答案
      break result.content
    end
  end
end

# 构建 SmartBot Agent - 工具在这里定义
SmartAgent.build_agent(
  :smart_bot,
  tools: [:read_file, :write_file, :edit_file, :list_dir, :exec, :web_search, :web_fetch],
)
