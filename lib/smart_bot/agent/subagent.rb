# frozen_string_literal: true

require "json"
require "securerandom"

module SmartBot
  module Agent
    class SubagentManager
      def initialize(provider:, workspace:, bus:, model: nil, brave_api_key: nil)
        @provider = provider
        @workspace = Pathname.new(workspace)
        @bus = bus
        @model = model || provider.default_model
        @brave_api_key = brave_api_key
        @running_tasks = {}
        @mutex = Mutex.new
      end

      def spawn(task:, label: nil, origin_channel: "cli", origin_chat_id: "direct")
        task_id = SecureRandom.hex(4)
        display_label = label || task[0...30] + (task.length > 30 ? "..." : "")

        origin = { channel: origin_channel, chat_id: origin_chat_id }

        thread = Thread.new do
          run_subagent(task_id, task, display_label, origin)
        end

        @mutex.synchronize do
          @running_tasks[task_id] = thread
        end

        thread.join(0.1) # Don't block

        SmartBot.logger.info "Spawned subagent [#{task_id}]: #{display_label}"
        "Subagent [#{display_label}] started (id: #{task_id}). I'll notify you when it completes."
      end

      def run_subagent(task_id, task, label, origin)
        SmartBot.logger.info "Subagent [#{task_id}] starting task: #{label}"

        begin
          tools = Tools::Registry.new
          tools.register(Tools::ReadFileTool.new)
          tools.register(Tools::WriteFileTool.new)
          tools.register(Tools::ListDirTool.new)
          tools.register(Tools::ShellTool.new(working_dir: @workspace.to_s))
          tools.register(Tools::WebSearchTool.new(api_key: @brave_api_key))
          tools.register(Tools::WebFetchTool.new)

          system_prompt = build_subagent_prompt(task)
          messages = [
            { role: "system", content: system_prompt },
            { role: "user", content: task }
          ]

          max_iterations = 15
          iteration = 0
          final_result = nil

          while iteration < max_iterations
            iteration += 1
            
            response = @provider.chat(
              messages: messages,
              tools: tools.get_definitions,
              model: @model
            )

            if response.has_tool_calls?
              tool_call_dicts = response.tool_calls.map do |tc|
                {
                  id: tc.id,
                  type: "function",
                  function: {
                    name: tc.name,
                    arguments: JSON.dump(tc.arguments)
                  }
                }
              end
              messages << { role: "assistant", content: response.content || "", tool_calls: tool_call_dicts }

              response.tool_calls.each do |tool_call|
                SmartBot.logger.debug "Subagent [#{task_id}] executing: #{tool_call.name}"
                result = tools.execute(tool_call.name, tool_call.arguments)
                messages << { role: "tool", tool_call_id: tool_call.id, name: tool_call.name, content: result }
              end
            else
              final_result = response.content
              break
            end
          end

          final_result ||= "Task completed but no final response was generated."
          
          SmartBot.logger.info "Subagent [#{task_id}] completed successfully"
          announce_result(task_id, label, task, final_result, origin, "ok")

        rescue => e
          error_msg = "Error: #{e.message}"
          SmartBot.logger.error "Subagent [#{task_id}] failed: #{e.message}"
          announce_result(task_id, label, task, error_msg, origin, "error")
        ensure
          @mutex.synchronize { @running_tasks.delete(task_id) }
        end
      end

      def announce_result(task_id, label, task, result, origin, status)
        status_text = status == "ok" ? "completed successfully" : "failed"
        
        announce_content = "[Subagent '#{label}' #{status_text}]\n\n" \
                          "Task: #{task}\n\n" \
                          "Result:\n#{result}\n\n" \
                          "Summarize this naturally for the user. Keep it brief (1-2 sentences). " \
                          "Do not mention technical details like 'subagent' or task IDs."

        msg = Bus::InboundMessage.new(
          channel: "system",
          sender_id: "subagent",
          chat_id: "#{origin[:channel]}:#{origin[:chat_id]}",
          content: announce_content
        )

        @bus.publish_inbound(msg)
        SmartBot.logger.debug "Subagent [#{task_id}] announced result to #{origin[:channel]}:#{origin[:chat_id]}"
      end

      def build_subagent_prompt(task)
        "# Subagent\n\n" \
        "You are a subagent spawned by the main agent to complete a specific task.\n\n" \
        "## Your Task\n#{task}\n\n" \
        "## Rules\n" \
        "1. Stay focused - complete only the assigned task, nothing else\n" \
        "2. Your final response will be reported back to the main agent\n" \
        "3. Do not initiate conversations or take on side tasks\n" \
        "4. Be concise but informative in your findings\n\n" \
        "## What You Can Do\n" \
        "- Read and write files in the workspace\n" \
        "- Execute shell commands\n" \
        "- Search the web and fetch web pages\n" \
        "- Complete the task thoroughly\n\n" \
        "## What You Cannot Do\n" \
        "- Send messages directly to users (no message tool available)\n" \
        "- Spawn other subagents\n" \
        "- Access the main agent's conversation history\n\n" \
        "## Workspace\n" \
        "Your workspace is at: #{@workspace}\n\n" \
        "When you have completed the task, provide a clear summary of your findings or actions."
      end

      def running_count
        @mutex.synchronize { @running_tasks.size }
      end
    end
  end
end
