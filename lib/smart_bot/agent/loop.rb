# frozen_string_literal: true

require "json"
require "pathname"

module SmartBot
  module Agent
    class Loop
      attr_reader :bus, :provider, :workspace, :model, :max_iterations

      def initialize(bus:, provider:, workspace:, model: nil, max_iterations: 20, brave_api_key: nil)
        @bus = bus
        @provider = provider
        @workspace = Pathname.new(workspace)
        @model = model || provider.default_model
        @max_iterations = max_iterations
        @brave_api_key = brave_api_key
        
        @context = Context.new(workspace)
        @sessions = Session::Manager.new(workspace)
        @tools = Tools::Registry.new
        @subagents = SubagentManager.new(
          provider: provider,
          workspace: workspace,
          bus: bus,
          model: @model,
          brave_api_key: brave_api_key
        )
        
        @running = false
        register_default_tools
      end

      def register_default_tools
        @tools.register(Tools::ReadFileTool.new)
        @tools.register(Tools::WriteFileTool.new)
        @tools.register(Tools::EditFileTool.new)
        @tools.register(Tools::ListDirTool.new)
        @tools.register(Tools::ShellTool.new(working_dir: @workspace.to_s))
        @tools.register(Tools::WebSearchTool.new(api_key: @brave_api_key))
        @tools.register(Tools::WebFetchTool.new)
        
        message_tool = Tools::MessageTool.new(send_callback: ->(msg) { @bus.publish_outbound(msg) })
        @tools.register(message_tool)
        
        spawn_tool = Tools::SpawnTool.new(@subagents)
        @tools.register(spawn_tool)
      end

      def run
        @running = true
        SmartBot.logger.info "Agent loop started"

        while @running
          msg = @bus.consume_inbound(timeout: 1)
          next unless msg

          begin
            response = process_message(msg)
            @bus.publish_outbound(response) if response
          rescue => e
            SmartBot.logger.error "Error processing message: #{e.message}"
            SmartBot.logger.debug e.backtrace.join("\n")
            
            @bus.publish_outbound(Bus::OutboundMessage.new(
              channel: msg.channel,
              chat_id: msg.chat_id,
              content: "Sorry, I encountered an error: #{e.message}"
            ))
          end
        end
      end

      def stop
        @running = false
        SmartBot.logger.info "Agent loop stopping"
      end

      def process_direct(content, session_key: "cli:direct")
        msg = Bus::InboundMessage.new(
          channel: "cli",
          sender_id: "user",
          chat_id: "direct",
          content: content
        )
        response = process_message(msg)
        response ? response.content : ""
      end

      def process_message(msg)
        return process_system_message(msg) if msg.channel == "system"

        SmartBot.logger.info "Processing message from #{msg.channel}:#{msg.sender_id}"

        session = @sessions.get_or_create(msg.session_key)
        
        # Update tool contexts
        message_tool = @tools.get(:message)
        message_tool.set_context(msg.channel, msg.chat_id) if message_tool
        
        spawn_tool = @tools.get(:spawn)
        spawn_tool.set_context(msg.channel, msg.chat_id) if spawn_tool

        messages = @context.build_messages(
          history: session.get_history,
          current_message: msg.content,
          media: msg.media
        )

        iteration = 0
        final_content = nil

        while iteration < @max_iterations
          iteration += 1
          
          tool_defs = @tools.get_definitions
          
          response = @provider.chat(
            messages: messages,
            tools: tool_defs,
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
            messages = @context.add_assistant_message(messages, response.content, tool_call_dicts)

            response.tool_calls.each do |tool_call|
              SmartBot.logger.debug "Executing tool: #{tool_call.name}"
              result = @tools.execute(tool_call.name, tool_call.arguments)
              messages = @context.add_tool_result(messages, tool_call.id, tool_call.name, result)
            end
          else
            final_content = response.content
            break
          end
        end

        final_content ||= "I've completed processing but have no response to give."

        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        @sessions.save(session)

        Bus::OutboundMessage.new(
          channel: msg.channel,
          chat_id: msg.chat_id,
          content: final_content
        )
      end

      def process_system_message(msg)
        SmartBot.logger.info "Processing system message from #{msg.sender_id}"

        # Parse origin from chat_id (format: "channel:chat_id")
        if msg.chat_id.include?(":")
          origin_channel, origin_chat_id = msg.chat_id.split(":", 2)
        else
          origin_channel = "cli"
          origin_chat_id = msg.chat_id
        end

        session_key = "#{origin_channel}:#{origin_chat_id}"
        session = @sessions.get_or_create(session_key)

        # Update tool contexts
        message_tool = @tools.get(:message)
        message_tool.set_context(origin_channel, origin_chat_id) if message_tool
        
        spawn_tool = @tools.get(:spawn)
        spawn_tool.set_context(origin_channel, origin_chat_id) if spawn_tool

        messages = @context.build_messages(
          history: session.get_history,
          current_message: msg.content
        )

        iteration = 0
        final_content = nil

        while iteration < @max_iterations
          iteration += 1
          
          response = @provider.chat(
            messages: messages,
            tools: @tools.get_definitions,
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
            messages = @context.add_assistant_message(messages, response.content, tool_call_dicts)

            response.tool_calls.each do |tool_call|
              result = @tools.execute(tool_call.name, tool_call.arguments)
              messages = @context.add_tool_result(messages, tool_call.id, tool_call.name, result)
            end
          else
            final_content = response.content
            break
          end
        end

        final_content ||= "Background task completed."

        session.add_message("user", "[System: #{msg.sender_id}] #{msg.content}")
        session.add_message("assistant", final_content)
        @sessions.save(session)

        Bus::OutboundMessage.new(
          channel: origin_channel,
          chat_id: origin_chat_id,
          content: final_content
        )
      end
    end
  end
end
