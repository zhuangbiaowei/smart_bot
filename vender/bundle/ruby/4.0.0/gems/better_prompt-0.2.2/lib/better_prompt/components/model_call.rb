module BetterPrompt
  module Component
    class ModelCall
      def self.build
        @current_pos = 0
        layout = RubyRich::Layout.new(name: "model_call", size: 14)
        draw_view(layout)
        register_event_listener(layout)
        return layout
      end

      def self.draw_view(layout)
        layout.update_content(RubyRich::Panel.new(
          "",
          title: "模型调用 (Shift+3) 打分 (Ctrl+f)"
        ))
      end

      def self.update_model_call_list(live, prompt_id)
        model_call = live.find_panel("model_call")
        @call_list = ORM::ModelCall.where_all(Sequel.like(:prompt_list, "%,#{prompt_id}]"))
        @page_count = (@call_list.size / 9.0).ceil
        model_call.content = generate_content()
      end

      def self.generate_content(pos=0, page_num=1)
        i = 0
        @current_page = page_num
        start_pos = (page_num - 1) * 9
        end_pos = start_pos + 9
        call_list_str = ""
        @last_pos = 0
        @call_list.each do |call|
          i += 1
          model = ORM::Model[call.model_id]
          row_str = "id: #{call.call_id}"
          row_str = row_str + " "*(10-row_str.length) + "model:" + model.model_provider + "/" + model.model_name
          if call.prompt_list.length < 42
            row_str = row_str + " "*(60-row_str.length) + "prompts: "+ call.prompt_list
          else
            row_str = row_str + " "*(60-row_str.length) + "prompts: ..."+ call.prompt_list[-42..-1]
          end
          if i > start_pos && i <= end_pos
            if i-start_pos==pos
              call_list_str += RubyRich::AnsiCode.color(:blue)+ "#{i-start_pos} #{row_str}\n" + RubyRich::AnsiCode.reset
            else
              call_list_str += RubyRich::AnsiCode.color(:blue)+ "#{i-start_pos}" + RubyRich::AnsiCode.reset + " #{row_str}\n"
            end
            @last_pos += 1
          end
        end
        @max_pos = i
        if start_pos>0 or i > end_pos
          call_list_str += " \n" * (11 - call_list_str.split("\n").size)
        end
        if start_pos>0
          call_list_str += " " * 11 + "←" + " " * 2
        else
          call_list_str += " " * 14          
        end
        if i > end_pos
          call_list_str += " " * 2 + "→"
        end
        return call_list_str
      end

      def self.try_parse(string)
        return JSON.parse(string)
      rescue JSON::ParserError
        begin
          return JSON.parse(string.gsub("=>",":").gsub("nil","\"nil\""))
        rescue JSON::ParserError
          return false
        end
      end

      def self.show_call(model_call)
        model_call.content = generate_content(@current_pos, @current_page)
        call = @call_list[(@current_page - 1)*9+@current_pos-1]
        response = ORM::Response.where(call_id: call.call_id).first
        if response
          response_content = response.response_content
          if json = try_parse(response_content)
            Response.set_content(json.dig("choices", 0, "message", "content"))
            Response.set_reason_content(json.dig("choices", 0, "message", "reasoning_content"))
            Response.set_tool_content(json.dig("choices",0,"message", "tool_calls"))
          else
            Response.set_content(response_content)
            Response.set_reason_content("")
            Response.set_tool_content("")
          end
          response_id = response.response_id
          feedback = ORM::Feedback.where(response_id: response_id).first
        else
            Response.set_content("响应不存在")
            Response.set_reason_content("响应不存在")
            Response.set_tool_content("响应不存在")
        end
        Response.show_content(:content, feedback)
        return feedback, response_id
      end

      def self.register_event_listener(layout)        
        layout.key(:string) do |event, live|
          model_call = live.find_panel("model_call")
          if model_call.border_style == :green && @call_list
            if event[:value].to_i >= 1 && event[:value].to_i <= @last_pos
              @current_pos = event[:value].to_i
              live.params[:feedback], live.params[:response_id] = show_call(model_call)
            end
          end
        end
        layout.key(:right) do |event, live|
          model_call = live.find_panel("model_call")
          if model_call.border_style == :green && @call_list
            if @current_page<@page_count
              model_call.content = generate_content(0, @current_page+1)
              @current_pos = 0
            end
          end
        end
        layout.key(:left) do |event, live|
          model_call = live.find_panel("model_call")
          if model_call.border_style == :green && @call_list
            if @current_page>1              
              model_call.content = generate_content(0, @current_page-1)
              @current_pos = 0
            end
          end
        end
        layout.key(:up) do |event, live|
          model_call = live.find_panel("model_call")
          if model_call.border_style == :green && @call_list
            if @current_pos > 1
              @current_pos -= 1
              show_call(model_call)
            end
          end
        end
        layout.key(:down) do |event, live|
          model_call = live.find_panel("model_call")
          if model_call.border_style == :green && @call_list
            if @current_pos < @max_pos
              @current_pos += 1
              show_call(model_call)
            end
          end
        end
        layout.key(:ctrl_f) do |event, live|
          model_call = live.find_panel("model_call")
          if model_call.border_style == :green && @current_pos>0 && @call_list
            dialog = Feedback.build(live)
            live.layout.show_dialog(dialog)
          end
        end
      end
    end
  end
end