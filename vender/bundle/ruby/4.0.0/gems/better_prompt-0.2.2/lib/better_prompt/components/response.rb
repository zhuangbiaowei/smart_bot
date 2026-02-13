module BetterPrompt
  module Component
    class Response
      def self.build
        layout = RubyRich::Layout.new(name: "response",  ratio: 1)
        draw_view(layout)
        register_event_listener(layout)
        return layout
      end

      def self.draw_view(layout)
        @panel = RubyRich::Panel.new(
          "",
          title: "回答  (Ctrl+r/Ctrl+a/Ctrl+t)  评分： 1~5"
        )
        layout.update_content(@panel)
      end

      def self.clean
        @panel.content = ""
      end

      def self.update_content(content)
        if content.strip.empty?          
          @panel.content = "<Empty>"
        else
          @panel.content = ""
          @panel.content = content
        end
        @panel.home
      end

      def self.set_content(content)
        @content = content
      end

      def self.set_reason_content(content)
        @reason_content = content
      end

      def self.set_tool_content(content)
        @tool_content = content
      end

      def self.show_content(type, feedback)
        rating = if feedback
          feedback.rating
        else
          "1~5"
        end
        if type==:content
          @panel.title = "回答  (Ctrl+r/Ctrl+a/Ctrl+t)  评分： #{rating}"
          update_content(@content.to_s)
        end
        if type==:reason
          @panel.title = "推理  (Ctrl+r/Ctrl+a/Ctrl+t)  评分： #{rating}"
          update_content(@reason_content.to_s)
        end
        if type==:tool
          @panel.title = "工具调用  (Ctrl+r/Ctrl+a/Ctrl+t)  评分： #{rating}"
          update_content(@tool_content.to_s)
        end
      end

      def self.register_event_listener(layout)
        layout.key(:ctrl_r) { |event, live|  show_content(:reason, live.params[:feedback])}
        layout.key(:ctrl_a) { |event, live|  show_content(:content, live.params[:feedback])}
        layout.key(:ctrl_t) { |event, live|  show_content(:tool, live.params[:feedback])}
        layout.key(:page_up) { |event, live| @panel.page_up}
        layout.key(:page_down) { |event, live| @panel.page_down}
      end
    end
  end
end