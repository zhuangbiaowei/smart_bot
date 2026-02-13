module BetterPrompt
  module Component
    class Root
      def self.build
        layout = RubyRich::Layout.new
        register_event_listener(layout)
        return layout
      end
      def self.register_event_listener(layout)
        layout.key(:string) do |event, live|
          key = event[:value]
          if key=="!"
            template_list = live.find_panel("template_list")
            template_list.border_style = :green
            prompt_list = live.find_panel("prompt_list")
            prompt_list.border_style = :white
            model_call = live.find_panel("model_call")
            model_call.border_style = :white
          elsif key=="@"
            template_list = live.find_panel("template_list")
            template_list.border_style = :white
            prompt_list = live.find_panel("prompt_list")
            prompt_list.border_style = :green
            model_call = live.find_panel("model_call")
            model_call.border_style = :white
          elsif key=="#"
            template_list = live.find_panel("template_list")
            template_list.border_style = :white
            prompt_list = live.find_panel("prompt_list")
            prompt_list.border_style = :white
            model_call = live.find_panel("model_call")
            model_call.border_style = :green
          end
        end
      end
    end
  end
end