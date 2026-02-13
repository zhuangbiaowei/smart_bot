module BetterPrompt
  module Component
    class ShowPrompt
      def self.build(content, live)
        width = 60
        dialog = RubyRich::Dialog.new(title: "提示词内容", content: content, width: width, height: 20, buttons: [:ok])
        dialog.live = live
        register_event_listener(dialog)
        return dialog
      end

      def self.register_event_listener(dialog)
        dialog.key(:enter){|event, live|
          live.layout.hide_dialog
        }
      end
    end
  end
end