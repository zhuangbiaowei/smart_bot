module BetterPrompt
  module Component
    class Feedback
      def self.build(live)
        dialog = RubyRich::Dialog.new(title: "加载对话", content: "请问这个回答的内容打分，1~5 之间。", width: 80, height: 20, buttons: [:cancel])
        dialog.live = live
        register_event_listener(dialog)
        return dialog
      end
      def self.register_event_listener(dialog)
        dialog.key(:escape){|event, live|
          live.layout.hide_dialog
        }
        dialog.key(:string){|event, live|
          key_value = event[:value].to_i
          if key_value >=1 && key_value <=5
            if feedback=live.params[:feedback]
              feedback.rating = key_value
              feedback.save
            else
              feedback = ORM::Feedback.new(response_id: live.params[:response_id], rating: key_value)
              feedback.save
              live.params[:feedback] = feedback
            end
            response = live.find_panel("response")
            response.title = "回答  (Ctrl+r/Ctrl+a/Ctrl+t)  评分： #{key_value}"
            live.layout.hide_dialog
          end
        }
      end
    end
  end
end