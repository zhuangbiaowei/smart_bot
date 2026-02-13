module BetterPrompt
  module Component
    class Sidebar
      def self.build
        layout = RubyRich::Layout.new(name: "sidebar", size: 30)
        register_event_listener(layout)
        return layout
      end
      
      def self.register_event_listener(layout)        
        layout.key(:ctrl_c) { |event, live| live.stop }        
      end
    end
  end
end