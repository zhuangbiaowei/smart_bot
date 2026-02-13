require "ruby_rich"

module BetterPrompt
  class CLI
    class << self
      def start(db_path)
        @layout = Component::Root.build
        @layout.split_row(
          Component::Sidebar.build,
          Component::Main.build
        )
        @layout["sidebar"].split_column(
          Component::TemplateList.build,
          Component::PromptList.build,
        )
        @layout["main"].split_column(
          Component::ModelCall.build,
          Component::Response.build
        )
        ORM.setup("sqlite://"+db_path)
        RubyRich::Live.start(@layout, refresh_rate: 24) do |live|
          live.listening = true
        end
      end
    end
  end
end