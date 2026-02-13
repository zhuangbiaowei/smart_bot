module BetterPrompt
  module Component
    class TemplateList
      def self.build
        layout = RubyRich::Layout.new(name: "template_list", size: 14)
        draw_view(layout)
        register_event_listener(layout)
        return layout
      end

      def self.get_list_content
        @template_list = ORM::Prompt
          .where(prompt_template_name: 'NULL')
          .invert
          .select(:prompt_template_name)
          .distinct
          .map(&:prompt_template_name)
        @page_count = (@template_list.size / 9.0).ceil
        generate_content
      end

      def self.generate_content(pos=0, page_num=1)
        i = 0
        @current_page = page_num
        start_pos = (page_num - 1) * 9
        end_pos = start_pos + 9
        template_list_str = ""
        @template_list.each do |template|
          i += 1
          if i > start_pos && i <= end_pos            
            if i-start_pos==pos
              template_list_str += RubyRich::AnsiCode.color(:blue)+ "#{i-start_pos} #{template}\n" + RubyRich::AnsiCode.reset
            else
              template_list_str += RubyRich::AnsiCode.color(:blue)+ "#{i-start_pos}" + RubyRich::AnsiCode.reset + " #{template}\n"
            end
          end          
        end
        if start_pos>0 or i > end_pos
          template_list_str += " \n" * (11 - template_list_str.split("\n").size)
        end
        if start_pos>0
          template_list_str += " " * 11 + "←" + " " * 2
        else
          template_list_str += " " * 14          
        end
        if i > end_pos
          template_list_str += " " * 2 + "→"
        end
        return template_list_str
      end

      def self.draw_view(layout)
        layout.update_content(RubyRich::Panel.new(
          get_list_content(),
          title: "模板列表 (Shift+1)",
          border_style: :green
        ))
      end

      def self.register_event_listener(layout)
        layout.key(:string) do |event, live|
          template_list = live.find_panel("template_list")
          if template_list.border_style == :green
            if event[:value].to_i >= 1 && event[:value].to_i <= @template_list.count
              template_list.content = generate_content(event[:value].to_i, @current_page)
              template_name = @template_list[(@current_page - 1)*9+event[:value].to_i-1]
              PromptList.update_prompt_list(live, template_name)
            end
          end
        end
        layout.key(:right) do |event, live|
          template_list = live.find_panel("template_list")
          if template_list.border_style == :green
            if @current_page<@page_count
              template_list.content = generate_content(0, @current_page+1)
            end
          end
        end
        layout.key(:left) do |event, live|
          template_list = live.find_panel("template_list")
          if template_list.border_style == :green
            if @current_page>1
              template_list.content = generate_content(0, @current_page-1)
            end
          end
        end
      end
    end
  end
end