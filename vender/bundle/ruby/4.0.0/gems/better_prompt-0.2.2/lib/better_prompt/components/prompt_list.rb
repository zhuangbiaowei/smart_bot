module BetterPrompt
  module Component
    class PromptList
      def self.build
        layout = RubyRich::Layout.new(name: "prompt_list", ratio: 4)
        draw_view(layout)
        register_event_listener(layout)
        return layout
      end

      def self.draw_view(layout)
        layout.update_content(RubyRich::Panel.new(
          "",
          title: "提示词列表  (Shift+2)",
        ))
      end

      def self.update_title(prompt)
        result = BetterPrompt.engine.call_worker(:short_title, { text: prompt[:prompt_content] })
        prompt[:prompt_title] = result.dig("choices", 0, "message", "content")
        prompt.save
        return prompt[:prompt_title]
      end

      def self.update_prompt_list(live, template_name, page_num = 1, pos = 0)
        @template_name = template_name
        @current_page = page_num
        prompt_list = live.find_panel("prompt_list")
        prompt_list.border_style = :red
        per_page = 9
        ds = BetterPrompt::ORM::Prompt.where(:prompt_template_name => template_name)
        total_count = ds.count
        @page_count = (total_count / per_page.to_f).ceil
        page_num = [page_num.to_i, 1].max
        offset = (page_num - 1) * per_page
        @prompt_list = ds
          .limit(per_page)
          .offset(offset)
          .all
        prompt_list.content = generate_content(pos)
        prompt_list.border_style = :green
      end

      def self.generate_content(pos)
        i = 0
        prompt_list_str = ""
        count = @prompt_list.size
        @prompt_list.each do |prompt|
          i += 1
          title = prompt.prompt_title
          if title == nil
            title = update_title(prompt)
          end
          title_width = title.chars.sum { |c| Unicode::DisplayWidth.of(c) }
          if title_width > 22
            new_title = ""
            title.chars.each do |c|
              if (new_title + c).chars.sum { |c| Unicode::DisplayWidth.of(c) } > 22
                new_title += ".."
                break
              else
                new_title += c
              end
            end
            title = new_title
          end
          if i > 0 && i <= count
            if i == pos
              prompt_list_str += RubyRich::AnsiCode.color(:blue) + "#{i} #{title}\n" + RubyRich::AnsiCode.reset
            else
              prompt_list_str += RubyRich::AnsiCode.color(:blue) + "#{i}" + RubyRich::AnsiCode.reset + " #{title}\n"
            end
          end
        end
        if i > count
          prompt_list_str += " \n" * (25 - prompt_list_str.split("\n").size)
        end
        if @current_page > 1
          prompt_list_str += " " * 11 + "←" + " " * 2
        else
          prompt_list_str += " " * 14
        end
        if @current_page < @page_count
          prompt_list_str += " " * 2 + "→"
        end
        return prompt_list_str
      end

      def self.register_event_listener(layout)
        layout.key(:string) do |event, live|
          prompt_list = live.find_panel("prompt_list")
          if prompt_list.border_style == :green
            if event[:value].to_i >= 1 && event[:value].to_i <= @prompt_list.count
              update_prompt_list(live, @template_name, @current_page, event[:value].to_i)
              prompt = @prompt_list[event[:value].to_i - 1]
              dialog = ShowPrompt.build(prompt.prompt_content, live)
              live.layout.show_dialog(dialog)
              Response.clean
              ModelCall.update_model_call_list(live, prompt.prompt_id)
            end
          end
        end
        layout.key(:right) do |event, live|
          prompt_list = live.find_panel("prompt_list")
          if prompt_list.border_style == :green
            if @current_page < @page_count
              update_prompt_list(live, @template_name, @current_page + 1)
            end
          end
        end
        layout.key(:left) do |event, live|
          prompt_list = live.find_panel("prompt_list")
          if prompt_list.border_style == :green
            if @current_page > 1
              update_prompt_list(live, @template_name, @current_page - 1)
            end
          end
        end
      end
    end
  end
end
