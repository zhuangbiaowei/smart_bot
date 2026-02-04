# frozen_string_literal: true

# 网页抓取工具
SmartAgent::Tool.define :web_fetch do
  desc "Fetch URL and extract readable content"
  param_define :url, "URL to fetch", :string
  param_define :extract_mode, "Extract mode: markdown or text", :string
  
  tool_proc do
    require "net/http"
    require "nokogiri"
    
    url = input_params["url"]
    extract_mode = input_params["extract_mode"] || "markdown"
    
    begin
      uri = URI(url)
      response = Net::HTTP.get_response(uri)
      
      unless response.is_a?(Net::HTTPSuccess)
        return { error: "Failed to fetch: #{response.code}" }
      end
      
      html = response.body
      doc = Nokogiri::HTML(html)
      
      # 移除脚本和样式
      doc.css('script, style, nav, footer, header, aside').remove
      
      # 提取标题和正文
      title = doc.at_css('title')&.text || ""
      
      content = if extract_mode == "markdown"
        # 简单转换为 markdown
        doc.css('h1, h2, h3, h4, h5, h6, p, li').map do |elem|
          text = elem.text.strip
          next if text.empty?
          
          case elem.name
          when 'h1' then "# #{text}"
          when 'h2' then "## #{text}"
          when 'h3' then "### #{text}"
          when 'li' then "- #{text}"
          else text
          end
        end.compact.join("\n\n")
      else
        doc.css('body').text.strip.gsub(/\s+/, ' ')
      end
      
      {
        url: url,
        title: title,
        content: content[0..8000],
        truncated: content.length > 8000
      }
    rescue => e
      { error: "Error fetching page: #{e.message}" }
    end
  end
end
