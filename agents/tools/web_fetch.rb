# frozen_string_literal: true

# 网页抓取工具
SmartAgent::Tool.define :web_fetch do
  desc "Fetch URL and extract readable content"
  param_define :url, "URL to fetch", :string
  param_define :extract_mode, "Extract mode: markdown or text", :string
  
  tool_proc do
    require "net/http"
    require "nokogiri"
    require "openssl"
    
    url = input_params["url"]
    extract_mode = input_params["extract_mode"] || "markdown"
    
    begin
      uri = URI(url)
      
      # 创建 HTTP 请求，支持重定向
      http = Net::HTTP.new(uri.host, uri.port)
      http.use_ssl = uri.scheme == 'https'
      http.verify_mode = OpenSSL::SSL::VERIFY_PEER
      http.open_timeout = 10
      http.read_timeout = 30
      
      request = Net::HTTP::Get.new(uri.request_uri)
      request["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
      request["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
      request["Accept-Language"] = "zh-CN,zh;q=0.9,en;q=0.8"
      request["Accept-Encoding"] = "gzip, deflate, br"
      
      response = http.request(request)
      
      # 处理重定向
      if response.is_a?(Net::HTTPRedirection)
        redirect_url = response['location']
        if redirect_url
          # 处理相对 URL
          redirect_url = URI.join(url, redirect_url).to_s
          
          # 递归跟随重定向（最多3次）
          result = follow_redirect(redirect_url, 1, extract_mode)
          return result
        end
      end
      
      unless response.is_a?(Net::HTTPSuccess)
        return { error: "Failed to fetch: #{response.code} #{response.message}" }
      end
      
      # 处理 gzip 压缩
      body = response.body
      if response['content-encoding'] == 'gzip'
        require 'zlib'
        body = Zlib::GzipReader.new(StringIO.new(body)).read
      end
      
      html = body
      doc = Nokogiri::HTML(html)
      
      # 移除脚本和样式
      doc.css('script, style, nav, footer, header, aside, iframe, noscript').remove
      
      # 提取标题（优先使用 og:title，这是微信公众号等平台的常用方式）
      title = doc.at_css("meta[property=\"og:title\"]")&.[]("content")&.strip
      title ||= doc.at_css("meta[name=\"og:title\"]")&.[]("content")&.strip
      title ||= doc.at_css('title')&.text&.strip
      title ||= ""
      
      # 微信公众号特殊处理
      if url.include?("mp.weixin.qq.com")
        content_div = doc.at_css('#js_content') || doc.at_css('.rich_media_content')
        if content_div
          content = extract_wechat_content(content_div, extract_mode)
        else
          content = extract_generic_content(doc, extract_mode)
        end
      else
        content = extract_generic_content(doc, extract_mode)
      end
      
      {
        url: url,
        title: title,
        content: content[0..10000],
        truncated: content.length > 10000,
        length: content.length
      }
    rescue => e
      { error: "Error fetching page: #{e.message}" }
    end
  end
  
  # 递归跟随重定向
  def follow_redirect(url, depth, extract_mode)
    return { error: "Too many redirects" } if depth > 3
    
    uri = URI(url)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = uri.scheme == 'https'
    http.open_timeout = 10
    http.read_timeout = 30
    
    request = Net::HTTP::Get.new(uri.request_uri)
    request["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    request["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    
    response = http.request(request)
    
    if response.is_a?(Net::HTTPRedirection) && response['location']
      redirect_url = URI.join(url, response['location']).to_s
      return follow_redirect(redirect_url, depth + 1, extract_mode)
    end
    
    unless response.is_a?(Net::HTTPSuccess)
      return { error: "Failed to fetch after redirect: #{response.code}" }
    end
    
    # 处理响应并提取内容
    body = response.body
    if response['content-encoding'] == 'gzip'
      require 'zlib'
      body = Zlib::GzipReader.new(StringIO.new(body)).read
    end
    
    html = body
    doc = Nokogiri::HTML(html)
    doc.css('script, style, nav, footer, header, aside, iframe, noscript').remove
    
    title = doc.at_css("meta[property=\"og:title\"]")&.[]("content")&.strip
    title ||= doc.at_css("meta[name=\"og:title\"]")&.[]("content")&.strip
    title ||= doc.at_css('title')&.text&.strip
    title ||= ""
    
    if url.include?("mp.weixin.qq.com")
      content_div = doc.at_css('#js_content') || doc.at_css('.rich_media_content')
      if content_div
        content = extract_wechat_content(content_div, extract_mode)
      else
        content = extract_generic_content(doc, extract_mode)
      end
    else
      content = extract_generic_content(doc, extract_mode)
    end
    
    {
      url: url,
      title: title,
      content: content[0..10000],
      truncated: content.length > 10000,
      length: content.length
    }
  end
  
  # 提取微信公众号内容
  def extract_wechat_content(content_div, extract_mode)
    if extract_mode == "markdown"
      paragraphs = []
      
      content_div.children.each do |node|
        case node.name
        when 'p'
          text = node.text.strip
          paragraphs << text unless text.empty?
        when 'section'
          # 处理 section 中的内容
          node.css('p, span').each do |child|
            text = child.text.strip
            paragraphs << text unless text.empty?
          end
        when 'img'
          src = node['data-src'] || node['src']
          paragraphs << "![Image](#{src})" if src
        when 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'
          level = node.name[1].to_i
          text = node.text.strip
          paragraphs << "#{'#' * level} #{text}" unless text.empty?
        end
      end
      
      paragraphs.join("\n\n")
    else
      content_div.text.strip.gsub(/\s+/, ' ')
    end
  end
  
  # 通用内容提取
  def extract_generic_content(doc, extract_mode)
    if extract_mode == "markdown"
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
  end
end
