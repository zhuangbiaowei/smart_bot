# frozen_string_literal: true

# 文件读取工具
SmartAgent::Tool.define :read_file do
  desc "Read the contents of a file at the given path"
  param_define :path, "The file path to read", :string
  
  tool_proc do
    require "pathname"
    
    path = input_params["path"]
    file_path = Pathname.new(File.expand_path(path))
    
    unless file_path.exist?
      return { error: "File not found: #{path}" }
    end
    
    unless file_path.file?
      return { error: "Not a file: #{path}" }
    end
    
    begin
      content = file_path.read(encoding: "UTF-8")
      { content: content, size: content.bytesize }
    rescue Errno::EACCES
      { error: "Permission denied: #{path}" }
    rescue => e
      { error: "Error reading file: #{e.message}" }
    end
  end
end
