# frozen_string_literal: true

# 文件写入工具
SmartAgent::Tool.define :write_file do
  desc "Write content to a file at the given path"
  param_define :path, "The file path to write to", :string
  param_define :content, "The content to write", :string
  
  tool_proc do
    require "pathname"
    require "fileutils"
    
    path = input_params["path"]
    content = input_params["content"]
    
    file_path = Pathname.new(File.expand_path(path))
    
    begin
      file_path.parent.mkpath
      file_path.write(content, encoding: "UTF-8")
      { success: true, bytes_written: content.bytesize, path: path }
    rescue Errno::EACCES
      { error: "Permission denied: #{path}" }
    rescue => e
      { error: "Error writing file: #{e.message}" }
    end
  end
end
