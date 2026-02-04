# frozen_string_literal: true

# 目录列表工具
SmartAgent::Tool.define :list_dir do
  desc "List the contents of a directory"
  param_define :path, "The directory path to list", :string
  
  tool_proc do
    require "pathname"
    
    path = input_params["path"]
    dir_path = Pathname.new(File.expand_path(path))
    
    unless dir_path.exist?
      return { error: "Directory not found: #{path}" }
    end
    
    unless dir_path.directory?
      return { error: "Not a directory: #{path}" }
    end
    
    begin
      items = dir_path.children.sort.map do |item|
        {
          name: item.basename.to_s,
          type: item.directory? ? "directory" : "file",
          size: item.file? ? item.size : nil
        }
      end
      
      { path: path, items: items, count: items.length }
    rescue Errno::EACCES
      { error: "Permission denied: #{path}" }
    rescue => e
      { error: "Error listing directory: #{e.message}" }
    end
  end
end
