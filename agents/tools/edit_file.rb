# frozen_string_literal: true

# 文件编辑工具
SmartAgent::Tool.define :edit_file do
  desc "Edit a file by replacing old_text with new_text"
  param_define :path, "The file path to edit", :string
  param_define :old_text, "The exact text to find and replace", :string
  param_define :new_text, "The text to replace with", :string
  
  tool_proc do
    require "pathname"
    
    path = input_params["path"]
    old_text = input_params["old_text"]
    new_text = input_params["new_text"]
    
    file_path = Pathname.new(File.expand_path(path))
    
    unless file_path.exist?
      return { error: "File not found: #{path}" }
    end
    
    begin
      content = file_path.read(encoding: "UTF-8")
      
      unless content.include?(old_text)
        return { error: "old_text not found in file" }
      end
      
      count = content.scan(/#{Regexp.escape(old_text)}/).length
      if count > 1
        return { error: "old_text appears #{count} times. Please provide more context." }
      end
      
      new_content = content.sub(old_text, new_text)
      file_path.write(new_content, encoding: "UTF-8")
      
      { success: true, path: path }
    rescue Errno::EACCES
      { error: "Permission denied: #{path}" }
    rescue => e
      { error: "Error editing file: #{e.message}" }
    end
  end
end
