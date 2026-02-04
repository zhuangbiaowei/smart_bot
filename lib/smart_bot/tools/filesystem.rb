# frozen_string_literal: true

require "pathname"

module SmartBot
  module Tools
    class ReadFileTool < Base
      def initialize
        super(
          name: :read_file,
          description: "Read the contents of a file at the given path.",
          parameters: {
            type: "object",
            properties: {
              path: { type: "string", description: "The file path to read" }
            },
            required: ["path"]
          }
        )
      end

      def execute(path:)
        file_path = Pathname.new(File.expand_path(path))
        return "Error: File not found: #{path}" unless file_path.exist?
        return "Error: Not a file: #{path}" unless file_path.file?

        file_path.read(encoding: "UTF-8")
      rescue Errno::EACCES
        "Error: Permission denied: #{path}"
      rescue => e
        "Error reading file: #{e.message}"
      end
    end

    class WriteFileTool < Base
      def initialize
        super(
          name: :write_file,
          description: "Write content to a file at the given path. Creates parent directories if needed.",
          parameters: {
            type: "object",
            properties: {
              path: { type: "string", description: "The file path to write to" },
              content: { type: "string", description: "The content to write" }
            },
            required: ["path", "content"]
          }
        )
      end

      def execute(path:, content:)
        file_path = Pathname.new(File.expand_path(path))
        file_path.parent.mkpath
        file_path.write(content, encoding: "UTF-8")
        "Successfully wrote #{content.bytesize} bytes to #{path}"
      rescue Errno::EACCES
        "Error: Permission denied: #{path}"
      rescue => e
        "Error writing file: #{e.message}"
      end
    end

    class EditFileTool < Base
      def initialize
        super(
          name: :edit_file,
          description: "Edit a file by replacing old_text with new_text. The old_text must exist exactly in the file.",
          parameters: {
            type: "object",
            properties: {
              path: { type: "string", description: "The file path to edit" },
              old_text: { type: "string", description: "The exact text to find and replace" },
              new_text: { type: "string", description: "The text to replace with" }
            },
            required: ["path", "old_text", "new_text"]
          }
        )
      end

      def execute(path:, old_text:, new_text:)
        file_path = Pathname.new(File.expand_path(path))
        return "Error: File not found: #{path}" unless file_path.exist?

        content = file_path.read(encoding: "UTF-8")
        return "Error: old_text not found in file. Make sure it matches exactly." unless content.include?(old_text)

        count = content.scan(/#{Regexp.escape(old_text)}/).length
        return "Warning: old_text appears #{count} times. Please provide more context to make it unique." if count > 1

        new_content = content.sub(old_text, new_text)
        file_path.write(new_content, encoding: "UTF-8")
        "Successfully edited #{path}"
      rescue Errno::EACCES
        "Error: Permission denied: #{path}"
      rescue => e
        "Error editing file: #{e.message}"
      end
    end

    class ListDirTool < Base
      def initialize
        super(
          name: :list_dir,
          description: "List the contents of a directory.",
          parameters: {
            type: "object",
            properties: {
              path: { type: "string", description: "The directory path to list" }
            },
            required: ["path"]
          }
        )
      end

      def execute(path:)
        dir_path = Pathname.new(File.expand_path(path))
        return "Error: Directory not found: #{path}" unless dir_path.exist?
        return "Error: Not a directory: #{path}" unless dir_path.directory?

        items = dir_path.children.sort.map do |item|
          prefix = item.directory? ? "📁 " : "📄 "
          "#{prefix}#{item.basename}"
        end

        return "Directory #{path} is empty" if items.empty?
        items.join("\n")
      rescue Errno::EACCES
        "Error: Permission denied: #{path}"
      rescue => e
        "Error listing directory: #{e.message}"
      end
    end
  end
end
