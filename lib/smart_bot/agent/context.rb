# frozen_string_literal: true

require "base64"
require "mime/types"

module SmartBot
  module Agent
    class Context
      BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"].freeze

      attr_reader :workspace, :memory, :skills

      def initialize(workspace)
        @workspace = Pathname.new(workspace)
        @memory = Memory.new(workspace)
        @skills = Skills.new(workspace)
      end

      def build_system_prompt(skill_names: nil)
        parts = []
        parts << identity_section
        
        bootstrap = load_bootstrap_files
        parts << bootstrap if bootstrap

        memory_context = @memory.get_memory_context
        parts << "# Memory\n\n#{memory_context}" if memory_context

        always_skills = @skills.get_always_skills
        if always_skills.any?
          always_content = @skills.load_skills_for_context(always_skills)
          parts << "# Active Skills\n\n#{always_content}" if always_content
        end

        skills_summary = @skills.build_skills_summary
        if skills_summary
          parts << "# Skills\n\nThe following skills extend your capabilities. " \
                     "To use a skill, read its SKILL.md file using the read_file tool.\n\n#{skills_summary}"
        end

        parts.join("\n\n---\n\n")
      end

      def build_messages(history:, current_message:, skill_names: nil, media: nil)
        messages = []
        messages << { role: "system", content: build_system_prompt(skill_names: skill_names) }
        messages.concat(history)
        messages << { role: "user", content: build_user_content(current_message, media) }
        messages
      end

      def add_tool_result(messages, tool_call_id, tool_name, result)
        messages << {
          role: "tool",
          tool_call_id: tool_call_id,
          name: tool_name,
          content: result.to_s
        }
        messages
      end

      def add_assistant_message(messages, content, tool_calls = nil)
        msg = { role: "assistant", content: content || "" }
        msg[:tool_calls] = tool_calls if tool_calls
        messages << msg
        messages
      end

      private

      def identity_section
        now = Time.now.strftime("%Y-%m-%d %H:%M (%A)")
        workspace_path = @workspace.expand_path.to_s

        "# SmartBot 🤖\n\n" \
        "You are SmartBot, a helpful AI assistant. You have access to tools that allow you to:\n" \
        "- Read, write, and edit files\n" \
        "- Execute shell commands\n" \
        "- Search the web and fetch web pages\n" \
        "- Send messages to users on chat channels\n" \
        "- Spawn subagents for complex background tasks\n\n" \
        "## Current Time\n#{now}\n\n" \
        "## Workspace\nYour workspace is at: #{workspace_path}\n" \
        "- Memory files: #{workspace_path}/memory/MEMORY.md\n" \
        "- Daily notes: #{workspace_path}/memory/YYYY-MM-DD.md\n" \
        "- Custom skills: #{workspace_path}/skills/{skill-name}/SKILL.md\n\n" \
        "IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.\n" \
        "Only use the 'message' tool when you need to send a message to a specific chat channel.\n\n" \
        "Always be helpful, accurate, and concise."
      end

      def load_bootstrap_files
        parts = []
        BOOTSTRAP_FILES.each do |filename|
          file_path = @workspace / filename
          if file_path.exist?
            content = file_path.read(encoding: "UTF-8")
            parts << "## #{filename}\n\n#{content}"
          end
        end
        parts.join("\n\n") unless parts.empty?
      end

      def build_user_content(text, media)
        return text unless media && !media.empty?

        # For simplicity, just return text without image processing
        # Full implementation would encode images to base64
        text
      end
    end

    class Memory
      def initialize(workspace)
        @workspace = Pathname.new(workspace)
        @memory_dir = @workspace / "memory"
        @memory_file = @memory_dir / "MEMORY.md"
      end

      def get_today_file
        @memory_dir / "#{Utils::Helpers.today_date}.md"
      end

      def read_today
        file = get_today_file
        file.exist? ? file.read(encoding: "UTF-8") : ""
      end

      def append_today(content)
        file = get_today_file
        if file.exist?
          existing = file.read(encoding: "UTF-8")
          content = existing + "\n" + content
        else
          content = "# #{Utils::Helpers.today_date}\n\n" + content
        end
        file.write(content, encoding: "UTF-8")
      end

      def read_long_term
        @memory_file.exist? ? @memory_file.read(encoding: "UTF-8") : ""
      end

      def write_long_term(content)
        @memory_file.write(content, encoding: "UTF-8")
      end

      def get_recent_memories(days: 7)
        memories = []
        today = Date.today
        
        days.times do |i|
          date = today - i
          date_str = date.strftime("%Y-%m-%d")
          file = @memory_dir / "#{date_str}.md"
          memories << file.read(encoding: "UTF-8") if file.exist?
        end
        
        memories.join("\n\n---\n\n")
      end

      def get_memory_context
        parts = []
        
        long_term = read_long_term
        parts << "## Long-term Memory\n" + long_term if long_term && !long_term.empty?
        
        today = read_today
        parts << "## Today's Notes\n" + today if today && !today.empty?
        
        parts.join("\n\n") unless parts.empty?
      end
    end

    class Skills
      def initialize(workspace, builtin_skills_dir: nil)
        @workspace = Pathname.new(workspace)
        @workspace_skills = @workspace / "skills"
        @builtin_skills = builtin_skills_dir
      end

      def list_skills(filter_unavailable: true)
        skills = []
        
        # Workspace skills
        if @workspace_skills.directory?
          @workspace_skills.children.select(&:directory?).each do |skill_dir|
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exist?
              skills << { name: skill_dir.basename.to_s, path: skill_file.to_s, source: "workspace" }
            end
          end
        end

        # Built-in skills
        if @builtin_skills && File.directory?(@builtin_skills)
          Dir.glob(File.join(@builtin_skills, "*")).select { |f| File.directory?(f) }.each do |skill_dir|
            name = File.basename(skill_dir)
            next if skills.any? { |s| s[:name] == name }
            
            skill_file = File.join(skill_dir, "SKILL.md")
            if File.exist?(skill_file)
              skills << { name: name, path: skill_file, source: "builtin" }
            end
          end
        end

        filter_unavailable ? skills.select { |s| check_requirements(s[:name]) } : skills
      end

      def load_skill(name)
        workspace_skill = @workspace_skills / name / "SKILL.md"
        return workspace_skill.read(encoding: "UTF-8") if workspace_skill.exist?

        if @builtin_skills
          builtin_skill = File.join(@builtin_skills, name, "SKILL.md")
          return File.read(builtin_skill, encoding: "UTF-8") if File.exist?(builtin_skill)
        end

        nil
      end

      def load_skills_for_context(skill_names)
        parts = skill_names.map do |name|
          content = load_skill(name)
          content ? "### Skill: #{name}\n\n#{strip_frontmatter(content)}" : nil
        end.compact

        parts.join("\n\n---\n\n") unless parts.empty?
      end

      def build_skills_summary
        all_skills = list_skills(filter_unavailable: false)
        return nil if all_skills.empty?

        lines = ["<skills>"]
        all_skills.each do |s|
          available = check_requirements(s[:name])
          desc = get_skill_description(s[:name])
          lines << "  <skill available=\"#{available}\">"
          lines << "    <name>#{escape_xml(s[:name])}</name>"
          lines << "    <description>#{escape_xml(desc)}</description>"
          lines << "    <location>#{s[:path]}</location>"
          lines << "  </skill>"
        end
        lines << "</skills>"
        lines.join("\n")
      end

      def get_always_skills
        list_skills.select { |s| always_skill?(s[:name]) }.map { |s| s[:name] }
      end

      private

      def check_requirements(name)
        # Simplified - would check bins and env vars from frontmatter
        true
      end

      def always_skill?(name)
        # Simplified - would check 'always' flag from frontmatter
        false
      end

      def get_skill_description(name)
        content = load_skill(name)
        return name unless content

        # Extract description from frontmatter or first line
        if content =~ /^---\n.*?description:\s*(.+?)\n.*?---/m
          $1.strip
        else
          name
        end
      end

      def strip_frontmatter(content)
        if content =~ /^---\n.*?\n---\n/m
          content[$~.end(0)..-1].strip
        else
          content
        end
      end

      def escape_xml(text)
        text.gsub("&", "&amp;").gsub("<", "&lt;").gsub(">", "&gt;")
      end
    end
  end
end
