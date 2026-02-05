# frozen_string_literal: true

require_relative "claude_skill_adapter"

module SmartBot
  module SkillAdapters
    # 统一加载所有类型的 Skills
    class UnifiedSkillLoader
      class << self
        # 统一加载所有 Skills（原生 Ruby + 外部 Markdown）
        # @param skills_dir [String] Skills 根目录
        # @return [Array<String>] 成功加载的 skill 名称列表
        def load_all(skills_dir = nil)
          skills_dir ||= File.expand_path("~/smart_ai/smart_bot/skills")
          return [] unless File.directory?(skills_dir)

          loaded = []

          # 1. 加载原生 Ruby Skills (skill.rb)
          ruby_skills = load_ruby_skills(skills_dir)
          loaded.concat(ruby_skills)

          # 2. 加载外部 Markdown Skills (SKILL.md)
          # 排除已经有 skill.rb 的目录
          md_skills = load_markdown_skills(skills_dir, ruby_skills)
          loaded.concat(md_skills)

          # 3. 加载外部 Skills 目录（如 awesome-claude-skills）
          external_dirs = [
            File.expand_path("~/smart_ai/awesome-claude-skills"),
            File.expand_path("~/smart_ai/skills/skills") # ClawdHub skills
          ]

          external_dirs.each do |external_dir|
            if File.directory?(external_dir)
              external_skills = ClaudeSkillAdapter.load_all(external_dir)
              loaded.concat(external_skills)
            end
          end

          # 静默加载，不再输出总数日志
          loaded
        end

        # 激活所有已加载的 Skills
        def activate_all!
          SmartBot::Skill.activate_all!
        end

        private

        # 加载原生 Ruby Skills
        def load_ruby_skills(skills_dir)
          loaded = []
          skill_files = Dir.glob(File.join(skills_dir, "*/skill.rb")).sort

          skill_files.each do |file|
            begin
              load file
              skill_name = File.basename(File.dirname(file))
              loaded << skill_name
              # 静默加载，不再输出日志
            rescue
              # 静默跳过加载失败的 Ruby skill
            end
          end

          loaded
        end

        # 加载 Markdown Skills
        def load_markdown_skills(skills_dir, exclude_names)
          loaded = []

          Dir.glob(File.join(skills_dir, "*/SKILL.md")).each do |skill_md|
            skill_path = File.dirname(skill_md)
            skill_name = File.basename(skill_path)

            # 跳过已经有 skill.rb 的目录
            next if exclude_names.include?(skill_name)

            if ClaudeSkillAdapter.load(skill_path)
              loaded << skill_name
            end
          end

          loaded
        end
      end
    end
  end
end
