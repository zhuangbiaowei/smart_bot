# frozen_string_literal: true

require "yaml"
require "date"
require_relative "skill_package"
require_relative "metadata"
require_relative "registry"
require_relative "../adapters/openclaw_adapter"

module SmartBot
  module SkillSystem
    # Unified loader for all skill formats
    # Supports: Ruby native, Markdown (SKILL.md), Script-backed, OpenClaw
    class UnifiedLoader
      DISCOVERY_PATHS = [
        "%{workspace}/skills",
        "%{repo_root}/.agents/skills",
        "%{repo_root}/.claude/skills",
        "%{home}/.agents/skills",
        "%{home}/.claude/skills",
        "/etc/agent/skills"
      ].freeze

      BUILTIN_SKILLS_DIR = File.expand_path("~/smart_ai/smart_bot/skills")

      def initialize(workspace:, repo_root: nil, home: nil)
        @workspace = workspace
        @repo_root = repo_root || Dir.pwd
        @home = home || Dir.home
      end

      def load_all
        skills = []

        discovery_paths.each do |path|
          next unless File.directory?(path)

          skills.concat(load_from_directory(path))
        end

        # Load builtin skills
        skills.concat(load_from_directory(BUILTIN_SKILLS_DIR)) if File.directory?(BUILTIN_SKILLS_DIR)

        skills
      end

      def load_skill(name)
        discovery_paths.each do |path|
          skill_path = File.join(path, name)
          skill = load_from_path(skill_path)
          return skill if skill
        end

        # Try builtin
        builtin_path = File.join(BUILTIN_SKILLS_DIR, name)
        load_from_path(builtin_path)
      end

      def load_from_directory(dir)
        return [] unless File.directory?(dir)

        skills = []

        Dir.glob(File.join(dir, "*/")).each do |skill_dir|
          skill = load_from_path(skill_dir)
          skills << skill if skill
        end

        skills
      end

      def load_from_path(path)
        return nil unless File.directory?(path)

        # Try loading in order of specificity:
        # 1. Ruby native (skill.rb) - highest priority, full programmatic control
        # 2. Full Skill (skill.yaml + SKILL.md) - structured metadata + content
        # 3. OpenClaw format (SKILL.md with metadata.openclaw) - compatibility
        # 4. Simple Markdown (SKILL.md with frontmatter) - basic instruction skill
        load_ruby_skill(path) ||
          load_full_skill_with_yaml(path) ||
          load_openclaw_skill(path) ||
          load_simple_markdown_skill(path)
      end

      private

      def discovery_paths
        DISCOVERY_PATHS.map do |template|
          template % {
            workspace: @workspace,
            repo_root: @repo_root,
            home: @home
          }
        end
      end

      def load_ruby_skill(path)
        skill_rb = File.join(path, "skill.rb")
        return nil unless File.exist?(skill_rb)

        # Load the Ruby file to register the skill
        begin
          load skill_rb

          skill_name = File.basename(path)
          definition = SmartBot::Skill.find(skill_name.to_sym)

          return nil unless definition

          frontmatter = load_skill_frontmatter(path)
          merged_description = frontmatter&.dig("description") || definition.description
          merged_triggers = Array(frontmatter&.dig("triggers")).compact
          merged_anti_triggers = Array(frontmatter&.dig("anti_triggers")).compact

          if merged_triggers.empty? && frontmatter
            merged_triggers = infer_triggers_from_frontmatter(frontmatter, merged_description)
          end
          merged_triggers = infer_triggers(definition) if merged_triggers.empty?

          # Create metadata from definition
          metadata = SkillMetadata.new(
            name: skill_name,
            description: merged_description,
            version: frontmatter&.dig("version") || definition.version,
            author: frontmatter&.dig("author") || definition.author,
            triggers: merged_triggers,
            anti_triggers: merged_anti_triggers,
            cost_hint: frontmatter&.dig("cost_hint")&.to_sym,
            always: frontmatter&.dig("always"),
            parallel_safe: frontmatter&.dig("parallel_safe"),
            type: :ruby_native
          )

          SkillPackage.new(
            name: skill_name,
            source_path: path,
            metadata: metadata,
            type: :ruby_native,
            content: nil
          )
        rescue => e
          warn "Failed to load Ruby skill #{path}: #{e.message}"
          nil
        end
      end

      def load_full_skill_with_yaml(path)
        skill_md = File.join(path, "SKILL.md")
        skill_yaml = File.join(path, "skill.yaml")
        
        # Both files must exist for a full skill
        return nil unless File.exist?(skill_md) && File.exist?(skill_yaml)

        content = File.read(skill_md, encoding: "UTF-8")
        load_full_skill(path, skill_md, skill_yaml, content)
      end

      def load_openclaw_skill(path)
        skill_md = File.join(path, "SKILL.md")
        return nil unless File.exist?(skill_md)

        content = File.read(skill_md, encoding: "UTF-8")
        
        # Parse frontmatter; plain SKILL.md instruction skills are also treated
        # as OpenClaw-compatible skills for unified execution behavior.
        frontmatter, _remaining = parse_frontmatter(content)
        return nil unless frontmatter

        openclaw_frontmatter = Adapters::OpenClawAdapter.can_parse?(frontmatter)

        # Convert OpenClaw format (or plain markdown frontmatter) to a
        # unified OpenClaw instruction package.
        metadata = Adapters::OpenClawAdapter.convert(frontmatter, content)
        return nil unless metadata

        SkillPackage.new(
          name: metadata.name,
          source_path: path,
          metadata: metadata,
          type: :openclaw_instruction,
          content: content,
          original_format: openclaw_frontmatter ? :openclaw : :markdown
        )
      rescue => e
        logger = defined?(SmartBot) && SmartBot.respond_to?(:logger) ? SmartBot.logger : nil
        if logger
          logger.error "Failed to load OpenClaw skill #{path}: #{e.message}"
        else
          warn "Failed to load OpenClaw skill #{path}: #{e.message}"
          warn "  #{e.backtrace.first(3).join("\n  ")}"
        end
        nil
      end

      def load_skill_frontmatter(path)
        skill_md = File.join(path, "SKILL.md")
        return nil unless File.exist?(skill_md)

        content = File.read(skill_md, encoding: "UTF-8")
        frontmatter, _remaining = parse_frontmatter(content)
        frontmatter.is_a?(Hash) ? frontmatter : nil
      rescue
        nil
      end

      def load_simple_markdown_skill(path)
        skill_md = File.join(path, "SKILL.md")
        return nil unless File.exist?(skill_md)

        # Skip if skill.yaml exists (should be handled by load_full_skill_with_yaml)
        return nil if File.exist?(File.join(path, "skill.yaml"))

        content = File.read(skill_md, encoding: "UTF-8")

        # Parse frontmatter from SKILL.md
        frontmatter, remaining = parse_frontmatter(content)
        return nil unless frontmatter

        # Skip if this looks like OpenClaw format (should have been caught above)
        if Adapters::OpenClawAdapter.can_parse?(frontmatter)
          return nil
        end

        # Extract description from first line if not in frontmatter
        description = frontmatter["description"]
        if description.nil? && remaining
          lines = remaining.lines
          description = lines.first&.strip
        end

        # Auto-infer triggers if not provided
        unless frontmatter["triggers"]
          frontmatter["triggers"] = infer_triggers_from_frontmatter(frontmatter, description)
        end

        metadata = SkillMetadata.from_frontmatter(frontmatter, description)
        return nil unless metadata&.name

        SkillPackage.new(
          name: metadata.name,
          source_path: path,
          metadata: metadata,
          type: :instruction,
          content: content
        )
      end

      # Legacy method - kept for compatibility but now delegates to specific loaders
      def load_markdown_skill(path)
        load_full_skill_with_yaml(path) ||
          load_openclaw_skill(path) ||
          load_simple_markdown_skill(path)
      end

      def load_full_skill(path, _skill_md, skill_yaml, content)
        yaml_content = YAML.load_file(skill_yaml)
        return nil unless yaml_content.is_a?(Hash)

        metadata = SkillMetadata.from_skill_yaml(yaml_content)
        return nil unless metadata

        type = detect_type(yaml_content)

        SkillPackage.new(
          name: metadata.name,
          source_path: path,
          metadata: metadata,
          type: type,
          content: content
        )
      rescue Psych::SyntaxError => e
        SmartBot.logger&.error "Invalid YAML in #{skill_yaml}: #{e.message}"
        nil
      end

      def load_simple_skill(path, _skill_md, content)
        # Parse frontmatter from SKILL.md
        frontmatter, remaining = parse_frontmatter(content)
        return nil unless frontmatter

        # Extract description from first line if not in frontmatter
        description = frontmatter["description"]
        if description.nil? && remaining
          lines = remaining.lines
          description = lines.first&.strip
        end

        # Auto-infer triggers if not provided
        unless frontmatter["triggers"]
          frontmatter["triggers"] = infer_triggers_from_frontmatter(frontmatter, description)
        end

        metadata = SkillMetadata.from_frontmatter(frontmatter, description)
        return nil unless metadata&.name

        SkillPackage.new(
          name: metadata.name,
          source_path: path,
          metadata: metadata,
          type: :instruction,
          content: content
        )
      end

      def infer_triggers_from_frontmatter(frontmatter, description)
        triggers = []
        
        # Add skill name
        name = frontmatter["name"]
        triggers << name if name
        
        # Add name variations (split by - or _)
        if name
          triggers.concat(name.split(/[-_]/))
        end
        
        # Extract keywords from description
        desc = description || frontmatter["description"]
        if desc
          # Extract meaningful words (4+ characters)
          words = desc.downcase.scan(/\b[a-z0-9\u4e00-\u9fa5]{2,}\b/)
          words.each do |word|
            triggers << word if word.length >= 2
          end
          
          # Add first few words as compound triggers
          words.first(3).each_cons(2) do |pair|
            triggers << pair.join(" ")
          end
        end
        
        # Add tags as triggers
        if frontmatter["tags"]
          triggers.concat(frontmatter["tags"])
        end
        
        triggers.uniq.compact
      end

      def parse_frontmatter(content)
        return [nil, content] unless content.start_with?("---")

        if content =~ /\A---\s*\n(.+?)\n---\s*\n(.*)\z/m
          yaml_content = $1
          remaining = $2

          begin
            frontmatter = YAML.safe_load(yaml_content, permitted_classes: [Date, Time], aliases: true)
            [frontmatter, remaining]
          rescue Psych::SyntaxError
            [nil, content]
          end
        else
          [nil, content]
        end
      end

      def detect_type(yaml)
        spec = yaml["spec"] || {}
        type = spec["type"]

        case type
        when "script" then :script
        when "instruction" then :instruction
        else :instruction
        end
      end

      def infer_triggers(definition)
        triggers = [definition.name.to_s]

        # Add words from description
        if definition.description
          definition.description.downcase.scan(/[a-z0-9]+|[\u4e00-\u9fff]+/).each do |word|
            clean = word.gsub(/[^a-z0-9\u4e00-\u9fff]/, "")
            next if clean.empty?

            triggers << clean if clean.length >= 2
          end
        end

        triggers.uniq
      end
    end
  end
end
