# frozen_string_literal: true

module SmartBot
  module SkillSystem
    # Unified representation of a skill package
    # Supports multiple formats: Ruby native, Markdown, Script-backed
    class SkillPackage
      SKILL_TYPES = %i[instruction script ruby_native openclaw_instruction].freeze

      attr_reader :name, :source_path, :metadata, :type, :original_format

      def initialize(name:, source_path:, metadata:, type:, content: nil, original_format: nil)
        @name = normalize_name(name)
        @source_path = source_path
        @metadata = metadata
        @type = validate_type(type)
        @content = content
        @original_format = original_format
        @full_content_loaded = !content.nil?
      end

      def content
        load_full_content unless @full_content_loaded
        @content
      end

      def load_full_content
        return @content if @full_content_loaded

        skill_md = File.join(@source_path, "SKILL.md")
        if File.exist?(skill_md)
          @content = File.read(skill_md, encoding: "UTF-8")
          @full_content_loaded = true
        end

        @content
      end

      def scripts
        @scripts ||= discover_scripts
      end

      def references
        @references ||= discover_references
      end

      def available?
        @metadata.available?
      end

      def matches_trigger?(query)
        return false unless query.is_a?(String)

        normalized_query = query.downcase
        @metadata.triggers.any? { |t| normalized_query.include?(t.downcase) }
      end

      def matches_anti_trigger?(query)
        return false unless query.is_a?(String)

        normalized_query = query.downcase
        @metadata.anti_triggers.any? { |t| normalized_query.include?(t.downcase) }
      end

      def entrypoint_for(action = "default")
        @metadata.entrypoints.find { |ep| ep.name == action }
      end

      def always_load?
        @metadata.always
      end

      def parallel_safe?
        @metadata.parallel_safe
      end

      def description
        @metadata.description
      end

      def version
        @metadata.version
      end

      def to_h
        {
          name: @name,
          type: @type,
          source_path: @source_path,
          description: description,
          version: version,
          available: available?,
          always: always_load?,
          parallel_safe: parallel_safe?,
          metadata: @metadata.to_h
        }
      end

      private

      def normalize_name(name)
        name.to_s.downcase
            .gsub(/[^a-z0-9]+/, "_")
            .gsub(/^_+|_+$/, "")
            .gsub(/_+/, "_")
      end

      def validate_type(type)
        type_sym = type.to_sym
        raise ArgumentError, "Invalid skill type: #{type}" unless SKILL_TYPES.include?(type_sym)

        type_sym
      end

      def discover_scripts
        scripts_dir = File.join(@source_path, "scripts")
        return [] unless File.directory?(scripts_dir)

        Dir.glob(File.join(scripts_dir, "*")).select { |f| File.file?(f) }
      end

      def discover_references
        refs_dir = File.join(@source_path, "references")
        return [] unless File.directory?(refs_dir)

        Dir.glob(File.join(refs_dir, "*")).select { |f| File.file?(f) }
      end
    end
  end
end
