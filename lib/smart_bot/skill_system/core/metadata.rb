# frozen_string_literal: true

module SmartBot
  module SkillSystem
    # Unified metadata model for skills
    # Parses both skill.yaml and SKILL.md frontmatter
    class SkillMetadata
      attr_reader :name, :description, :version, :license, :author
      attr_reader :triggers, :anti_triggers, :cost_hint
      attr_reader :prerequisites, :permissions, :execution_policy
      attr_reader :entrypoints, :parallel_safe, :always
      attr_accessor :openclaw_meta  # Store OpenClaw-specific metadata

      DEFAULT_VERSION = "0.1.0"
      DEFAULT_AUTHOR = "Unknown"

      def initialize(attrs = {})
        @name = attrs[:name]
        @description = attrs[:description]
        @version = attrs[:version] || DEFAULT_VERSION
        @license = attrs[:license]
        @author = attrs[:author] || DEFAULT_AUTHOR

        @triggers = Array(attrs[:triggers])
        @anti_triggers = Array(attrs[:anti_triggers])
        @cost_hint = attrs[:cost_hint] || :medium
        @always = attrs[:always] || false
        @parallel_safe = attrs[:parallel_safe] || false

        @prerequisites = if attrs[:prerequisites].is_a?(Array)
          # Accept pre-built prerequisite arrays, including empty arrays.
          attrs[:prerequisites]
        else
          build_prerequisites(attrs[:prerequisites] || {})
        end
        
        @permissions = if attrs[:permissions].is_a?(PermissionSet)
          attrs[:permissions]
        else
          build_permissions(attrs[:permissions] || {})
        end
        
        @execution_policy = if attrs[:execution_policy].is_a?(ExecutionPolicy)
          attrs[:execution_policy]
        else
          build_execution_policy(attrs[:execution_policy] || {})
        end
        
        @entrypoints = build_entrypoints(attrs[:entrypoints] || [])
        
        @openclaw_meta = attrs[:openclaw_meta] || {}
      end

      def self.from_skill_yaml(yaml_content)
        return nil unless yaml_content.is_a?(Hash)

        spec = yaml_content["spec"] || {}
        metadata = yaml_content["metadata"] || {}

        new(
          name: metadata["name"],
          description: metadata["description"],
          version: metadata["version"],
          license: metadata["license"],
          author: metadata["author"],
          triggers: spec["triggers"],
          anti_triggers: spec["anti_triggers"],
          cost_hint: spec["cost_hint"]&.to_sym,
          always: spec["always"],
          parallel_safe: spec["parallel_safe"],
          prerequisites: spec["prerequisites"],
          permissions: spec["permissions"],
          execution_policy: spec["execution"],
          entrypoints: spec["entrypoints"]
        )
      end

      def self.from_frontmatter(frontmatter, description_fallback = nil)
        return nil unless frontmatter.is_a?(Hash)

        new(
          name: frontmatter["name"],
          description: frontmatter["description"] || description_fallback,
          version: frontmatter["version"],
          license: frontmatter["license"],
          author: frontmatter["author"],
          triggers: frontmatter["triggers"],
          anti_triggers: frontmatter["anti_triggers"],
          cost_hint: frontmatter["cost_hint"]&.to_sym,
          always: frontmatter["always"],
          parallel_safe: frontmatter["parallel_safe"]
        )
      end

      def cost_penalty
        case @cost_hint
        when :low then 0.0
        when :medium then -0.05
        when :high then -0.10
        else 0.0
        end
      end

      def available?
        @prerequisites.all?(&:satisfied?)
      end

      def to_h
        {
          name: @name,
          description: @description,
          version: @version,
          license: @license,
          author: @author,
          triggers: @triggers,
          anti_triggers: @anti_triggers,
          cost_hint: @cost_hint,
          always: @always,
          parallel_safe: @parallel_safe,
          prerequisites: @prerequisites.map(&:to_h),
          permissions: @permissions.to_h,
          execution_policy: @execution_policy.to_h,
          entrypoints: @entrypoints.map(&:to_h),
          available: available?,
          openclaw_meta: @openclaw_meta
        }
      end

      private

      def build_prerequisites(prereq_hash)
        prereqs = []

        if prereq_hash["system"]
          prereq_hash["system"].each do |bin|
            prereqs << Prerequisite.new(type: :bin, name: bin)
          end
        end

        if prereq_hash["env"]
          prereq_hash["env"].each do |env_var|
            prereqs << Prerequisite.new(type: :env, name: env_var)
          end
        end

        if prereq_hash["files"]
          prereq_hash["files"].each do |file|
            prereqs << Prerequisite.new(type: :file, name: file)
          end
        end

        prereqs
      end

      def build_permissions(perm_hash)
        PermissionSet.new(
          filesystem: perm_hash["filesystem"] || {},
          network: perm_hash["network"] || {},
          environment: perm_hash["environment"] || {}
        )
      end

      def build_execution_policy(policy_hash)
        ExecutionPolicy.new(
          sandbox: (policy_hash["sandbox"] || "process").to_sym,
          approval: (policy_hash["approval"] || "ask").to_sym,
          timeout: policy_hash["timeout"] || 120
        )
      end

      def build_entrypoints(entrypoints_array)
        return [] unless entrypoints_array.is_a?(Array)

        entrypoints_array.map do |ep|
          Entrypoint.new(
            name: ep["name"] || "default",
            runtime: ep["runtime"] || "ruby",
            command: ep["command"],
            inputs: ep["inputs"] || {},
            outputs: ep["outputs"] || {}
          )
        end
      end
    end
  end
end
