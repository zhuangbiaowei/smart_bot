# frozen_string_literal: true

module SmartBot
  module SkillSystem
    module Adapters
      # Adapter for converting OpenClaw format skills to SmartBot internal format
      class OpenClawAdapter
        OPENCLAW_METADATA_KEYS = %w[openclaw claude].freeze

        # Check if frontmatter contains OpenClaw metadata
        # @param frontmatter [Hash] Parsed YAML frontmatter
        # @return [Boolean]
        def self.can_parse?(frontmatter)
          return false unless frontmatter.is_a?(Hash)

          metadata = frontmatter["metadata"]
          return false unless metadata.is_a?(Hash)

          OPENCLAW_METADATA_KEYS.any? { |key| metadata.key?(key) }
        end

        # Convert OpenClaw format to SkillMetadata
        # @param frontmatter [Hash] Parsed YAML frontmatter
        # @param content [String] Full skill content
        # @return [SkillMetadata]
        def self.convert(frontmatter, content)
          new(frontmatter, content).convert
        end

        def initialize(frontmatter, content)
          @frontmatter = frontmatter
          @content = content
          @oc_metadata = extract_openclaw_metadata
        end

        def convert
          explicit_triggers = Array(@frontmatter["triggers"]).compact
          explicit_anti_triggers = Array(@frontmatter["anti_triggers"]).compact

          SkillMetadata.new(
            name: extract_name,
            description: extract_description,
            version: extract_version,
            author: extract_author,
            triggers: explicit_triggers.empty? ? infer_triggers : explicit_triggers,
            anti_triggers: explicit_anti_triggers,
            cost_hint: (@frontmatter["cost_hint"] || :medium).to_sym,
            always: @frontmatter.fetch("always", @oc_metadata["always"] || false),
            parallel_safe: @frontmatter.fetch("parallel_safe", false),
            prerequisites: build_prerequisites,
            permissions: build_permissions,
            execution_policy: build_execution_policy,
            entrypoints: []
          ).tap do |metadata|
            # Attach OpenClaw-specific metadata as a struct attribute
            metadata.instance_variable_set(:@openclaw_meta, {
              emoji: @oc_metadata["emoji"],
              homepage: @frontmatter["homepage"] || @oc_metadata["homepage"],
              skill_key: @oc_metadata["skillKey"] || extract_name,
              os: @oc_metadata["os"] || [],
              primary_env: @oc_metadata["primaryEnv"],
              install: @oc_metadata["install"] || []
            })
            
            # Define accessor for openclaw_meta
            metadata.define_singleton_method(:openclaw_meta) do
              @openclaw_meta || {}
            end
          end
        end

        private

        def extract_openclaw_metadata
          metadata = @frontmatter["metadata"] || {}
          
          # Try openclaw first, then claude (for compatibility)
          oc_meta = metadata["openclaw"] || metadata["claude"] || {}
          oc_meta.is_a?(Hash) ? oc_meta : {}
        end

        def extract_name
          # Priority: frontmatter.name > metadata.skillKey > inferred from content
          @frontmatter["name"] ||
            @oc_metadata["skillKey"] ||
            infer_name_from_content
        end

        def extract_description
          @frontmatter["description"] ||
            @frontmatter["desc"] ||
            infer_description_from_content
        end

        def extract_version
          @frontmatter["version"] ||
            @oc_metadata["version"] ||
            "0.1.0"
        end

        def extract_author
          @frontmatter["author"] ||
            @frontmatter["author_name"] ||
            @oc_metadata["author"] ||
            "Unknown"
        end

        def infer_triggers
          triggers = []

          # 1. Add skill name
          name = extract_name
          triggers << name if name

          # 2. Add name variations (split by - or _)
          if name
            parts = name.split(/[-_]/)
            triggers.concat(parts) if parts.length > 1
          end

          # 3. Extract keywords from description
          desc = extract_description.to_s.downcase
          words = desc.scan(/\b[a-z0-9\u4e00-\u9fa5]{2,}\b/)
          
          # Add meaningful words (3+ chars or Chinese)
          words.each do |word|
            triggers << word if word.length >= 3 || word.match?(/\p{Han}/)
          end

          # 4. Add common trigger words based on skill type
          triggers.concat(infer_common_triggers(name, desc))

          triggers.uniq.compact
        end

        def infer_common_triggers(name, desc)
          common = []
          name_str = name.to_s.downcase
          desc_str = desc.to_s.downcase

          # Define trigger mappings
          trigger_mappings = {
            /weather|天气|forecast|预报/ => ["weather", "天气", "forecast", "预报", "temperature", "温度", "sunny", "rain"],
            /search|搜索|find|查找/ => ["search", "搜索", "find", "查找", "lookup", "query", "查询"],
            /image|图片|照片|photo/ => ["image", "图片", "photo", "照片", "picture", "pic", "图"],
            /video|视频|movie|电影/ => ["video", "视频", "movie", "电影", "film", "clip"],
            /audio|音频|sound|音乐|music/ => ["audio", "音频", "sound", "音乐", "music", "song", "voice"],
            /code|代码|program|编程/ => ["code", "代码", "program", "编程", "script", "脚本", "develop"],
            /web|网页|fetch|抓取|scrape/ => ["web", "网页", "fetch", "抓取", "scrape", "download", "下载"],
            /file|文件|doc|document/ => ["file", "文件", "document", "文档", "read", "write"],
            /git|版本|version|commit/ => ["git", "版本", "commit", "branch", "merge"],
            /health|健康|check|检查|security|安全/ => ["health", "健康", "check", "检查", "security", "安全", "audit", "审计"],
            /note|笔记|memo/ => ["note", "笔记", "memo", "记录"],
            /reminder|提醒|alarm/ => ["reminder", "提醒", "alarm", "定时"],
            /calendar|日历|schedule/ => ["calendar", "日历", "schedule", "日程"],
            /email|邮件|mail/ => ["email", "邮件", "mail", "message"],
            /slack|discord|chat|消息/ => ["slack", "discord", "chat", "消息", "im"],
            /translate|翻译/ => ["translate", "翻译", "language", "语言"],
            /summarize|总结|summary/ => ["summarize", "总结", "summary", "摘要", "outline"],
            /generate|生成|create/ => ["generate", "生成", "create", "创建", "make"],
            /analyze|分析|analysis/ => ["analyze", "分析", "analysis", "统计"],
            /convert|转换|transform/ => ["convert", "转换", "transform", "change"]
          }

          trigger_mappings.each do |pattern, words|
            if name_str.match?(pattern) || desc_str.match?(pattern)
              common.concat(words)
            end
          end

          common
        end

        def infer_name_from_content
          # Extract from first heading if available
          if @content =~ /^#\s+(.+)$/m
            $1.strip.downcase.gsub(/\s+/, "-").gsub(/[^a-z0-9_-]/, "")
          else
            "unnamed-skill"
          end
        end

        def infer_description_from_content
          # Find first non-empty, non-heading line
          lines = @content.lines.map(&:strip).reject(&:empty?)
          lines.find { |l| !l.start_with?("#") }
        end

        def build_prerequisites
          prereqs = []
          requires = @oc_metadata["requires"] || {}

          # Binary dependencies
          if requires["bins"]
            requires["bins"].each do |bin|
              prereqs << Prerequisite.new(type: :bin, name: bin)
            end
          end

          # Any-of binary dependencies (anyBins)
          if requires["anyBins"] && requires["anyBins"].is_a?(Array)
            prereqs << Prerequisite.new(
              type: :any_bins,
              name: requires["anyBins"]
            )
          end

          # Environment variable dependencies
          if requires["env"]
            requires["env"].each do |env|
              prereqs << Prerequisite.new(type: :env, name: env)
            end
          end

          # Config path dependencies (OpenClaw style)
          if requires["config"]
            requires["config"].each do |config_path|
              prereqs << Prerequisite.new(
                type: :openclaw_config,
                name: config_path
              )
            end
          end

          # OS requirements
          os_list = @oc_metadata["os"] || []
          if os_list.any?
            prereqs << Prerequisite.new(
              type: :openclaw_os,
              name: os_list
            )
          end

          prereqs
        end

        def build_permissions
          # Infer permissions from content analysis
          content_lower = @content.downcase

          needs_network = content_lower.match?(/\b(curl|wget|fetch|http|https|api|url|download|get|post)\b/)
          needs_filesystem = content_lower.match?(/\b(file|read|write|save|path|dir|folder|目录|文件|读取|保存)\b/)
          needs_env = content_lower.match?(/\b(env|environment|变量|config|setting)\b/)

          PermissionSet.new(
            filesystem: needs_filesystem ? { read: ["~"], write: ["/tmp", "~/.smart_bot"] } : { read: [], write: [] },
            network: { outbound: needs_network },
            environment: { allow: needs_env ? ["*"] : [] }
          )
        end

        def build_execution_policy
          ExecutionPolicy.new(
            sandbox: :process,
            approval: :ask,
            timeout: 120
          )
        end
      end
    end
  end
end
