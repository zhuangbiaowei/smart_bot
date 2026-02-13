# frozen_string_literal: true

require "json"
require "open3"
require "tmpdir"
require "uri"

module SmartBot
  module SkillSystem
  module Execution
      # Executor for OpenClaw format skills
      # Handles LLM-driven execution of Markdown-based skills
      class OpenClawExecutor
        attr_reader :llm_engine, :llm_name, :config

        DEFAULT_LLM = "deepseek"
        DEFAULT_TIMEOUT = 120

        def initialize(llm_engine: nil, llm_name: nil, config: {})
          @llm_engine = llm_engine
          @llm_name = llm_name || DEFAULT_LLM
          @config = config
        end

        # Execute an OpenClaw skill
        #
        # @param skill [SkillPackage] The skill to execute
        # @param parameters [Hash] Execution parameters including :task
        # @param context [Hash] Additional context
        # @return [ExecutionResult]
        def execute(skill, parameters, context = {})
          validate_skill!(skill)

          task = extract_task(parameters)
          unless task
            return ExecutionResult.failure(
              skill: skill,
              error: "No task provided for skill execution"
            )
          end

          # Extract frontmatter metadata for enhanced context
          oc_meta = extract_openclaw_metadata(skill)

          # Special handling for skills that require real tool execution
          if youtube_summarizer_skill?(skill, task)
            return execute_youtube_summarizer(skill, task, context)
          end

          # Build system prompt from skill content
          system_prompt = build_system_prompt(skill, oc_meta)

          # Build user prompt with task and context
          user_prompt = build_user_prompt(task, context, parameters)

          # Execute via LLM
          result = execute_via_llm(system_prompt, user_prompt, skill, llm: context[:llm] || context["llm"])

          ExecutionResult.success(
            skill: skill,
            value: {
              success: true,
              output: result[:output],
              execution_time: result[:execution_time],
              worker_name: result[:worker_name]
            },
            metadata: {
              executor: :openclaw,
              format: :instruction,
              skill_name: skill.name,
              skill_description: skill.metadata&.description,
              has_frontmatter: !oc_meta.empty?,
              execution_time: result[:execution_time]
            }
          )
        rescue Timeout::Error => e
          ExecutionResult.failure(
            skill: skill,
            error: "Execution timeout after #{timeout}s: #{e.message}",
            metadata: { executor: :openclaw, timeout: timeout }
          )
        rescue => e
          ExecutionResult.failure(
            skill: skill,
            error: "OpenClaw execution failed: #{e.message}",
            metadata: { executor: :openclaw, error_class: e.class.name }
          )
        end

        # Quick execution without full setup
        # @param skill_path [String] Path to skill directory
        # @param task [String] Task to execute
        # @return [ExecutionResult]
        def self.execute_file(skill_path, task, context = {})
          skill_md = File.join(skill_path, "SKILL.md")
          unless File.exist?(skill_md)
            return ExecutionResult.failure(
              skill: nil,
              error: "SKILL.md not found in #{skill_path}"
            )
          end

          content = File.read(skill_md, encoding: "UTF-8")
          executor = new

          # Create a minimal skill package
          skill = OpenStruct.new(
            name: File.basename(skill_path),
            source_path: skill_path,
            content: content,
            metadata: OpenStruct.new(description: "OpenClaw skill from #{skill_path}"),
            type: :openclaw_instruction
          )

          executor.execute(skill, { task: task }, context)
        end

        private

        def validate_skill!(skill)
          unless skill.respond_to?(:content)
            raise ArgumentError, "Skill must have content method"
          end

          unless skill.respond_to?(:name)
            raise ArgumentError, "Skill must have name method"
          end
        end

        def extract_task(parameters)
          task = parameters[:task] || parameters["task"]
          return task if task

          # Try to extract from query parameter
          query = parameters[:query] || parameters["query"]
          return query if query

          # Try to build from other parameters
          if parameters.any?
            relevant_params = parameters.reject { |k, _| [:context, :history].include?(k) }
            relevant_params.map { |k, v| "#{k}: #{v}" }.join(", ")
          end
        end

        def extract_openclaw_metadata(skill)
          return {} unless skill.respond_to?(:metadata) && skill.metadata

          # Check for openclaw_meta field (set by adapter)
          if skill.metadata.respond_to?(:openclaw_meta)
            return skill.metadata.openclaw_meta || {}
          end

          # Try to parse from raw content
          if skill.respond_to?(:content) && skill.content
            parse_frontmatter(skill.content)&.dig("metadata", "openclaw") || {}
          else
            {}
          end
        end

        def parse_frontmatter(content)
          return nil unless content.start_with?("---")

          if content =~ /\A---\s*\n(.+?)\n---\s*\n/m
            yaml_content = $1
            begin
              YAML.safe_load(yaml_content, permitted_classes: [Date, Time, Symbol], aliases: true)
            rescue Psych::SyntaxError
              nil
            end
          else
            nil
          end
        end

        def build_system_prompt(skill, oc_meta)
          content = skill.respond_to?(:load_full_content) ? skill.load_full_content : skill.content

          # Extract instruction content (after frontmatter)
          instruction_content = extract_instruction_content(content)

          # Get skill metadata
          name = skill.name
          description = skill.metadata&.description || "Execute #{name} skill"
          emoji = oc_meta[:emoji] || "📦"
          homepage = oc_meta[:homepage]

          # Build enhanced system prompt
          prompt_parts = []

          # Header with skill identity
          prompt_parts << "#{emoji} You are an expert assistant specialized in: #{name}"
          prompt_parts << ""
          prompt_parts << "Description: #{description}"
          prompt_parts << "Homepage: #{homepage}" if homepage
          prompt_parts << ""

          # Core instructions
          prompt_parts << "## Instructions"
          prompt_parts << ""
          prompt_parts << instruction_content
          prompt_parts << ""

          # Guidelines for execution
          prompt_parts << "## Guidelines"
          prompt_parts << ""
          prompt_parts << "1. Follow the instructions above carefully and precisely"
          prompt_parts << "2. Use available tools when needed to complete the task"
          prompt_parts << "3. Be thorough but concise in your responses"
          prompt_parts << "4. If you need clarification, ask specific questions"
          prompt_parts << "5. When executing commands, show the commands you use"
          prompt_parts << "6. Provide clear, actionable results"
          prompt_parts << ""

          # Tool usage reminder (if skill mentions tools)
          if instruction_content.include?("tool") || instruction_content.include?("执行") || instruction_content.include?("命令")
            prompt_parts << "## Tool Usage"
            prompt_parts << ""
            prompt_parts << "You have access to various tools. Use them when appropriate:"
            prompt_parts << "- read: Read files to get information"
            prompt_parts << "- write: Write files to save results"
            prompt_parts << "- exec: Execute shell commands"
            prompt_parts << "- web_fetch: Fetch web pages"
            prompt_parts << ""
          end

          prompt_parts.join("\n")
        end

        def extract_instruction_content(content)
          # Remove frontmatter
          if content =~ /\A---\s*\n.*?\n---\s*\n(.*)\z/m
            $1.strip
          else
            content.strip
          end
        end

        def build_user_prompt(task, context, parameters)
          parts = []

          # Main task
          parts << "## Task"
          parts << ""
          parts << task
          parts << ""

          # Context information
          if context && !context.empty?
            parts << "## Context"
            parts << ""
            context.each do |key, value|
              parts << "#{key}: #{value}"
            end
            parts << ""
          end

          # Additional parameters
          other_params = parameters.except(:task, :query, :context)
          if other_params.any?
            parts << "## Parameters"
            parts << ""
            other_params.each do |key, value|
              parts << "- #{key}: #{value}"
            end
            parts << ""
          end

          parts.join("\n")
        end

        def execute_youtube_summarizer(skill, task, context)
          url = extract_youtube_url(task)
          unless url
            return ExecutionResult.failure(
              skill: skill,
              error: "No YouTube URL found in request"
            )
          end

          unless command_available?("yt-dlp")
            return ExecutionResult.failure(
              skill: skill,
              error: "yt-dlp is not available in PATH. Please install it first."
            )
          end

          metadata = fetch_youtube_metadata(url)
          unless metadata
            return ExecutionResult.failure(
              skill: skill,
              error: "Failed to fetch YouTube metadata. The video may be unavailable or restricted."
            )
          end

          transcript, transcript_diag = fetch_youtube_transcript(url, metadata["id"])
          if transcript.to_s.strip.empty?
            return ExecutionResult.failure(
              skill: skill,
              error: "Transcript not available for this video. #{transcript_diag}".strip
            )
          end

          summary_prompt = build_youtube_summary_prompt(url, metadata, transcript, task)
          llm_result = execute_via_llm(
            "You are a factual video summarizer. Only use provided metadata/transcript. Never fabricate execution steps.",
            summary_prompt,
            skill,
            llm: context[:llm] || context["llm"]
          )

          ExecutionResult.success(
            skill: skill,
            value: {
              success: true,
              output: llm_result[:output],
              execution_time: llm_result[:execution_time],
              worker_name: llm_result[:worker_name],
              source: :yt_dlp
            },
            metadata: {
              executor: :openclaw,
              format: :youtube_verified,
              skill_name: skill.name,
              execution_time: llm_result[:execution_time]
            }
          )
        rescue => e
          ExecutionResult.failure(
            skill: skill,
            error: "YouTube summarization failed: #{e.message}"
          )
        end

        def youtube_summarizer_skill?(skill, task)
          return false unless task.to_s.match?(%r{https?://(www\.)?(youtube\.com|youtu\.be)/}i)

          skill_name = skill.name.to_s.downcase
          skill_name.include?("youtube") && skill_name.include?("summar")
        end

        def execute_via_llm(system_prompt, user_prompt, skill, llm: nil)
          start_time = Time.now

          engine = llm_engine || default_llm_engine
          selected_llm = llm || @llm_name

          worker_name = :"openclaw_skill_#{skill.name}_#{Time.now.to_i}"

          SmartPrompt.define_worker worker_name do
            use selected_llm
            sys_msg system_prompt
            prompt user_prompt
            send_msg
          end

          result = engine.call_worker(worker_name, {})
          execution_time = Time.now - start_time

          {
            output: result,
            execution_time: execution_time.round(2),
            worker_name: worker_name
          }
        end

        def extract_youtube_url(text)
          return nil if text.to_s.empty?

          match = text.match(%r{https?://[^\s]+})
          return nil unless match

          url = match[0]
          uri = URI.parse(url)
          return nil unless uri.host&.match?(/youtube\.com|youtu\.be/i)

          normalize_youtube_url(url)
        rescue URI::InvalidURIError
          nil
        end

        def fetch_youtube_metadata(url)
          yt_dlp_arg_sets.each do |arg_set|
            stdout, _stderr, status = Open3.capture3(
              "yt-dlp",
              *arg_set,
              "--dump-json",
              "--skip-download",
              url
            )
            next unless status.success? && !stdout.to_s.strip.empty?

            return JSON.parse(stdout)
          end

          nil
        rescue JSON::ParserError
          nil
        end

        def fetch_youtube_transcript(url, video_id)
          Dir.mktmpdir("smartbot_yt_") do |tmpdir|
            output_template = File.join(tmpdir, "%(id)s.%(ext)s")

            attempts = [
              ["--write-auto-subs", "--write-subs", "--sub-langs", "en.*,en-US,en-GB,en,zh.*,zh", "--sub-format", "vtt/srt/json3", "--skip-download"],
              ["--write-auto-subs", "--write-subs", "--sub-langs", "all", "--sub-format", "vtt/srt/json3/best", "--skip-download"],
              ["--write-auto-subs", "--write-subs", "--skip-download"]
            ]

            diagnostics = []

            yt_dlp_arg_sets.each do |arg_set|
              attempts.each do |flags|
                _stdout, stderr, _status = Open3.capture3(
                  "yt-dlp",
                  *arg_set,
                  *flags,
                  "-o",
                  output_template,
                  url
                )
                diagnostics << stderr.to_s.strip unless stderr.to_s.strip.empty?

                subtitle_files = find_subtitle_files(tmpdir, video_id)
                subtitle_files.each do |file_path|
                  content = parse_subtitle_file(file_path)
                  return [content, ""] unless content.to_s.strip.empty?
                end
              end
            end

            diag = diagnostics.reject(&:empty?).join(" | ")[0, 500]
            diag = "Please try a video with captions enabled." if diag.to_s.empty?
            if diag.match?(/android client https formats require a GVS PO Token/i)
              diag = "Subtitles were not retrieved. The video may not expose captions to unauthenticated requests. Try another video with public captions."
            end
            if diag.match?(/remote components challenge solver|n challenge solving failed|supported Java/i)
              diag = "#{diag} | Hint: update yt-dlp and install deno, or allow remote components for JS challenge solving."
            end
            [nil, diag]
          end
        end

        def find_subtitle_files(tmpdir, video_id)
          patterns = if video_id.to_s.empty?
                       [File.join(tmpdir, "*.*")]
                     else
                       [File.join(tmpdir, "#{video_id}*.*")]
                     end

          files = patterns.flat_map { |p| Dir.glob(p) }.select { |f| File.file?(f) }
          preferred_order = %w[.vtt .srt .json3 .srv3 .ttml .txt]

          files.sort_by do |path|
            ext = File.extname(path).downcase
            idx = preferred_order.index(ext)
            idx.nil? ? 999 : idx
          end
        end

        def parse_subtitle_file(file_path)
          ext = File.extname(file_path).downcase
          raw_text = File.read(file_path, encoding: "UTF-8")

          case ext
          when ".json3", ".srv3"
            normalize_json3_subtitle(raw_text)
          else
            normalize_subtitle_text(raw_text)
          end
        rescue => _e
          ""
        end

        def normalize_subtitle_text(raw_text)
          return "" if raw_text.to_s.empty?

          lines = raw_text.lines.map(&:strip)
          cleaned = lines.reject do |line|
            line.empty? ||
              line.match?(/^\d+$/) ||
              line.match?(/^\d{2}:\d{2}:\d{2}[.,]\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}[.,]\d{3}$/) ||
              line.match?(/^\d{2}:\d{2}[.,]\d{3}\s+-->\s+\d{2}:\d{2}[.,]\d{3}$/) ||
              line.match?(/^\d{2}:\d{2}:\d{2}\.\d+\s+-->\s+\d{2}:\d{2}:\d{2}\.\d+$/) ||
              line == "WEBVTT" ||
              line.start_with?("Kind:") ||
              line.start_with?("Language:")
          end

          deduped = []
          cleaned.each do |line|
            normalized_line = line.gsub(/<[^>]+>/, "").gsub("&nbsp;", " ").strip
            next if normalized_line.empty?

            deduped << normalized_line unless deduped.last == normalized_line
          end

          deduped.join("\n")
        end

        def normalize_json3_subtitle(raw_json)
          data = JSON.parse(raw_json)
          events = data["events"] || []
          chunks = []

          events.each do |event|
            segs = event["segs"] || []
            text = segs.map { |seg| seg["utf8"].to_s }.join
            chunks << text unless text.empty?
          end

          normalize_subtitle_text(chunks.join("\n"))
        rescue JSON::ParserError
          ""
        end

        def build_youtube_summary_prompt(url, metadata, transcript, user_task)
          title = metadata["title"] || "Unknown title"
          uploader = metadata["uploader"] || "Unknown channel"
          views = metadata["view_count"] || "unknown"
          upload_date = metadata["upload_date"] || "unknown"
          duration = metadata["duration"] || "unknown"
          clipped_transcript = transcript[0, 24_000]

          <<~PROMPT
            You must summarize this exact YouTube video using only the provided data.
            If data is insufficient, explicitly say what is missing. Do not claim commands were run.

            User request:
            #{user_task}

            Video URL: #{url}
            Title: #{title}
            Channel: #{uploader}
            Views: #{views}
            Upload Date: #{upload_date}
            Duration (seconds): #{duration}

            Transcript (possibly truncated):
            #{clipped_transcript}

            Output in Chinese with this structure:
            1) 视频信息
            2) 核心观点（1-2句）
            3) 关键要点（3-5条）
            4) 结论
            5) 可信度说明（说明是否因字幕缺失/截断导致不确定）
          PROMPT
        end

        def command_available?(command)
          stdout, _stderr, status = Open3.capture3("which", command)
          status.success? && !stdout.to_s.strip.empty?
        end

        def yt_dlp_arg_sets
          [
            ["--remote-components", "ejs:github", "--extractor-args", "youtube:player_client=web,web_creator,mweb"],
            ["--remote-components", "ejs:github"],
            []
          ]
        end

        def normalize_youtube_url(url)
          uri = URI.parse(url)
          host = uri.host.to_s.downcase

          if host.include?("youtube.com") && uri.path.start_with?("/shorts/")
            video_id = uri.path.split("/")[2]
            return "https://www.youtube.com/watch?v=#{video_id}" if video_id && !video_id.empty?
          end

          url
        rescue URI::InvalidURIError
          url
        end

        def default_llm_engine
          config_path = File.expand_path("~/.smart_bot/smart_prompt.yml")

          unless File.exist?(config_path)
            raise "SmartPrompt config not found at #{config_path}"
          end

          SmartPrompt::Engine.new(config_path)
        end

        def timeout
          @config[:timeout] || DEFAULT_TIMEOUT
        end
      end
    end
  end
end
