# frozen_string_literal: true

require_relative "result"

module SmartBot
  module SkillSystem
    # Repair Loop for self-healing skill execution failures
    # Attempts to fix "near-success" failures automatically
    class RepairLoop
      MAX_REPAIR_ATTEMPTS = 2
      PROGRESS_THRESHOLD = 0.70
      MAX_PATCHED_FILES = 3
      MAX_PATCH_HUNKS = 8

      attr_reader :executor, :observer, :budget, :repair_confirmation_callback

      def initialize(executor:, observer: nil, repair_confirmation_callback: nil)
        @executor = executor
        @observer = observer
        @repair_confirmation_callback = repair_confirmation_callback
        @budget = RepairBudget.new(
          max_attempts: MAX_REPAIR_ATTEMPTS,
          max_patched_files: MAX_PATCHED_FILES,
          max_patched_hunks: MAX_PATCH_HUNKS
        )
      end

      def execute_with_repair(skill, parameters, context)
        result = @executor.execute_skill(skill, parameters, context)
        return result if result.success?

        unless repairable?(result)
          @observer&.repair_skipped(skill, result, "Not repairable")
          return result
        end

        attempt_repair(skill, parameters, context, result)
      end

      private

      def repairable?(result)
        return false unless result.failure?
        return false unless result.skill

        error = result.error.to_s.downcase

        repairable_patterns = [
          /parameter/i,
          /missing.*field/i,
          /not found/i,
          /path.*error/i,
          /template/i,
          /reference/i,
          /undefined/i,
          /no such file/i,
          /cannot find/i,
          /invalid.*argument/i
        ]

        repairable_patterns.any? { |p| error =~ p }
      end

      def near_success?(result)
        return false unless result.failure?

        metadata = result.metadata || {}

        checks = [
          metadata[:task_progress].to_f >= PROGRESS_THRESHOLD,
          metadata[:root_cause_count].to_i <= 2,
          metadata[:change_scope_within_skill] != false,
          metadata[:estimated_fix_attempts].to_i <= 1
        ]

        checks.count(true) >= 3
      end

      def attempt_repair(skill, parameters, context, original_result)
        attempt = 0
        current_result = original_result
        last_error_signature = error_signature(original_result.error)

        while attempt < MAX_REPAIR_ATTEMPTS && @budget.has_budget?
          attempt += 1
          @observer&.repair_attempted(skill, attempt)

          diagnosis = diagnose_failure(skill, current_result)
          repair_plan = generate_repair_plan(skill, diagnosis)

          unless repair_plan.valid?
            @observer&.repair_failed(skill, attempt, "Invalid repair plan")
            break
          end

          decision = confirm_repair(skill, attempt, diagnosis, repair_plan)
          unless decision[:approved]
            @observer&.repair_failed(skill, attempt, "Repair rejected by user")
            break
          end

          apply_user_suggestion(repair_plan, decision[:suggestion])

          patches_applied = apply_patches(skill, repair_plan)

          unless patches_applied.any?
            @observer&.repair_failed(skill, attempt, "No patches applied")
            break
          end

          @budget.consume_patch(
            files: patches_applied.size,
            hunks: repair_plan.total_hunks
          )

          new_result = @executor.execute_skill(skill, parameters, context)

          if improved?(current_result, new_result, last_error_signature)
            current_result = new_result
            last_error_signature = error_signature(new_result.error) if new_result.failure?

            if new_result.success?
              @observer&.repair_succeeded(skill, attempt)
              return new_result
            end

            unless repairable?(new_result) && near_success?(new_result)
              @observer&.repair_failed(skill, attempt, "No longer repairable")
              break
            end
          else
            @observer&.repair_no_improvement(skill, attempt)
            break
          end

          @budget.consume_attempt
        end

        current_result
      end

      def confirm_repair(skill, attempt, diagnosis, repair_plan)
        return { approved: true, suggestion: nil } unless @repair_confirmation_callback

        result = @repair_confirmation_callback.call(
          skill: skill,
          attempt: attempt,
          diagnosis: diagnosis.to_h,
          repair_plan: repair_plan.to_h
        )

        case result
        when false
          { approved: false, suggestion: nil }
        when Hash
          { approved: !!result[:approved], suggestion: result[:suggestion] }
        when String
          { approved: true, suggestion: result }
        else
          { approved: true, suggestion: nil }
        end
      rescue => e
        @observer&.repair_failed(skill, attempt, "Repair confirmation failed: #{e.message}")
        { approved: false, suggestion: nil }
      end

      def apply_user_suggestion(repair_plan, suggestion)
        suggestion_text = suggestion.to_s.strip
        return if suggestion_text.empty?

        repair_plan.add_patch(
          file: "SKILL.md",
          description: "Apply user-provided repair guidance",
          action: :append_section,
          content: <<~SECTION
            ## User Repair Guidance

            #{suggestion_text}
          SECTION
        )
      end

      def diagnose_failure(skill, result)
        error = result.error.to_s
        error_type = classify_error(error)

        Diagnosis.new(
          error_type: error_type,
          error_message: error,
          error_location: locate_error(skill, error),
          affected_files: affected_files(skill, error, error_type),
          skill_path: skill.source_path,
          metadata: result.metadata || {}
        )
      end

      def classify_error(error)
        error_lower = error.to_s.downcase

        case error_lower
        when /parameter|argument|param/
          :parameter_error
        when /template|reference|undefined/
          :template_error
        when /permission|denied|unauthorized/
          :permission_error
        when /timeout|timed out/
          :timeout_error
        when /syntax|parse|invalid.*format/
          :syntax_error
        when /file|path|directory|not found|no such file/
          :file_error
        when /missing/
          :template_error
        else
          :unknown_error
        end
      end

      def locate_error(skill, error)
        return { file: nil, line: nil } unless error.is_a?(String)

        file_match = error.match(/#{skill.source_path}\/([^:]+):(\d+)/)
        return { file: file_match[1], line: file_match[2].to_i } if file_match

        file_match = error.match(/([^\s:]+\.rb|\.py|\.sh|\.js):(\d+)/)
        return { file: file_match[1], line: file_match[2].to_i } if file_match

        { file: nil, line: nil }
      end

      def affected_files(skill, error, error_type)
        files = []

        files << "SKILL.md"

        case error_type
        when :parameter_error
          files << "skill.yaml" if File.exist?(File.join(skill.source_path, "skill.yaml"))
        when :file_error
          if error =~ /scripts\/(\w+)/
            files << "scripts/#{$1}"
          end
        when :template_error
          files << "references/" if Dir.exist?(File.join(skill.source_path, "references"))
        end

        files.uniq
      end

      def generate_repair_plan(skill, diagnosis)
        plan = RepairPlan.new(skill: skill, diagnosis: diagnosis)

        case diagnosis.error_type
        when :parameter_error
          plan.add_patch(
            file: "SKILL.md",
            description: "Add parameter documentation",
            action: :append_section,
            content: generate_parameter_section(diagnosis)
          )

          if File.exist?(File.join(skill.source_path, "skill.yaml"))
            plan.add_patch(
              file: "skill.yaml",
              description: "Update entrypoint inputs",
              action: :update_yaml,
              path: "spec.entrypoints.0.inputs",
              content: infer_input_schema(diagnosis)
            )
          end

        when :file_error
          missing_file = diagnosis.error_message[/['"]([^'"]+)['"]/, 1]
          if missing_file && !missing_file.include?("/")
            plan.add_patch(
              file: "SKILL.md",
              description: "Add file creation note",
              action: :append_note,
              content: "Note: Ensure #{missing_file} exists before execution"
            )
          end

        when :template_error
          if diagnosis.error_message =~ /template[`'"]([^'"]+)['"`]/
            template_name = $1
            plan.add_patch(
              file: "SKILL.md",
              description: "Add template reference",
              action: :append_section,
              content: "## Templates\n\n- #{template_name}: Required template file"
            )
          end

        when :syntax_error
          plan.add_patch(
            file: diagnosis.error_location[:file] || "SKILL.md",
            description: "Fix syntax error",
            action: :mark_for_review,
            content: "Syntax error detected - manual fix required"
          )
        end

        plan
      end

      def apply_patches(skill, plan)
        applied = []

        plan.patches.each do |patch|
          break unless @budget.can_patch?(patch)

          file_path = File.join(skill.source_path, patch.file)

          case patch.action
          when :append_section
            if apply_append_section(file_path, patch.content)
              applied << patch
              @budget.record_hunk(patch.hunks)
            end

          when :append_note
            if apply_append_note(file_path, patch.content)
              applied << patch
              @budget.record_hunk(1)
            end

          when :update_yaml
            if apply_yaml_patch(file_path, patch.path, patch.content)
              applied << patch
              @budget.record_hunk(1)
            end

          when :mark_for_review
            applied << patch
            @budget.record_hunk(1)
          end
        end

        applied
      end

      def apply_append_section(file_path, content)
        return false unless File.exist?(file_path)

        File.open(file_path, "a") do |f|
          f.puts "\n\n#{content}"
        end
        true
      rescue
        false
      end

      def apply_append_note(file_path, content)
        return false unless File.exist?(file_path)

        File.open(file_path, "a") do |f|
          f.puts "\n\n> #{content}"
        end
        true
      rescue
        false
      end

      def apply_yaml_patch(file_path, path, content)
        return false unless File.exist?(file_path)

        yaml_content = YAML.load_file(file_path)
        keys = path.split(".")

        target = keys[0..-2].reduce(yaml_content) do |obj, key|
          obj[key] ||= {}
          obj[key]
        end

        target[keys.last] = content

        File.write(file_path, YAML.dump(yaml_content))
        true
      rescue
        false
      end

      def improved?(old_result, new_result, old_error_signature)
        return true if new_result.success? && !old_result.success?

        new_error_signature = error_signature(new_result.error)
        return true if new_error_signature != old_error_signature

        old_fatal = old_result.error.to_s.match?(/fatal|critical|error/i)
        new_fatal = new_result.error.to_s.match?(/fatal|critical|error/i)
        return true if old_fatal && !new_fatal

        old_metadata = old_result.metadata || {}
        new_metadata = new_result.metadata || {}

        return true if new_metadata[:task_progress].to_f > old_metadata[:task_progress].to_f

        false
      end

      def error_signature(error)
        return "" unless error
        error.to_s.downcase.gsub(/\d+/, "#").gsub(/['"][^'"]+['"]/, "'...'").strip[0..100]
      end

      def generate_parameter_section(diagnosis)
        <<~SECTION
          ## Input Parameters

          This skill accepts the following parameters:

          - `task`: The main task description (required)
          - `context`: Additional context information (optional)

          Please ensure all required parameters are provided when invoking this skill.
        SECTION
      end

      def infer_input_schema(diagnosis)
        {
          "task" => { "type" => "string", "required" => true },
          "context" => { "type" => "string", "required" => false }
        }
      end

      class RepairBudget
        attr_reader :attempts, :patched_files, :patched_hunks

        def initialize(max_attempts:, max_patched_files:, max_patched_hunks:)
          @max_attempts = max_attempts
          @max_patched_files = max_patched_files
          @max_patched_hunks = max_patched_hunks
          @attempts = 0
          @patched_files = 0
          @patched_hunks = 0
        end

        def has_budget?
          @attempts < @max_attempts &&
            @patched_files < @max_patched_files &&
            @patched_hunks < @max_patched_hunks
        end

        def can_patch?(patch)
          @patched_files < @max_patched_files &&
            (@patched_hunks + patch.hunks) <= @max_patched_hunks
        end

        def consume_attempt
          @attempts += 1
        end

        def consume_patch(files:, hunks:)
          @patched_files += files
          @patched_hunks += hunks
        end

        def record_hunk(hunks)
          @patched_hunks += hunks
        end
      end

      class Diagnosis
        attr_reader :error_type, :error_message, :error_location, :affected_files, :skill_path, :metadata

        def initialize(error_type:, error_message:, error_location:, affected_files:, skill_path:, metadata:)
          @error_type = error_type
          @error_message = error_message
          @error_location = error_location
          @affected_files = affected_files
          @skill_path = skill_path
          @metadata = metadata
        end

        def to_h
          {
            error_type: @error_type,
            error_message: @error_message,
            error_location: @error_location,
            affected_files: @affected_files,
            skill_path: @skill_path
          }
        end
      end

      class RepairPlan
        attr_reader :skill, :diagnosis, :patches

        def initialize(skill:, diagnosis:)
          @skill = skill
          @diagnosis = diagnosis
          @patches = []
        end

        def add_patch(file:, description:, action:, content: nil, path: nil)
          @patches << RepairPatch.new(
            file: file,
            description: description,
            action: action,
            content: content,
            path: path
          )
        end

        def valid?
          @patches.any? && @patches.all?(&:valid?)
        end

        def total_hunks
          @patches.sum(&:hunks)
        end

        def to_h
          {
            skill: @skill.name,
            diagnosis: @diagnosis.to_h,
            patches: @patches.map(&:to_h),
            total_hunks: total_hunks
          }
        end
      end

      class RepairPatch
        attr_reader :file, :description, :action, :content, :path

        def initialize(file:, description:, action:, content: nil, path: nil)
          @file = file
          @description = description
          @action = action
          @content = content
          @path = path
        end

        def valid?
          !!(@file && !@file.empty? && @action && @description)
        end

        def hunks
          @content ? @content.lines.count { |l| !l.strip.empty? } : 1
        end

        def to_h
          {
            file: @file,
            description: @description,
            action: @action,
            path: @path,
            content_preview: @content.to_s.lines.first(3).join.strip,
            hunks: hunks
          }
        end
      end
    end
  end
end
