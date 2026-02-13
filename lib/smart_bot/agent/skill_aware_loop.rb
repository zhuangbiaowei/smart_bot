# frozen_string_literal: true

require_relative "../skill_system"

module SmartBot
  module Agent
    # Agent Loop with integrated Skill System
    # Routes user messages to appropriate skills automatically
    class SkillAwareLoop
      HIGH_CONFIDENCE_THRESHOLD = 0.80
      MEDIUM_CONFIDENCE_THRESHOLD = 0.65

      def initialize
        @skill_system = SmartBot::SkillSystem
        @conversation_history = []
        @execution_stats = {}
      end

      # Initialize and load all skills
      def setup
        say "🔄 Loading Skill System...", :cyan
        stats = @skill_system.load_all
        say "✅ Loaded #{stats[:total]} skills (#{stats[:available]} available)", :green

        router = @skill_system.router
        if router.semantic_index
          semantic_stats = router.semantic_stats
          say "📊 Semantic index: #{semantic_stats[:unique_terms]} unique terms", :blue
        end

        self
      end

      # Process a user message through the skill system
      def process(message, context: {}, llm_name: nil)
        return nil if message.nil? || message.strip.empty?

        # Check for explicit skill invocation first
        if (explicit = extract_explicit_skill(message))
          return execute_explicit_skill(explicit, message, context)
        end

        # Route through skill system
        plan = @skill_system.route(
          message,
          context: context,
          history: @conversation_history,
          stats: @execution_stats
        )

        if plan.empty?
          return { handled: false, reason: "no_matching_skills" }
        end

        # Check confidence level
        primary_score = calculate_primary_score(plan)

        if primary_score >= HIGH_CONFIDENCE_THRESHOLD
          execute_high_confidence(plan, message, context)
        elsif primary_score >= MEDIUM_CONFIDENCE_THRESHOLD
          execute_medium_confidence(plan, message, context)
        else
          suggest_skills(plan, message)
        end
      end

      # Check if this message should be handled by skills
      def should_handle?(message)
        return false if message.nil? || message.strip.empty?

        # Always handle explicit invocations
        return true if extract_explicit_skill(message)

        # Check if any skill matches
        plan = @skill_system.route(message, history: @conversation_history)
        return false if plan.empty?

        # Check if primary skill score is above minimum threshold
        primary_score = calculate_primary_score(plan)
        primary_score >= 0.50
      end

      # Get skill suggestions for a message
      def suggest(message, limit: 3)
        plan = @skill_system.route(message, history: @conversation_history)
        return [] if plan.empty?

        plan.skills.first(limit).map do |skill|
          {
            name: skill.name,
            description: skill.description,
            confidence: calculate_skill_confidence(skill, message)
          }
        end
      end

      # Record execution result for learning
      def record_result(skill_name, success)
        @execution_stats[skill_name] ||= { successes: 0, total: 0 }
        @execution_stats[skill_name][:total] += 1
        @execution_stats[skill_name][:successes] += 1 if success
      end

      # Add message to conversation history
      def add_to_history(role, message)
        @conversation_history << { role: role, message: message, timestamp: Time.now }
        @conversation_history = @conversation_history.last(10) # Keep last 10
      end

      # Refresh skill system (reload skills)
      def refresh
        say "🔄 Refreshing Skill System...", :cyan
        @skill_system.reset!
        stats = @skill_system.load_all
        say "✅ Refreshed: #{stats[:total]} skills", :green
        stats
      end

      # Get current skill system status
      def status
        {
          skills_loaded: @skill_system.registry.size,
          skills_available: @skill_system.registry.list_available.size,
          execution_stats: @execution_stats,
          conversation_length: @conversation_history.size
        }
      end

      private

      def extract_explicit_skill(message)
        patterns = [
          /^\$(\w+)/,
          /使用\s+(\w+)\s+skill/i,
          /run_skill\s+(\w+)/i
        ]

        patterns.each do |pattern|
          if match = message.match(pattern)
            skill_name = match[1].downcase
            skill = @skill_system.registry.find(skill_name)
            return skill_name if skill
          end
        end

        nil
      end

      def execute_explicit_skill(skill_name, message, context)
        skill = @skill_system.registry.find(skill_name)
        return { handled: false, error: "Skill not found" } unless skill

        say "🎯 Executing skill: #{skill_name}", :green

        result = @skill_system.run("$#{skill_name} #{message}", context: context)

        record_result(skill_name, result.success?)

        if result.success?
          { handled: true, result: result.value, skill: skill_name }
        else
          { handled: true, error: result.error, skill: skill_name }
        end
      end

      def execute_high_confidence(plan, message, context)
        skill = plan.primary_skill
        say "🎯 Auto-executing skill: #{skill.name} (confidence: #{(calculate_primary_score(plan) * 100).round(1)}%)", :green

        result = @skill_system.execute(plan, context: context)
        record_result(skill.name, result.success?)

        if result.success?
          { handled: true, result: result.value, skill: skill.name, auto: true }
        else
          { handled: true, error: result.error, skill: skill.name, auto: true }
        end
      end

      def execute_medium_confidence(plan, message, context)
        skill = plan.primary_skill
        say "🤔 Suggesting skill: #{skill.name} (confidence: #{(calculate_primary_score(plan) * 100).round(1)}%)", :yellow

        # Execute but mark as suggestion
        result = @skill_system.execute(plan, context: context)
        record_result(skill.name, result.success?)

        if result.success?
          { handled: true, result: result.value, skill: skill.name, suggested: true }
        else
          { handled: true, error: result.error, skill: skill.name, suggested: true }
        end
      end

      def suggest_skills(plan, message)
        suggestions = plan.skills.first(3).map do |skill|
          "• #{skill.name}: #{skill.description}"
        end

        say "🔍 Multiple skills could help:", :cyan
        suggestions.each { |s| say "  #{s}" }

        {
          handled: false,
          suggestions: plan.skills.first(3).map { |s| { name: s.name, description: s.description } },
          message: "Please specify which skill to use (e.g., '$skill_name your task')"
        }
      end

      def calculate_primary_score(plan)
        return 0.0 if plan.skills.empty?

        # Estimate based on plan composition
        base_score = 0.7

        # Boost if semantic match
        if plan.skills.first && plan.skills.first.metadata.triggers.any?
          base_score += 0.1
        end

        # Adjust based on fallback chain length
        if plan.fallback_chain.size <= 2
          base_score += 0.1
        end

        [base_score, 0.95].min
      end

      def calculate_skill_confidence(skill, message)
        # Simple confidence calculation
        return 0.0 unless skill.description

        desc_words = skill.description.downcase.split.to_set
        message_words = message.downcase.split.to_set

        overlap = (desc_words & message_words).size
        total = (desc_words | message_words).size

        return 0.0 if total.zero?

        (overlap.to_f / total * 0.8) + 0.1
      end

      def say(message, color = nil)
        if defined?(Thor::Shell::Basic)
          shell = Thor::Shell::Basic.new
          shell.say(message, color)
        else
          puts message
        end
      end
    end
  end
end
