# frozen_string_literal: true

require_relative "semantic_index"

module SmartBot
  module SkillSystem
    ScoredCandidate = Struct.new(:candidate, :score, keyword_init: true) do
      def skill
        candidate.skill
      end

      def source
        candidate.source
      end

      def forced?
        source == :forced
      end
    end

    SkillCandidate = Struct.new(:skill, :source, keyword_init: true)

    class Router
      DEFAULT_THRESHOLD = 0.45
      MAX_PARALLEL_SKILLS = 2

      attr_reader :registry, :scorer, :semantic_index

      def initialize(registry:, scorer: nil, enable_semantic: true)
        @registry = registry
        @scorer = scorer || SkillScorer.new
        @semantic_index = enable_semantic ? SemanticIndex.new : nil

        build_semantic_index if @semantic_index
      end

      def route(query:, context: {}, history: [], stats: {})
        candidates = recall_candidates(query, context)
        return empty_plan if candidates.empty?

        scored = score_candidates(candidates, query, context, stats)
        selected = select_candidates(scored)

        build_activation_plan(selected, query, context)
      end

      def refresh_semantic_index
        return unless @semantic_index

        @semantic_index.clear
        build_semantic_index
      end

      def semantic_stats
        @semantic_index&.stats || {}
      end

      private

      def build_semantic_index
        @registry.list_available.each do |skill|
          @semantic_index.add_skill(skill)
        end
        @semantic_index.rebuild_index
      end

      def recall_candidates(query, _context)
        candidates = []

        hard_trigger_candidates(query, candidates)
        rule_based_candidates(query, candidates)
        semantic_candidates(query, candidates)

        candidates.uniq { |c| c.skill.name }
      end

      def hard_trigger_candidates(query, candidates)
        patterns = [
          /\$(\w+)/,
          /使用\s+(\w+)\s+skill/i,
          /run_skill\s+(\w+)/i
        ]

        patterns.each do |pattern|
          if (match = query.match(pattern))
            skill_name = match[1].downcase
            skill = @registry.find(skill_name)
            if skill
              candidates << SkillCandidate.new(skill: skill, source: :forced)
            end
          end
        end
      end

      def rule_based_candidates(query, candidates)
        trigger_matches = @registry.find_by_trigger(query)

        trigger_matches.each do |skill|
          next if candidates.any? { |c| c.skill.name == skill.name }
          next if skill.matches_anti_trigger?(query)

          candidates << SkillCandidate.new(skill: skill, source: :rule)
        end
      end

      def semantic_candidates(query, candidates)
        return unless @semantic_index

        already_matched = candidates.map { |c| c.skill.name }.to_set

        matches = @semantic_index.search(query, top_k: 3, threshold: 0.15)

        matches.each do |skill_name, score|
          next if already_matched.include?(skill_name)

          skill = @registry.find(skill_name)
          next unless skill
          next if skill.matches_anti_trigger?(query)

          candidate = SkillCandidate.new(skill: skill, source: :semantic)
          candidate.instance_variable_set(:@semantic_score, score)
          candidates << candidate
        end
      end

      def score_candidates(candidates, query, context, stats)
        candidates.map do |candidate|
          score = @scorer.score(
            candidate: candidate,
            query: query,
            context: context,
            stats: stats
          )

          if candidate.source == :semantic && candidate.instance_variable_defined?(:@semantic_score)
            semantic_boost = candidate.instance_variable_get(:@semantic_score) * 0.2
            score += semantic_boost
          end

          ScoredCandidate.new(candidate: candidate, score: score)
        end
      end

      def select_candidates(scored)
        forced = scored.select(&:forced?)
        return forced if forced.any?

        regular = scored.select { |s| s.score >= DEFAULT_THRESHOLD }
        regular.sort_by(&:score).reverse
      end

      def build_activation_plan(selected, query, _context)
        return empty_plan if selected.empty?

        skills = selected.map(&:skill)
        primary_skill = skills.first

        ActivationPlan.new(
          skills: skills,
          parameters: { task: query },
          primary_skill: primary_skill,
          fallback_chain: build_fallback_chain(selected),
          parallel_groups: group_parallelizable(selected),
          estimated_cost: estimate_cost(selected)
        )
      end

      def empty_plan
        ActivationPlan.new(
          skills: [],
          parameters: {},
          primary_skill: nil,
          fallback_chain: [],
          parallel_groups: [],
          estimated_cost: 0
        )
      end

      def build_fallback_chain(selected)
        chain = selected.drop(1).map(&:skill)
        chain << :generic_tools
        chain
      end

      def group_parallelizable(selected)
        groups = []
        current_group = []

        selected.each do |scored|
          skill = scored.skill

          if skill.parallel_safe? && current_group.size < MAX_PARALLEL_SKILLS
            current_group << skill
          else
            groups << current_group unless current_group.empty?
            current_group = [skill]
          end
        end

        groups << current_group unless current_group.empty?
        groups
      end

      def estimate_cost(selected)
        selected.sum do |scored|
          cost_weight(scored.skill.metadata.cost_hint)
        end
      end

      def cost_weight(hint)
        case hint
        when :low then 1
        when :medium then 2
        when :high then 3
        else 2
        end
      end
    end
  end
end
