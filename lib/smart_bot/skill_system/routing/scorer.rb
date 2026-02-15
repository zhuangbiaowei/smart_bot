# frozen_string_literal: true
require "set"

module SmartBot
  module SkillSystem
    # Scoring algorithm for skill candidates
    # Based on weights from skill_routing_spec.md
    class SkillScorer
      WEIGHTS = {
        intent_match: 0.40,
        trigger_match: 0.20,
        success_rate: 0.15,
        context_readiness: 0.10,
        cost_penalty: 0.10,
        conflict_penalty: 0.05
      }.freeze

      def score(candidate:, query:, context:, stats:)
        skill = candidate.skill

        scores = {
          intent_match: calculate_intent_match(skill, query),
          trigger_match: calculate_trigger_match(candidate, query),
          success_rate: calculate_success_rate(skill, stats),
          context_readiness: calculate_context_readiness(skill, context),
          cost_penalty: skill.metadata.cost_penalty,
          conflict_penalty: calculate_conflict_penalty(skill, query)
        }

        total = WEIGHTS.sum { |key, weight| weight * scores[key] }

        # Apply availability penalty
        total *= 0.5 unless skill.available?

        total.clamp(0.0, 1.0)
      end

      private

      def calculate_intent_match(skill, query)
        query_tokens = tokenize(query)
        return 0.5 if query_tokens.empty?

        # Route purely from SKILL metadata text and skill identity.
        skill_tokens = tokenize([
          skill.name,
          skill.description,
          skill.metadata.triggers.join(" "),
          skill.metadata.anti_triggers.join(" ")
        ].join(" "))

        return 0.5 if skill_tokens.empty?

        overlap = (query_tokens & skill_tokens).size
        union = (query_tokens | skill_tokens).size
        jaccard = union.zero? ? 0.0 : overlap.to_f / union

        # Keep neutral baseline for sparse text while allowing strong lexical matches.
        (0.30 + (jaccard * 0.70)).clamp(0.0, 1.0)
      end

      def calculate_trigger_match(candidate, query)
        base_score = case candidate.source
                     when :forced then 1.0
                     when :rule then 0.9
                     when :semantic then 0.75
                     else 0.5
                     end

        if candidate.source == :rule
          overlap_boost = trigger_overlap_boost(candidate.skill, query)
          return [base_score + overlap_boost, 0.98].min
        end

        if candidate.source == :semantic && candidate.instance_variable_defined?(:@semantic_score)
          semantic_score = candidate.instance_variable_get(:@semantic_score)
          # Boost semantic candidates more aggressively
          boosted = base_score + (semantic_score * 0.3)
          [boosted, 0.95].min
        else
          base_score
        end
      end

      def calculate_success_rate(skill, stats)
        skill_stats = stats[skill.name]
        return 0.5 unless skill_stats

        total = skill_stats[:total].to_i
        return 0.5 if total.zero?

        successes = skill_stats[:successes].to_i
        (successes.to_f / total).clamp(0.0, 1.0)
      end

      def calculate_context_readiness(skill, context)
        return 1.0 if context.empty?
        return 1.0 if skill.metadata.prerequisites.empty?

        satisfied = skill.metadata.prerequisites.count(&:satisfied?)
        total = skill.metadata.prerequisites.size

        satisfied.to_f / total
      end

      def calculate_conflict_penalty(skill, query)
        return -1.0 if skill.matches_anti_trigger?(query)

        0.0
      end

      def trigger_overlap_boost(skill, query)
        q = query.to_s.downcase
        matched = skill.metadata.triggers.select { |t| q.include?(t.to_s.downcase) }
        return 0.0 if matched.empty?

        # Prefer more specific (longer) matched triggers without hardcoding skill names.
        longest = matched.map { |m| m.to_s.length }.max.to_f
        size_boost = [longest / 40.0, 0.08].min
        count_boost = [matched.size * 0.02, 0.06].min

        size_boost + count_boost
      end

      def tokenize(text)
        text.to_s.downcase.scan(/[a-z0-9\u4e00-\u9fff]+/).to_set
      end
    end
  end
end
