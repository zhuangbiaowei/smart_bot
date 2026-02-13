# frozen_string_literal: true

module SmartBot
  module SkillSystem
    # Scoring algorithm for skill candidates
    # Based on weights from skill_routing_spec.md
    class SkillScorer
      INTENT_KEYWORDS = {
        download: %w[download 下载 保存 grab 抓取 下载器],
        summarize: %w[summarize summary transcript 总结 摘要 转录],
        search: %w[search find 查找 搜索 查询],
        weather: %w[weather 天气 预报 forecast],
        code: %w[code 编码 编程 script 脚本]
      }.freeze

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
        return 0.5 if skill.description.nil?

        # Baseline overlap from description
        desc_words = tokenize(skill.description)
        query_words = query.downcase.split.to_set

        return 0.5 if desc_words.empty? || query_words.empty?

        overlap = (desc_words & query_words).size
        total = (desc_words | query_words).size
        base = (overlap.to_f / total * 0.5) + 0.25

        # Generic intent affinity boost/penalty from metadata text.
        query_intents = detect_intents(query)
        return base if query_intents.empty?

        skill_text = [
          skill.name,
          skill.description,
          skill.metadata.triggers.join(" "),
          skill.metadata.anti_triggers.join(" ")
        ].join(" ").downcase

        aligned = query_intents.count do |intent|
          keywords = INTENT_KEYWORDS[intent]
          keywords.any? { |k| skill_text.include?(k) }
        end

        anti_aligned = query_intents.count do |intent|
          keywords = INTENT_KEYWORDS[intent]
          skill.metadata.anti_triggers.any? do |anti|
            anti_str = anti.to_s.downcase
            keywords.any? { |k| anti_str.include?(k) }
          end
        end

        boosted = base + (aligned * 0.08) - (anti_aligned * 0.12)
        boosted -= negative_intent_penalty(skill, query_intents)
        boosted.clamp(0.0, 1.0)
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

      def detect_intents(query)
        q = query.to_s.downcase
        INTENT_KEYWORDS.filter_map do |intent, keywords|
          intent if keywords.any? { |kw| q.include?(kw) }
        end
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

      def negative_intent_penalty(skill, query_intents)
        return 0.0 if query_intents.empty?

        text = [skill.description, skill.metadata.triggers.join(" ")].join(" ").downcase
        penalty = 0.0

        if query_intents.include?(:download)
          # Generic negation patterns that indicate "this skill is not for downloading".
          penalty += 0.25 if text.match?(/not\s+for\s+download|not\s+for\s+downloading|不.*下载|不是.*下载|非.*下载/)
        end

        if query_intents.include?(:summarize)
          # Symmetric rule for summary/transcript intent.
          penalty += 0.25 if text.match?(/not\s+for\s+summar|not\s+for\s+transcript|不.*总结|不是.*总结|非.*总结|不.*转录/)
        end

        penalty
      end
    end
  end
end
