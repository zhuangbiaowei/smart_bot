# frozen_string_literal: true

require "set"

module SmartBot
  module SkillSystem
    # Semantic search using TF-IDF and cosine similarity
    # Lightweight alternative to embedding-based search
    class SemanticIndex
      attr_reader :skills, :term_document_matrix, :idf, :skill_vectors

      def initialize
        @skills = {}
        @corpus_tokens = {}
        @document_frequency = Hash.new(0)
        @total_documents = 0
        @skill_vectors = {}
        @idf = {}
        @dirty = false
      end

      # Add a skill to the index
      def add_skill(skill)
        return unless skill.is_a?(SkillPackage)
        return if skill.description.nil? || skill.description.empty?

        @skills[skill.name] = skill
        text = build_search_text(skill)
        tokens = tokenize(text)

        @corpus_tokens[skill.name] = tokens
        update_document_frequency(tokens)
        @total_documents += 1
        @dirty = true

        skill
      end

      # Remove a skill from the index
      def remove_skill(skill_name)
        return unless @skills.key?(skill_name)

        @skills.delete(skill_name)
        tokens = @corpus_tokens.delete(skill_name)

        if tokens
          tokens.uniq.each { |token| @document_frequency[token] -= 1 }
          @total_documents -= 1
        end

        @dirty = true
      end

      # Search for skills similar to query
      # Returns array of [skill_name, similarity_score] sorted by score
      def search(query, top_k: 3, threshold: 0.1)
        rebuild_vectors if @dirty

        return [] if @skills.empty?

        query_tokens = tokenize(query)
        query_vector = vectorize(query_tokens)

        results = @skills.map do |name, _skill|
          skill_vector = @skill_vectors[name]
          similarity = cosine_similarity(query_vector, skill_vector)
          [name, similarity]
        end

        results
          .select { |_, score| score >= threshold }
          .sort_by { |_, score| -score }
          .first(top_k)
      end

      # Get similarity score between query and specific skill
      def similarity(query, skill_name)
        rebuild_vectors if @dirty

        return 0.0 unless @skills.key?(skill_name)

        query_tokens = tokenize(query)
        query_vector = vectorize(query_tokens)
        skill_vector = @skill_vectors[skill_name]

        cosine_similarity(query_vector, skill_vector)
      end

      # Rebuild the entire index (useful after batch updates)
      def rebuild_index
        @document_frequency = Hash.new(0)
        @total_documents = @skills.size

        @corpus_tokens.each_value do |tokens|
          tokens.uniq.each { |token| @document_frequency[token] += 1 }
        end

        @dirty = true
        rebuild_vectors
      end

      # Clear all indexed skills
      def clear
        @skills.clear
        @corpus_tokens.clear
        @document_frequency.clear
        @skill_vectors.clear
        @idf.clear
        @total_documents = 0
        @dirty = false
      end

      # Statistics
      def stats
        {
          skills_indexed: @skills.size,
          unique_terms: @document_frequency.size,
          avg_terms_per_skill: @total_documents > 0 ? @corpus_tokens.values.sum(&:size).to_f / @total_documents : 0
        }
      end

      private

      def build_search_text(skill)
        parts = []

        # Name (high weight by repeating)
        3.times { parts << skill.name }

        # Description
        parts << skill.description.to_s

        # Triggers
        skill.metadata.triggers.each { |t| parts << t }

        # Entrypoint names and descriptions
        skill.metadata.entrypoints.each do |ep|
          parts << ep.name.to_s
        end

        parts.join(" ")
      end

      def tokenize(text)
        return [] if text.nil? || text.empty?

        # Normalize and split
        text.downcase
            .gsub(/[^\w\s]/, " ")  # Remove punctuation
            .split(/\s+/)            # Split on whitespace
            .reject { |w| w.empty? || stopword?(w) }
            .map { |w| stem(w) }
      end

      def stopword?(word)
        STOPWORDS.include?(word)
      end

      def stem(word)
        # Simple stemming - remove common suffixes
        # For production, consider using ruby-stemmer gem
        word
          .gsub(/(ing|ed|s|es)$/, "")
          .gsub(/(tion|ness|ment)$/, "")
      end

      def update_document_frequency(tokens)
        tokens.uniq.each { |token| @document_frequency[token] += 1 }
      end

      def rebuild_vectors
        return unless @dirty

        # Calculate IDF for all terms
        # Use smoothed IDF: log((N + 1) / (df + 1)) + 1 to avoid zero values
        @idf = {}
        @document_frequency.each do |term, df|
          # Smoothed IDF that handles single-document case
          @idf[term] = Math.log((@total_documents + 1).to_f / (df + 1)) + 1.0
        end

        # Build TF-IDF vectors for each skill
        @skill_vectors = {}
        @corpus_tokens.each do |name, tokens|
          @skill_vectors[name] = vectorize(tokens)
        end

        @dirty = false
      end

      def vectorize(tokens)
        return {} if tokens.empty?

        # Calculate term frequency
        tf = Hash.new(0.0)
        tokens.each { |token| tf[token] += 1.0 }

        # Normalize TF (term frequency / max frequency)
        max_freq = tf.values.max
        tf.transform_values! { |v| v / max_freq }

        # Calculate TF-IDF
        vector = {}
        tf.each do |term, freq|
          idf = @idf[term] || 0.0
          vector[term] = freq * idf
        end

        vector
      end

      def cosine_similarity(vec1, vec2)
        return 0.0 if vec1.empty? || vec2.empty?

        # Get all unique terms
        all_terms = (vec1.keys + vec2.keys).uniq

        # Calculate dot product and magnitudes
        dot_product = 0.0
        magnitude1 = 0.0
        magnitude2 = 0.0

        all_terms.each do |term|
          v1 = vec1[term] || 0.0
          v2 = vec2[term] || 0.0

          dot_product += v1 * v2
          magnitude1 += v1 * v1
          magnitude2 += v2 * v2
        end

        magnitude1 = Math.sqrt(magnitude1)
        magnitude2 = Math.sqrt(magnitude2)

        return 0.0 if magnitude1.zero? || magnitude2.zero?

        dot_product / (magnitude1 * magnitude2)
      end

      # Common English stopwords
      STOPWORDS = Set.new(%w[
        a an and are as at be by for from has he in is it its of on that the to
        was will with i you they we she him her them their this these those
        have had do does did can could would should may might must shall
        about above across after against along among around at before behind
        below beneath beside between beyond but despite down during except
        following inside into like near off onto outside over past since
        through throughout till toward under until up upon within without
        的 是 在 和 了 有 我 他 她 它 你 这 那 个 上 下 中 就 都 而 及 与 或 等
      ]).freeze
    end
  end
end
