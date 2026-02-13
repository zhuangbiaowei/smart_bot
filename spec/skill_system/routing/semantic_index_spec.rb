# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/routing/semantic_index"
require "smart_bot/skill_system/core/skill_package"
require "smart_bot/skill_system/core/metadata"

RSpec.describe SmartBot::SkillSystem::SemanticIndex do
  let(:index) { described_class.new }

  let(:weather_metadata) do
    SmartBot::SkillSystem::SkillMetadata.new(
      name: "weather",
      description: "Get weather information and forecasts for any location",
      triggers: ["weather", "forecast", "temperature"]
    )
  end

  let(:weather_skill) do
    SmartBot::SkillSystem::SkillPackage.new(
      name: "weather",
      source_path: "/tmp/weather",
      metadata: weather_metadata,
      type: :instruction
    )
  end

  let(:search_metadata) do
    SmartBot::SkillSystem::SkillMetadata.new(
      name: "search",
      description: "Search the web for information and news",
      triggers: ["search", "find", "lookup"]
    )
  end

  let(:search_skill) do
    SmartBot::SkillSystem::SkillPackage.new(
      name: "search",
      source_path: "/tmp/search",
      metadata: search_metadata,
      type: :instruction
    )
  end

  describe "#add_skill" do
    it "adds a skill to the index" do
      index.add_skill(weather_skill)
      expect(index.skills).to include("weather" => weather_skill)
    end

    it "ignores skills without description" do
      metadata = SmartBot::SkillSystem::SkillMetadata.new(
        name: "empty",
        description: ""
      )
      skill = SmartBot::SkillSystem::SkillPackage.new(
        name: "empty",
        source_path: "/tmp",
        metadata: metadata,
        type: :instruction
      )

      index.add_skill(skill)
      expect(index.skills).not_to include("empty")
    end
  end

  describe "#search" do
    before do
      index.add_skill(weather_skill)
      index.add_skill(search_skill)
      index.rebuild_index
    end

    it "returns matching skills sorted by relevance" do
      results = index.search("What's the weather like today?")

      expect(results).not_to be_empty
      expect(results.first[0]).to eq("weather")
      expect(results.first[1]).to be > 0.0
    end

    it "returns top_k results" do
      results = index.search("search for information", top_k: 1, threshold: 0.05)
      expect(results.size).to eq(1)
    end

    it "filters by threshold" do
      results = index.search("completely unrelated query xyz123", threshold: 0.5)
      expect(results).to be_empty
    end

    it "matches related concepts" do
      results = index.search("temperature forecast")

      weather_score = results.find { |name, _| name == "weather" }
      expect(weather_score).not_to be_nil
      expect(weather_score[1]).to be > 0.1
    end
  end

  describe "#similarity" do
    before do
      index.add_skill(weather_skill)
      index.rebuild_index
    end

    it "returns similarity score for matching query" do
      # Use words that are definitely in the skill's indexed content
      score = index.similarity("weather information forecast location", "weather")
      expect(score).to be > 0.0
    end

    it "returns 0 for unknown skill" do
      score = index.similarity("weather", "unknown")
      expect(score).to eq(0.0)
    end
  end

  describe "#remove_skill" do
    before do
      index.add_skill(weather_skill)
      index.rebuild_index
    end

    it "removes skill from index" do
      index.remove_skill("weather")
      expect(index.skills).not_to include("weather")
    end
  end

  describe "#clear" do
    before do
      index.add_skill(weather_skill)
      index.rebuild_index
    end

    it "clears all indexed skills" do
      index.clear
      expect(index.skills).to be_empty
      expect(index.stats[:skills_indexed]).to eq(0)
    end
  end

  describe "#stats" do
    before do
      index.add_skill(weather_skill)
      index.add_skill(search_skill)
      index.rebuild_index
    end

    it "returns statistics" do
      stats = index.stats
      expect(stats[:skills_indexed]).to eq(2)
      expect(stats[:unique_terms]).to be > 0
    end
  end

  describe "tokenization" do
    it "normalizes text" do
      index.add_skill(weather_skill)
      index.rebuild_index

      tokens = index.instance_variable_get(:@corpus_tokens)["weather"]
      expect(tokens).to include("weather")
      expect(tokens).to include("forecast")
    end

    it "removes stopwords" do
      index.add_skill(weather_skill)
      index.rebuild_index

      tokens = index.instance_variable_get(:@corpus_tokens)["weather"]
      expect(tokens).not_to include("the")
      expect(tokens).not_to include("and")
    end
  end

  describe "cosine similarity" do
    let(:vec1) { { "a" => 1.0, "b" => 0.5 } }
    let(:vec2) { { "a" => 0.8, "b" => 0.4 } }
    let(:vec3) { { "c" => 1.0 } }

    it "calculates similarity for similar vectors" do
      similarity = index.send(:cosine_similarity, vec1, vec2)
      expect(similarity).to be > 0.9
      expect(similarity).to be <= 1.0
    end

    it "returns 0 for orthogonal vectors" do
      similarity = index.send(:cosine_similarity, vec1, vec3)
      expect(similarity).to eq(0.0)
    end

    it "returns 0 for empty vectors" do
      similarity = index.send(:cosine_similarity, {}, vec1)
      expect(similarity).to eq(0.0)
    end
  end
end
