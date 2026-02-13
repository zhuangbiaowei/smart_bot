# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/routing/router"
require "smart_bot/skill_system/routing/scorer"
require "smart_bot/skill_system/routing/activation_plan"
require "smart_bot/skill_system/core/value_objects"
require "smart_bot/skill_system/core/registry"
require "smart_bot/skill_system/core/skill_package"
require "smart_bot/skill_system/core/metadata"

RSpec.describe SmartBot::SkillSystem::Router do
  let(:registry) { SmartBot::SkillSystem::SkillRegistry.new }

  let(:weather_metadata) do
    SmartBot::SkillSystem::SkillMetadata.new(
      name: "weather",
      description: "Get weather information",
      triggers: ["weather", "天气"]
    )
  end

  let(:weather_package) do
    SmartBot::SkillSystem::SkillPackage.new(
      name: "weather",
      source_path: "/tmp/weather",
      metadata: weather_metadata,
      type: :instruction
    )
  end

  let(:router) { described_class.new(registry: registry) }

  before do
    registry.register(weather_package)
  end

  describe "#route" do
    it "returns empty plan for no matches" do
      plan = router.route(query: "No matching query")
      expect(plan.empty?).to be true
    end

    it "matches by trigger" do
      plan = router.route(query: "What's the weather today?")
      expect(plan.skills).to include(weather_package)
    end

    it "matches by hard trigger" do
      plan = router.route(query: "$weather check")
      expect(plan.skills.first).to eq(weather_package)
    end

    it "sets primary skill" do
      plan = router.route(query: "weather forecast")
      expect(plan.primary_skill).to eq(weather_package)
    end

    it "includes fallback chain" do
      plan = router.route(query: "weather")
      expect(plan.fallback_chain).to include(:generic_tools)
    end

    it "prefers youtube_downloader for youtube download intent" do
      downloader_metadata = SmartBot::SkillSystem::SkillMetadata.new(
        name: "youtube_downloader",
        description: "Download YouTube videos",
        triggers: ["download youtube", "下载视频"]
      )
      downloader = SmartBot::SkillSystem::SkillPackage.new(
        name: "youtube_downloader",
        source_path: "/tmp/youtube_downloader",
        metadata: downloader_metadata,
        type: :script
      )

      summarizer_metadata = SmartBot::SkillSystem::SkillMetadata.new(
        name: "youtube_summarizer",
        description: "Summarize YouTube videos",
        triggers: ["youtube", "视频"]
      )
      summarizer = SmartBot::SkillSystem::SkillPackage.new(
        name: "youtube_summarizer",
        source_path: "/tmp/youtube_summarizer",
        metadata: summarizer_metadata,
        type: :openclaw_instruction
      )

      registry.register(downloader)
      registry.register(summarizer)

      plan = router.route(query: "下载YouTube视频：https://www.youtube.com/watch?v=BRvFndSPX5M")
      expect(plan.primary_skill.name).to eq("youtube_downloader")
    end

    it "penalizes skills that explicitly say not for downloading" do
      downloader_metadata = SmartBot::SkillSystem::SkillMetadata.new(
        name: "youtube_downloader",
        description: "Download YouTube videos with quality options",
        triggers: ["download", "下载youtube", "下载视频"]
      )
      downloader = SmartBot::SkillSystem::SkillPackage.new(
        name: "youtube_downloader",
        source_path: "/tmp/youtube_downloader",
        metadata: downloader_metadata,
        type: :script
      )

      summarizer_metadata = SmartBot::SkillSystem::SkillMetadata.new(
        name: "youtube_summarizer",
        description: "This skill is for summarizing and transcribing videos, NOT for downloading them.",
        triggers: ["youtube", "download", "下载", "summarize", "总结"]
      )
      summarizer = SmartBot::SkillSystem::SkillPackage.new(
        name: "youtube_summarizer",
        source_path: "/tmp/youtube_summarizer",
        metadata: summarizer_metadata,
        type: :openclaw_instruction
      )

      registry.register(downloader)
      registry.register(summarizer)

      plan = router.route(query: "下载YouTube视频：https://www.youtube.com/watch?v=BRvFndSPX5M")
      expect(plan.primary_skill.name).to eq("youtube_downloader")
    end
  end

  describe "hard trigger patterns" do
    ["$weather", "使用 weather skill", "run_skill weather: task"].each do |pattern|
      it "matches '#{pattern}'" do
        plan = router.route(query: "#{pattern} check")
        expect(plan.skills).not_to be_empty
      end
    end
  end
end

RSpec.describe SmartBot::SkillSystem::SkillScorer do
  let(:scorer) { described_class.new }

  let(:metadata) do
    SmartBot::SkillSystem::SkillMetadata.new(
      name: "test",
      description: "A test skill for testing",
      triggers: ["test"]
    )
  end

  let(:package) do
    SmartBot::SkillSystem::SkillPackage.new(
      name: "test",
      source_path: "/tmp",
      metadata: metadata,
      type: :instruction
    )
  end

  let(:candidate) do
    SmartBot::SkillSystem::SkillCandidate.new(
      skill: package,
      source: :rule
    )
  end

  describe "#score" do
    it "returns score between 0 and 1" do
      score = scorer.score(
        candidate: candidate,
        query: "test query",
        context: {},
        stats: {}
      )
      expect(score).to be_between(0.0, 1.0)
    end

    it "penalizes unavailable skills" do
      unavailable_metadata = SmartBot::SkillSystem::SkillMetadata.new(
        name: "unavailable",
        description: "desc",
        prerequisites: { "system" => ["nonexistent_binary"] }
      )
      unavailable_package = SmartBot::SkillSystem::SkillPackage.new(
        name: "unavailable",
        source_path: "/tmp",
        metadata: unavailable_metadata,
        type: :instruction
      )
      unavailable_candidate = SmartBot::SkillSystem::SkillCandidate.new(
        skill: unavailable_package,
        source: :rule
      )

      score = scorer.score(
        candidate: unavailable_candidate,
        query: "test",
        context: {},
        stats: {}
      )

      normal_score = scorer.score(
        candidate: candidate,
        query: "test",
        context: {},
        stats: {}
      )

      expect(score).to be < normal_score
    end
  end
end

RSpec.describe SmartBot::SkillSystem::ActivationPlan do
  let(:skills) do
    [
      instance_double(SmartBot::SkillSystem::SkillPackage, name: "skill1"),
      instance_double(SmartBot::SkillSystem::SkillPackage, name: "skill2")
    ]
  end

  let(:plan) do
    described_class.new(
      skills: skills,
      parameters: { task: "test" },
      primary_skill: skills.first,
      fallback_chain: [skills.last, :generic_tools],
      parallel_groups: [[skills.first], [skills.last]],
      estimated_cost: 3
    )
  end

  describe "#empty?" do
    it "returns false when skills present" do
      expect(plan.empty?).to be false
    end

    it "returns true when no skills" do
      empty_plan = described_class.new(
        skills: [],
        parameters: {},
        primary_skill: nil,
        fallback_chain: [],
        parallel_groups: [],
        estimated_cost: 0
      )
      expect(empty_plan.empty?).to be true
    end
  end

  describe "#parallelizable?" do
    it "returns true when multiple groups" do
      expect(plan.parallelizable?).to be true
    end
  end

  describe "#to_h" do
    it "returns hash representation" do
      h = plan.to_h
      expect(h[:skills]).to eq(["skill1", "skill2"])
      expect(h[:primary]).to eq("skill1")
      expect(h[:estimated_cost]).to eq(3)
    end
  end
end
