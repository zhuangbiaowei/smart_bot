# frozen_string_literal: true

require "spec_helper"
require "smart_bot/agent/skill_aware_loop"
require "smart_bot/skill_system"

RSpec.describe SmartBot::Agent::SkillAwareLoop do
  let(:loop) { described_class.new }

  before do
    allow(loop).to receive(:say)
    SmartBot::SkillSystem.reset!
  end

  describe "#setup" do
    it "loads skills and displays stats" do
      expect(SmartBot::SkillSystem).to receive(:load_all).and_return(
        { total: 5, available: 4, always: 1, by_type: {} }
      )

      result = loop.setup
      expect(result).to eq(loop)
    end
  end

  describe "#process" do
    before do
      allow(SmartBot::SkillSystem).to receive(:load_all).and_return(
        { total: 2, available: 2, always: 0, by_type: {} }
      )
      loop.setup
    end

    it "returns nil for empty message" do
      expect(loop.process("")).to be_nil
      expect(loop.process(nil)).to be_nil
    end

    it "handles explicit skill invocation" do
      allow(SmartBot::SkillSystem).to receive(:run).and_return(
        SmartBot::SkillSystem::ExecutionResult.success(skill: double(name: "weather"), value: "Sunny")
      )

      result = loop.process("$weather Shanghai")

      expect(result[:handled]).to be true
      expect(result[:skill]).to eq("weather")
    end

    it "returns not handled when no skills match" do
      allow(SmartBot::SkillSystem).to receive(:route).and_return(
        SmartBot::SkillSystem::ActivationPlan.new(
          skills: [],
          parameters: {},
          primary_skill: nil,
          fallback_chain: [],
          parallel_groups: [],
          estimated_cost: 0
        )
      )

      result = loop.process("Some random query")
      expect(result[:handled]).to be false
      expect(result[:reason]).to eq("no_matching_skills")
    end
  end

  describe "#should_handle?" do
    before do
      allow(SmartBot::SkillSystem).to receive(:load_all).and_return(
        { total: 2, available: 2, always: 0, by_type: {} }
      )
      loop.setup
    end

    it "returns false for empty message" do
      expect(loop.should_handle?("")).to be false
    end

    it "returns true for explicit skill invocation" do
      expect(loop.should_handle?("$weather")).to be true
    end

    it "checks skill matching" do
      allow(SmartBot::SkillSystem).to receive(:route).and_return(
        double(empty?: false, skills: [double], primary_skill: double)
      )

      expect(loop.should_handle?("test query")).to be true
    end
  end

  describe "#suggest" do
    before do
      allow(SmartBot::SkillSystem).to receive(:load_all).and_return(
        { total: 2, available: 2, always: 0, by_type: {} }
      )
      loop.setup
    end

    it "returns suggestions for matching query" do
      skill = double(name: "weather", description: "Get weather info")
      allow(SmartBot::SkillSystem).to receive(:route).and_return(
        double(empty?: false, skills: [skill])
      )

      suggestions = loop.suggest("weather", limit: 1)

      expect(suggestions).not_to be_empty
      expect(suggestions.first[:name]).to eq("weather")
    end

    it "returns empty array when no matches" do
      allow(SmartBot::SkillSystem).to receive(:route).and_return(
        double(empty?: true)
      )

      expect(loop.suggest("test")).to be_empty
    end
  end

  describe "#record_result" do
    it "records successful execution" do
      loop.record_result("weather", true)
      stats = loop.status[:execution_stats]

      expect(stats["weather"][:successes]).to eq(1)
      expect(stats["weather"][:total]).to eq(1)
    end

    it "records failed execution" do
      loop.record_result("weather", false)
      stats = loop.status[:execution_stats]

      expect(stats["weather"][:successes]).to eq(0)
      expect(stats["weather"][:total]).to eq(1)
    end
  end

  describe "#add_to_history" do
    it "adds message to history" do
      loop.add_to_history("user", "Hello")
      expect(loop.status[:conversation_length]).to eq(1)
    end

    it "limits history to 10 messages" do
      12.times { |i| loop.add_to_history("user", "Message #{i}") }
      expect(loop.status[:conversation_length]).to eq(10)
    end
  end

  describe "#status" do
    it "returns current status" do
      allow(SmartBot::SkillSystem).to receive(:load_all).and_return(
        { total: 3, available: 2, always: 1, by_type: {} }
      )
      loop.setup

      status = loop.status

      expect(status[:skills_loaded]).to eq(3)
      expect(status[:skills_available]).to eq(2)
    end
  end
end
