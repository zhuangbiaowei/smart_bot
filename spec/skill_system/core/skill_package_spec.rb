# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/core/skill_package"
require "smart_bot/skill_system/core/metadata"

RSpec.describe SmartBot::SkillSystem::SkillPackage do
  let(:metadata) do
    SmartBot::SkillSystem::SkillMetadata.new(
      name: "test_skill",
      description: "A test skill",
      triggers: ["test"],
      type: :instruction
    )
  end

  let(:package) do
    described_class.new(
      name: "test_skill",
      source_path: "/tmp/test",
      metadata: metadata,
      type: :instruction,
      content: "Test content"
    )
  end

  describe "#initialize" do
    it "normalizes name" do
      pkg = described_class.new(
        name: "Test-Skill_123",
        source_path: "/tmp",
        metadata: metadata,
        type: :instruction
      )
      expect(pkg.name).to eq("test_skill_123")
    end

    it "validates type" do
      expect {
        described_class.new(
          name: "test",
          source_path: "/tmp",
          metadata: metadata,
          type: :invalid_type
        )
      }.to raise_error(ArgumentError)
    end
  end

  describe "#matches_trigger?" do
    it "matches trigger in query" do
      expect(package.matches_trigger?("This is a test")).to be true
      expect(package.matches_trigger?("No match here")).to be false
    end

    it "is case insensitive" do
      expect(package.matches_trigger?("This is a TEST")).to be true
    end
  end

  describe "#matches_anti_trigger?" do
    let(:anti_metadata) do
      SmartBot::SkillSystem::SkillMetadata.new(
        name: "test",
        description: "desc",
        anti_triggers: ["exclude"]
      )
    end

    let(:anti_package) do
      described_class.new(
        name: "test",
        source_path: "/tmp",
        metadata: anti_metadata,
        type: :instruction
      )
    end

    it "matches anti-trigger" do
      expect(anti_package.matches_anti_trigger?("exclude this")).to be true
      expect(anti_package.matches_anti_trigger?("include this")).to be false
    end
  end

  describe "#available?" do
    it "delegates to metadata" do
      expect(package.available?).to be true
    end
  end

  describe "#to_h" do
    it "returns hash representation" do
      h = package.to_h
      expect(h[:name]).to eq("test_skill")
      expect(h[:type]).to eq(:instruction)
      expect(h[:available]).to be true
    end
  end
end
