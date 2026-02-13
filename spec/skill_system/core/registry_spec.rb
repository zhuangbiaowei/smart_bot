# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/core/registry"
require "smart_bot/skill_system/core/skill_package"
require "smart_bot/skill_system/core/metadata"

RSpec.describe SmartBot::SkillSystem::SkillRegistry do
  let(:metadata) do
    SmartBot::SkillSystem::SkillMetadata.new(
      name: "test_skill",
      description: "A test skill",
      triggers: ["test", "example"]
    )
  end

  let(:package) do
    SmartBot::SkillSystem::SkillPackage.new(
      name: "test_skill",
      source_path: "/tmp/test",
      metadata: metadata,
      type: :instruction
    )
  end

  let(:registry) { described_class.new }

  describe "#register" do
    it "registers a skill" do
      registry.register(package)
      expect(registry.find("test_skill")).to eq(package)
    end

    it "raises error for non-SkillPackage" do
      expect { registry.register("not a package") }.to raise_error(ArgumentError)
    end
  end

  describe "#find" do
    before { registry.register(package) }

    it "finds by name" do
      expect(registry.find("test_skill")).to eq(package)
    end

    it "normalizes name" do
      expect(registry.find("Test-Skill")).to eq(package)
    end

    it "returns nil for unknown" do
      expect(registry.find("unknown")).to be_nil
    end
  end

  describe "#find_by_trigger" do
    before { registry.register(package) }

    it "finds skills matching trigger" do
      matches = registry.find_by_trigger("This is a test")
      expect(matches).to include(package)
    end

    it "is case insensitive" do
      matches = registry.find_by_trigger("This is a TEST")
      expect(matches).to include(package)
    end

    it "returns empty array for no matches" do
      matches = registry.find_by_trigger("No matching words")
      expect(matches).to be_empty
    end
  end

  describe "#list_available" do
    it "returns available skills" do
      registry.register(package)
      expect(registry.list_available).to include(package)
    end
  end

  describe "#stats" do
    before { registry.register(package) }

    it "returns statistics" do
      stats = registry.stats
      expect(stats[:total]).to eq(1)
      expect(stats[:available]).to eq(1)
    end
  end

  describe "#each" do
    before { registry.register(package) }

    it "iterates over skills" do
      names = []
      registry.each { |s| names << s.name }
      expect(names).to eq(["test_skill"])
    end
  end

  describe "#clear" do
    before { registry.register(package) }

    it "removes all skills" do
      registry.clear
      expect(registry.empty?).to be true
    end
  end
end

RSpec.describe SmartBot::SkillSystem::SkillIndex do
  let(:index) { described_class.new }

  let(:metadata) do
    SmartBot::SkillSystem::SkillMetadata.new(
      name: "test",
      description: "desc",
      triggers: ["test", "example"]
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

  describe "#add" do
    it "adds skill to index" do
      index.add(package)
      expect(index.find_by_trigger("test")).to include(package)
    end
  end

  describe "#remove" do
    before { index.add(package) }

    it "removes skill from index" do
      index.remove(package)
      expect(index.find_by_trigger("test")).to be_empty
    end
  end

  describe "#find_by_trigger" do
    before { index.add(package) }

    it "finds by trigger" do
      expect(index.find_by_trigger("this is a test")).to include(package)
    end

    it "returns unique results" do
      # Add another skill with same trigger
      metadata2 = SmartBot::SkillSystem::SkillMetadata.new(
        name: "test2",
        description: "desc",
        triggers: ["test"]
      )
      package2 = SmartBot::SkillSystem::SkillPackage.new(
        name: "test2",
        source_path: "/tmp2",
        metadata: metadata2,
        type: :instruction
      )
      index.add(package2)

      results = index.find_by_trigger("test")
      expect(results.uniq.size).to eq(results.size)
    end
  end
end
