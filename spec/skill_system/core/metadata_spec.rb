# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/core/metadata"
require "smart_bot/skill_system/core/value_objects"

RSpec.describe SmartBot::SkillSystem::SkillMetadata do
  let(:valid_attrs) do
    {
      name: "test_skill",
      description: "A test skill",
      version: "1.0.0",
      triggers: ["test", "example"],
      cost_hint: :low
    }
  end

  describe "#initialize" do
    it "creates with required attributes" do
      metadata = described_class.new(valid_attrs)
      expect(metadata.name).to eq("test_skill")
      expect(metadata.description).to eq("A test skill")
      expect(metadata.version).to eq("1.0.0")
    end

    it "has defaults" do
      metadata = described_class.new(name: "test", description: "desc")
      expect(metadata.version).to eq("0.1.0")
      expect(metadata.author).to eq("Unknown")
      expect(metadata.cost_hint).to eq(:medium)
    end
  end

  describe "#cost_penalty" do
    it "returns correct penalty for each cost hint" do
      low = described_class.new(valid_attrs.merge(cost_hint: :low))
      expect(low.cost_penalty).to eq(0.0)

      medium = described_class.new(valid_attrs.merge(cost_hint: :medium))
      expect(medium.cost_penalty).to eq(-0.05)

      high = described_class.new(valid_attrs.merge(cost_hint: :high))
      expect(high.cost_penalty).to eq(-0.10)
    end
  end

  describe "#available?" do
    it "returns true when no prerequisites" do
      metadata = described_class.new(valid_attrs)
      expect(metadata.available?).to be true
    end

    it "checks prerequisites" do
      metadata = described_class.new(
        valid_attrs.merge(
          prerequisites: {
            "system" => ["nonexistent_binary_12345"]
          }
        )
      )
      expect(metadata.available?).to be false
    end
  end

  describe ".from_skill_yaml" do
    let(:yaml_hash) do
      {
        "metadata" => {
          "name" => "yaml_skill",
          "description" => "From YAML",
          "version" => "2.0.0"
        },
        "spec" => {
          "triggers" => ["yaml", "test"],
          "cost_hint" => "high",
          "execution" => {
            "sandbox" => "container",
            "timeout" => 60
          }
        }
      }
    end

    it "parses YAML hash correctly" do
      metadata = described_class.from_skill_yaml(yaml_hash)
      expect(metadata.name).to eq("yaml_skill")
      expect(metadata.cost_hint).to eq(:high)
      expect(metadata.execution_policy.sandbox).to eq(:container)
    end

    it "returns nil for invalid input" do
      expect(described_class.from_skill_yaml(nil)).to be_nil
    end
  end

  describe ".from_frontmatter" do
    let(:frontmatter) do
      {
        "name" => "fm_skill",
        "description" => "From frontmatter",
        "triggers" => ["fm"]
      }
    end

    it "parses frontmatter correctly" do
      metadata = described_class.from_frontmatter(frontmatter)
      expect(metadata.name).to eq("fm_skill")
      expect(metadata.triggers).to eq(["fm"])
    end

    it "uses fallback description" do
      fm = frontmatter.except("description")
      metadata = described_class.from_frontmatter(fm, "Fallback desc")
      expect(metadata.description).to eq("Fallback desc")
    end
  end
end
