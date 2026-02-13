# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/execution/result"
require "smart_bot/skill_system/execution/sandbox"
require "smart_bot/skill_system/execution/executor"
require "smart_bot/skill_system/execution/fallback"
require "smart_bot/skill_system/routing/activation_plan"
require "smart_bot/skill_system/core/value_objects"
require "smart_bot/skill_system/core/skill_package"
require "smart_bot/skill_system/core/metadata"

RSpec.describe SmartBot::SkillSystem::ExecutionResult do
  describe ".success" do
    it "creates successful result" do
      skill = double(name: "test")
      result = described_class.success(skill: skill, value: "output")
      expect(result.success?).to be true
      expect(result.value).to eq("output")
    end
  end

  describe ".failure" do
    it "creates failed result" do
      skill = double(name: "test")
      result = described_class.failure(skill: skill, error: "Something went wrong")
      expect(result.failure?).to be true
      expect(result.error).to eq("Something went wrong")
    end
  end

  describe "#to_h" do
    it "returns hash representation" do
      skill = double(name: "test")
      result = described_class.success(skill: skill, value: "output", metadata: { key: "value" })
      h = result.to_h
      expect(h[:skill]).to eq("test")
      expect(h[:success]).to be true
      expect(h[:value]).to eq("output")
    end
  end
end

RSpec.describe SmartBot::SkillSystem::Sandbox do
  let(:sandbox) { described_class.new }

  describe "#check_permissions" do
    it "returns true for empty permissions" do
      perms = SmartBot::SkillSystem::PermissionSet.new
      expect(sandbox.check_permissions(perms)).to be true
    end

    it "checks read permissions" do
      perms = SmartBot::SkillSystem::PermissionSet.new(
        filesystem: { read: ["/etc"] }
      )
      expect(sandbox.check_permissions(perms)).to be true
    end

    it "checks env permissions" do
      perms = SmartBot::SkillSystem::PermissionSet.new(
        environment: { allow: ["HOME"] }
      )
      expect(sandbox.check_permissions(perms)).to be true

      perms2 = SmartBot::SkillSystem::PermissionSet.new(
        environment: { allow: ["NONEXISTENT_12345"] }
      )
      expect(sandbox.check_permissions(perms2)).to be false
    end
  end
end

RSpec.describe SmartBot::SkillSystem::SkillExecutor do
  let(:executor) { described_class.new }

  let(:metadata) do
    SmartBot::SkillSystem::SkillMetadata.new(
      name: "test",
      description: "A test skill"
    )
  end

  let(:package) do
    SmartBot::SkillSystem::SkillPackage.new(
      name: "test",
      source_path: "/tmp",
      metadata: metadata,
      type: :instruction,
      content: "Test instructions"
    )
  end

  describe "#execute_skill" do
    it "returns failure when permissions check fails" do
      # Create a skill with impossible permissions
      restricted_metadata = SmartBot::SkillSystem::SkillMetadata.new(
        name: "restricted",
        description: "desc",
        permissions: {
          "environment" => { "allow" => ["NONEXISTENT_VAR"] }
        }
      )
      restricted_package = SmartBot::SkillSystem::SkillPackage.new(
        name: "restricted",
        source_path: "/tmp",
        metadata: restricted_metadata,
        type: :instruction
      )

      result = executor.execute_skill(restricted_package, {})
      expect(result.failure?).to be true
      expect(result.error).to include("Permission")
    end
  end
end

RSpec.describe SmartBot::SkillSystem::FallbackStateMachine do
  let(:skills) do
    [
      instance_double(SmartBot::SkillSystem::SkillPackage, name: "primary"),
      instance_double(SmartBot::SkillSystem::SkillPackage, name: "fallback")
    ]
  end

  let(:plan) do
    SmartBot::SkillSystem::ActivationPlan.new(
      skills: skills,
      parameters: { task: "test" },
      primary_skill: skills.first,
      fallback_chain: [skills.last, :generic_tools],
      parallel_groups: [[skills.first]],
      estimated_cost: 2
    )
  end

  let(:executor) { instance_double(SmartBot::SkillSystem::SkillExecutor) }

  let(:fsm) do
    described_class.new(plan: plan, executor: executor)
  end

  describe "#run" do
    it "transitions through states" do
      success_result = SmartBot::SkillSystem::ExecutionResult.success(
        skill: skills.first,
        value: "success"
      )

      allow(executor).to receive(:execute_skill).and_return(success_result)

      result = fsm.run
      expect(result.success?).to be true
    end

    it "handles failures and uses fallback" do
      failure_result = SmartBot::SkillSystem::ExecutionResult.failure(
        skill: skills.first,
        error: "Failed"
      )

      allow(executor).to receive(:execute_skill).and_return(failure_result)

      result = fsm.run
      expect(result).not_to be_nil
    end
  end

  describe "state transitions" do
    it "starts in selected state" do
      expect(fsm.state).to eq(:selected)
    end
  end
end
