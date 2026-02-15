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

  describe "ruby skill invocation" do
    let(:metadata) do
      SmartBot::SkillSystem::SkillMetadata.new(
        name: "weather",
        description: "weather skill",
        triggers: ["weather", "天气"]
      )
    end

    let(:skill) do
      SmartBot::SkillSystem::SkillPackage.new(
        name: "weather",
        source_path: "/tmp/weather",
        metadata: metadata,
        type: :ruby_native
      )
    end

    it "uses registered tool instance instead of SmartAgent::Tool.call class method" do
      definition = instance_double("SkillDefinition", tools: [{ name: :get_weather }])
      allow(SmartBot::Skill).to receive(:find).with(:weather).and_return(definition)

      tool_context = instance_double("ToolContext", params: { location: {}, unit: {} })
      tool = instance_double("Tool", context: tool_context)
      allow(tool).to receive(:call).with(hash_including("location")).and_return({ location: "Shanghai", temperature: "20C" })

      allow(SmartAgent::Tool).to receive(:find_tool).with(:weather_agent).and_return(nil)
      allow(SmartAgent::Tool).to receive(:find_tool).with("weather_agent").and_return(nil)
      allow(SmartAgent::Tool).to receive(:find_tool).with(:get_weather).and_return(tool)

      result = sandbox.send(:invoke_ruby_skill, skill, { "task" => "Today shanghai weather?" })
      expect(result.success?).to be true
      expect(result.value).to include(:location)
    end

    it "returns clear failure when no skill tool is available" do
      definition = instance_double("SkillDefinition", tools: [])
      allow(SmartBot::Skill).to receive(:find).with(:weather).and_return(definition)
      allow(SmartAgent::Tool).to receive(:find_tool).and_return(nil)

      result = sandbox.send(:invoke_ruby_skill, skill, { "task" => "weather shanghai" })
      expect(result.failure?).to be true
      expect(result.error).to include("No executable tool found")
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
