# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/execution/repair_loop"
require "smart_bot/skill_system/execution/executor"
require "smart_bot/skill_system/execution/result"
require "smart_bot/skill_system/core/value_objects"
require "smart_bot/skill_system/core/skill_package"
require "smart_bot/skill_system/core/metadata"

RSpec.describe SmartBot::SkillSystem::RepairLoop do
  let(:executor) { instance_double(SmartBot::SkillSystem::SkillExecutor) }
  let(:repair_loop) { described_class.new(executor: executor) }

  let(:metadata) do
    SmartBot::SkillSystem::SkillMetadata.new(
      name: "test_skill",
      description: "A test skill"
    )
  end

  let(:skill) do
    SmartBot::SkillSystem::SkillPackage.new(
      name: "test_skill",
      source_path: "/tmp/test_skill",
      metadata: metadata,
      type: :instruction
    )
  end

  describe "#execute_with_repair" do
    context "when skill succeeds on first attempt" do
      it "returns success result" do
        success_result = SmartBot::SkillSystem::ExecutionResult.success(
          skill: skill,
          value: "success"
        )

        allow(executor).to receive(:execute_skill).and_return(success_result)

        result = repair_loop.execute_with_repair(skill, {}, {})
        expect(result.success?).to be true
        expect(result.value).to eq("success")
      end
    end

    context "when skill fails with unrepairable error" do
      it "returns failure without repair attempts" do
        failure_result = SmartBot::SkillSystem::ExecutionResult.failure(
          skill: skill,
          error: "Permission denied"
        )

        allow(executor).to receive(:execute_skill).and_return(failure_result)

        result = repair_loop.execute_with_repair(skill, {}, {})
        expect(result.failure?).to be true
        expect(executor).to have_received(:execute_skill).once
      end
    end

    context "when skill fails with repairable error" do
      it "attempts repair and retry" do
        failure_result = SmartBot::SkillSystem::ExecutionResult.failure(
          skill: skill,
          error: "Missing parameter: task"
        )

        success_result = SmartBot::SkillSystem::ExecutionResult.success(
          skill: skill,
          value: "fixed"
        )

        call_count = 0
        allow(executor).to receive(:execute_skill) do
          call_count += 1
          call_count == 1 ? failure_result : success_result
        end

        # Allow the private method to apply patches (mock file operations)
        allow(repair_loop).to receive(:apply_patches).and_return([double("patch")])

        result = repair_loop.execute_with_repair(skill, {}, {})
        expect(result.success?).to be true
        expect(call_count).to be > 1
      end
    end
  end

  describe "#repairable?" do
    it "returns true for parameter errors" do
      result = SmartBot::SkillSystem::ExecutionResult.failure(
        skill: skill,
        error: "Missing required parameter"
      )
      expect(repair_loop.send(:repairable?, result)).to be true
    end

    it "returns true for file not found errors" do
      result = SmartBot::SkillSystem::ExecutionResult.failure(
        skill: skill,
        error: "No such file or directory"
      )
      expect(repair_loop.send(:repairable?, result)).to be true
    end

    it "returns false for permission errors" do
      result = SmartBot::SkillSystem::ExecutionResult.failure(
        skill: skill,
        error: "Permission denied"
      )
      # Permission denied is NOT in repairable patterns, so should be false
      expect(repair_loop.send(:repairable?, result)).to be false
    end

    it "returns false for nil skill" do
      result = SmartBot::SkillSystem::ExecutionResult.failure(
        skill: nil,
        error: "Some error"
      )
      expect(repair_loop.send(:repairable?, result)).to be false
    end

    it "returns true for template errors" do
      result = SmartBot::SkillSystem::ExecutionResult.failure(
        skill: skill,
        error: "Template not found"
      )
      expect(repair_loop.send(:repairable?, result)).to be true
    end
  end

  describe "#classify_error" do
    it "classifies parameter errors" do
      type = repair_loop.send(:classify_error, "Missing parameter: task")
      expect(type).to eq(:parameter_error)
    end

    it "classifies file errors" do
      type = repair_loop.send(:classify_error, "No such file or directory")
      expect(type).to eq(:file_error)
    end

    it "classifies template errors" do
      type = repair_loop.send(:classify_error, "Template not found")
      expect(type).to eq(:template_error)
    end

    it "classifies permission errors" do
      type = repair_loop.send(:classify_error, "Permission denied")
      expect(type).to eq(:permission_error)
    end

    it "classifies timeout errors" do
      type = repair_loop.send(:classify_error, "Execution timeout")
      expect(type).to eq(:timeout_error)
    end

    it "returns unknown for unrecognized errors" do
      type = repair_loop.send(:classify_error, "Something weird happened")
      expect(type).to eq(:unknown_error)
    end
  end

  describe "RepairBudget" do
    let(:budget) do
      SmartBot::SkillSystem::RepairLoop::RepairBudget.new(
        max_attempts: 2,
        max_patched_files: 3,
        max_patched_hunks: 8
      )
    end

    describe "#has_budget?" do
      it "returns true when within limits" do
        expect(budget.has_budget?).to be true
      end

      it "returns false when attempts exhausted" do
        2.times { budget.consume_attempt }
        expect(budget.has_budget?).to be false
      end
    end

    describe "#can_patch?" do
      it "returns true when within limits" do
        patch = double(hunks: 5)
        expect(budget.can_patch?(patch)).to be true
      end

      it "returns false when would exceed hunks limit" do
        patch = double(hunks: 10)
        expect(budget.can_patch?(patch)).to be false
      end
    end
  end

  describe "RepairPlan" do
    let(:diagnosis) do
      SmartBot::SkillSystem::RepairLoop::Diagnosis.new(
        error_type: :parameter_error,
        error_message: "Missing parameter",
        error_location: { file: nil, line: nil },
        affected_files: ["SKILL.md"],
        skill_path: "/tmp/test",
        metadata: {}
      )
    end

    let(:plan) do
      SmartBot::SkillSystem::RepairLoop::RepairPlan.new(
        skill: skill,
        diagnosis: diagnosis
      )
    end

    describe "#add_patch" do
      it "adds patches to the plan" do
        plan.add_patch(
          file: "SKILL.md",
          description: "Add parameters",
          action: :append_section,
          content: "Parameters section"
        )

        expect(plan.patches.size).to eq(1)
        expect(plan.patches.first.file).to eq("SKILL.md")
      end
    end

    describe "#valid?" do
      it "returns false for empty plan" do
        expect(plan.valid?).to be false
      end

      it "returns true with valid patches" do
        plan.add_patch(
          file: "SKILL.md",
          description: "Add parameters",
          action: :append_section,
          content: "Parameters"
        )
        expect(plan.valid?).to be true
      end
    end

    describe "#total_hunks" do
      it "calculates total hunks from patches" do
        plan.add_patch(
          file: "SKILL.md",
          description: "Add parameters",
          action: :append_section,
          content: "Line 1\nLine 2\nLine 3"
        )

        expect(plan.total_hunks).to eq(3)
      end
    end
  end

  describe "RepairPatch" do
    let(:patch) do
      SmartBot::SkillSystem::RepairLoop::RepairPatch.new(
        file: "SKILL.md",
        description: "Add section",
        action: :append_section,
        content: "Line 1\n\nLine 2"
      )
    end

    describe "#valid?" do
      it "returns true with required fields" do
        expect(patch.valid?).to be true
      end

      it "returns false without file" do
        invalid = SmartBot::SkillSystem::RepairLoop::RepairPatch.new(
          file: "",
          description: "test",
          action: :append
        )
        expect(invalid.valid?).to be false
      end
    end

    describe "#hunks" do
      it "counts non-empty lines" do
        expect(patch.hunks).to eq(2)
      end
    end
  end
end
