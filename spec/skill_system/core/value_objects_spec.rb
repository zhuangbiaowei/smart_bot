# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/core/value_objects"

RSpec.describe SmartBot::SkillSystem::PermissionSet do
  describe "#initialize" do
    it "creates with default values" do
      perms = described_class.new
      expect(perms.filesystem[:read]).to eq([])
      expect(perms.filesystem[:write]).to eq([])
      expect(perms.network[:outbound]).to be false
    end

    it "creates with custom values" do
      perms = described_class.new(
        filesystem: { read: ["/tmp"], write: ["/tmp/out"] },
        network: { outbound: true }
      )
      expect(perms.filesystem[:read]).to eq(["/tmp"])
      expect(perms.can_read?("/tmp/test")).to be true
      expect(perms.can_write?("/tmp/out/file")).to be true
    end
  end

  describe "#can_read?" do
    it "allows any path when read list is empty" do
      perms = described_class.new
      expect(perms.can_read?("/any/path")).to be true
    end

    it "checks path prefix" do
      perms = described_class.new(filesystem: { read: ["/allowed"] })
      expect(perms.can_read?("/allowed/file")).to be true
      expect(perms.can_read?("/other/file")).to be false
    end
  end
end

RSpec.describe SmartBot::SkillSystem::ExecutionPolicy do
  describe "#initialize" do
    it "has defaults" do
      policy = described_class.new
      expect(policy.sandbox).to eq(:process)
      expect(policy.approval).to eq(:ask)
      expect(policy.timeout).to eq(120)
    end

    it "can be customized" do
      policy = described_class.new(sandbox: :container, approval: :auto, timeout: 60)
      expect(policy.sandbox).to eq(:container)
      expect(policy.auto?).to be true
    end
  end
end

RSpec.describe SmartBot::SkillSystem::Prerequisite do
  describe "#satisfied?" do
    it "checks for bin" do
      prereq = described_class.new(type: :bin, name: "ruby")
      expect(prereq.satisfied?).to be true
    end

    it "checks for env var" do
      prereq = described_class.new(type: :env, name: "HOME")
      expect(prereq.satisfied?).to be true

      prereq2 = described_class.new(type: :env, name: "NONEXISTENT_VAR_12345")
      expect(prereq2.satisfied?).to be false
    end

    it "checks for file" do
      prereq = described_class.new(type: :file, name: "/etc/passwd")
      expect(prereq.satisfied?).to be true

      prereq2 = described_class.new(type: :file, name: "/nonexistent/file")
      expect(prereq2.satisfied?).to be false
    end
  end
end
