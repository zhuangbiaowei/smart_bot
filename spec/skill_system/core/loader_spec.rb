# frozen_string_literal: true

require "spec_helper"
require "tmpdir"
require "smart_bot/skill_system"

RSpec.describe SmartBot::SkillSystem::UnifiedLoader do
  describe "#load_from_path" do
    it "loads plain SKILL.md frontmatter as openclaw_instruction" do
      Dir.mktmpdir("loader_spec_") do |tmpdir|
        skill_dir = File.join(tmpdir, "plain_skill")
        Dir.mkdir(skill_dir)

        File.write(
          File.join(skill_dir, "SKILL.md"),
          <<~MD
            ---
            name: plain_skill
            version: 1.0.0
            description: plain markdown instruction skill
            ---

            # Plain Skill

            Do something useful.
          MD
        )

        loader = described_class.new(workspace: tmpdir, repo_root: tmpdir, home: tmpdir)
        skill = loader.load_from_path(skill_dir)

        expect(skill).not_to be_nil
        expect(skill.type).to eq(:openclaw_instruction)
        expect(skill.name).to eq("plain_skill")
        expect(skill.original_format).to eq(:markdown)
      end
    end

    it "preserves explicit triggers and anti_triggers from frontmatter" do
      Dir.mktmpdir("loader_spec_") do |tmpdir|
        skill_dir = File.join(tmpdir, "trigger_skill")
        Dir.mkdir(skill_dir)

        File.write(
          File.join(skill_dir, "SKILL.md"),
          <<~MD
            ---
            name: trigger_skill
            description: test skill
            triggers:
              - trigger_skill
            anti_triggers:
              - block_me
            cost_hint: low
            ---

            # Trigger Skill

            Instruction content.
          MD
        )

        loader = described_class.new(workspace: tmpdir, repo_root: tmpdir, home: tmpdir)
        skill = loader.load_from_path(skill_dir)

        expect(skill).not_to be_nil
        expect(skill.type).to eq(:openclaw_instruction)
        expect(skill.metadata.triggers).to eq(["trigger_skill"])
        expect(skill.metadata.anti_triggers).to eq(["block_me"])
        expect(skill.metadata.cost_hint).to eq(:low)
      end
    end

    it "loads OpenClaw metadata skill as openclaw_instruction with openclaw format" do
      Dir.mktmpdir("loader_spec_") do |tmpdir|
        skill_dir = File.join(tmpdir, "oc_skill")
        Dir.mkdir(skill_dir)

        File.write(
          File.join(skill_dir, "SKILL.md"),
          <<~MD
            ---
            name: oc_skill
            description: openclaw skill
            metadata:
              openclaw:
                emoji: "pkg"
                skillKey: "oc_skill"
            ---

            # OC Skill

            Instructions.
          MD
        )

        loader = described_class.new(workspace: tmpdir, repo_root: tmpdir, home: tmpdir)
        skill = loader.load_from_path(skill_dir)

        expect(skill).not_to be_nil
        expect(skill.type).to eq(:openclaw_instruction)
        expect(skill.original_format).to eq(:openclaw)
      end
    end
  end
end
