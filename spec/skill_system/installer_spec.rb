# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/installer"

RSpec.describe SmartBot::SkillSystem::SkillInstaller do
  let(:temp_dir) { Dir.mktmpdir }
  let(:installer) { described_class.new(target_dir: temp_dir) }

  after do
    FileUtils.rm_rf(temp_dir)
  end

  describe "#detect_source_type" do
    it "detects GitHub shorthand" do
      type = installer.send(:detect_source_type, "user/repo")
      expect(type).to eq(:github)
    end

    it "detects Git URL" do
      expect(installer.send(:detect_source_type, "https://github.com/user/repo.git")).to eq(:git)
      expect(installer.send(:detect_source_type, "git@github.com:user/repo.git")).to eq(:git)
    end

    it "detects local path" do
      Dir.mktmpdir do |dir|
        expect(installer.send(:detect_source_type, dir)).to eq(:local)
      end
    end

    it "detects NPM package" do
      expect(installer.send(:detect_source_type, "npm:package-name")).to eq(:npm)
    end

    it "detects PyPI package" do
      expect(installer.send(:detect_source_type, "pypi:package-name")).to eq(:pypi)
      expect(installer.send(:detect_source_type, "pip:package-name")).to eq(:pypi)
    end

    it "detects URL" do
      expect(installer.send(:detect_source_type, "https://example.com/skill.zip")).to eq(:url)
    end
  end

  describe "#install_from_local" do
    let(:skill_dir) { File.join(temp_dir, "test_skill") }

    before do
      FileUtils.mkdir_p(skill_dir)
      File.write(File.join(skill_dir, "SKILL.md"), "# Test Skill")
    end

    it "installs from local directory" do
      target = File.join(temp_dir, "installed")
      installer = described_class.new(target_dir: target)

      result = installer.install_from_local(skill_dir, name: "my_skill")

      expect(result.success?).to be true
      expect(File.exist?(File.join(target, "my_skill", "SKILL.md"))).to be true
    end

    it "fails if not a directory" do
      result = installer.install_from_local("/nonexistent/path")
      expect(result.success?).to be false
    end
  end

  describe "#validate_skill_structure" do
    it "passes with SKILL.md" do
      Dir.mktmpdir do |dir|
        File.write(File.join(dir, "SKILL.md"), "# Test")
        expect { installer.send(:validate_skill_structure, dir) }.not_to raise_error
      end
    end

    it "passes with skill.yaml" do
      Dir.mktmpdir do |dir|
        File.write(File.join(dir, "skill.yaml"), "name: test")
        expect { installer.send(:validate_skill_structure, dir) }.not_to raise_error
      end
    end

    it "fails without required files" do
      Dir.mktmpdir do |dir|
        expect { installer.send(:validate_skill_structure, dir) }.to raise_error(ArgumentError)
      end
    end
  end

  describe "#list_installed" do
    it "returns empty array when no skills" do
      expect(installer.list_installed).to be_empty
    end

    it "lists installed skills" do
      skill_dir = File.join(temp_dir, "test_skill")
      FileUtils.mkdir_p(skill_dir)
      File.write(File.join(skill_dir, "SKILL.md"), "# Test")

      skills = installer.list_installed
      expect(skills.size).to eq(1)
      expect(skills.first[:name]).to eq("test_skill")
    end
  end

  describe "#uninstall" do
    it "removes installed skill" do
      skill_dir = File.join(temp_dir, "test_skill")
      FileUtils.mkdir_p(skill_dir)

      result = installer.uninstall("test_skill")
      expect(result.success?).to be true
      expect(File.exist?(skill_dir)).to be false
    end

    it "fails if skill not found" do
      result = installer.uninstall("nonexistent")
      expect(result.success?).to be false
    end
  end
end
