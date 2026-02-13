# frozen_string_literal: true

require "spec_helper"
require "smart_bot/skill_system/execution/result"
require "smart_bot/skill_system/execution/openclaw_executor"

RSpec.describe SmartBot::SkillSystem::Execution::OpenClawExecutor do
  before do
    stub_const("SmartPrompt", Module.new) unless defined?(SmartPrompt)
  end

  describe "#execute" do
    it "returns normalized output payload for standard OpenClaw execution" do
      worker_context = Class.new do
        attr_accessor :used_llm

        def use(value)
          @used_llm = value
        end

        def sys_msg(_value); end

        def prompt(_value); end

        def send_msg; end
      end.new

      allow(SmartPrompt).to receive(:define_worker) do |_name, &block|
        worker_context.instance_eval(&block)
      end

      fake_engine = instance_double("SmartPrompt::Engine")
      allow(fake_engine).to receive(:call_worker).and_return("summary output")

      executor = described_class.new(llm_engine: fake_engine, llm_name: "deepseek")

      metadata = instance_double("SkillMetadata", description: "desc", openclaw_meta: {})
      skill = instance_double(
        "SkillPackage",
        name: "general_skill",
        content: "# Test Skill",
        metadata: metadata
      )

      result = executor.execute(skill, { task: "test task" }, { llm: "gemini" })

      expect(result.success?).to be(true)
      expect(result.value[:success]).to be(true)
      expect(result.value[:output]).to eq("summary output")
      expect(worker_context.used_llm).to eq("gemini")
    end

    it "fails clearly when youtube transcript is unavailable" do
      fake_engine = instance_double("SmartPrompt::Engine")
      executor = described_class.new(llm_engine: fake_engine, llm_name: "deepseek")

      metadata = instance_double("SkillMetadata", description: "desc", openclaw_meta: {})
      skill = instance_double(
        "SkillPackage",
        name: "youtube_summarizer",
        content: "# YouTube Skill",
        metadata: metadata
      )

      allow(executor).to receive(:command_available?).with("yt-dlp").and_return(true)
      allow(executor).to receive(:fetch_youtube_metadata).and_return({ "id" => "BRvFndSPX5M" })
      allow(executor).to receive(:fetch_youtube_transcript).and_return([nil, "no captions found"])

      result = executor.execute(
        skill,
        { task: "总结这个视频 https://www.youtube.com/watch?v=BRvFndSPX5M" },
        {}
      )

      expect(result.failure?).to be(true)
      expect(result.error).to include("Transcript not available")
      expect(result.error).to include("no captions found")
    end

    it "parses json3 subtitle content" do
      executor = described_class.new
      json3 = {
        "events" => [
          { "segs" => [{ "utf8" => "Hello " }, { "utf8" => "world" }] },
          { "segs" => [{ "utf8" => "Next line" }] }
        ]
      }.to_json

      parsed = executor.send(:normalize_json3_subtitle, json3)
      expect(parsed).to include("Hello world")
      expect(parsed).to include("Next line")
    end
  end
end
