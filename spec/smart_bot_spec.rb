# frozen_string_literal: true

require "spec_helper"

RSpec.describe SmartBot do
  it "has a version number" do
    expect(SmartBot::VERSION).not_to be nil
  end

  describe SmartBot::Config do
    it "creates default configuration" do
      config = SmartBot::Config.new
      expect(config.model).to eq("anthropic/claude-opus-4-5")
      expect(config.max_tokens).to eq(8192)
    end
  end

  describe SmartBot::Bus::Queue do
    it "publishes and consumes messages" do
      bus = SmartBot::Bus::Queue.new
      msg = SmartBot::Bus::InboundMessage.new(
        channel: "test",
        sender_id: "user",
        chat_id: "123",
        content: "Hello"
      )
      
      bus.publish_inbound(msg)
      expect(bus.inbound_size).to eq(1)
    end
  end

  describe SmartBot::Tools::Registry do
    it "registers and executes tools" do
      registry = SmartBot::Tools::Registry.new
      tool = SmartBot::Tools::ReadFileTool.new
      
      registry.register(tool)
      expect(registry.get(:read_file)).to eq(tool)
      expect(registry.tool_names).to include(:read_file)
    end
  end
end
