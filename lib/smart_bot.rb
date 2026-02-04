# frozen_string_literal: true

require "logger"
require "json"
require "pathname"
require "fileutils"
require "date"
require "thor"

require_relative "smart_bot/version"
require_relative "smart_bot/config"
require_relative "smart_bot/utils/helpers"
require_relative "smart_bot/bus/events"
require_relative "smart_bot/bus/queue"
require_relative "smart_bot/session/manager"
require_relative "smart_bot/providers/base"
require_relative "smart_bot/providers/openrouter"
require_relative "smart_bot/tools/base"
require_relative "smart_bot/tools/registry"
require_relative "smart_bot/tools/filesystem"
require_relative "smart_bot/tools/shell"
require_relative "smart_bot/tools/web"
require_relative "smart_bot/tools/message"
require_relative "smart_bot/tools/spawn"
require_relative "smart_bot/agent/context"
require_relative "smart_bot/agent/loop"
require_relative "smart_bot/agent/subagent"
require_relative "smart_bot/cron/types"
require_relative "smart_bot/cron/service"
require_relative "smart_bot/heartbeat/service"
require_relative "smart_bot/channels/base"
require_relative "smart_bot/channels/manager"
require_relative "smart_bot/cli/commands"

module SmartBot
  class Error < StandardError; end
  class ConfigurationError < Error; end
  class APIError < Error; end

  LOGO = "🤖"
  
  class << self
    attr_accessor :logger, :config

    def logger
      @logger ||= Logger.new($stdout).tap do |log|
        log.progname = "SmartBot"
        log.level = Logger::INFO
      end
    end

    def config
      @config ||= Config.load
    end

    def config=(config)
      @config = config
    end
  end
end
