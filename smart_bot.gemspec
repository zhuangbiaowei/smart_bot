# frozen_string_literal: true

require_relative "lib/smart_bot/version"

Gem::Specification.new do |spec|
  spec.name = "smart_bot"
  spec.version = SmartBot::VERSION
  spec.authors = ["Zhuang Biaowei"]
  spec.email = ["zbw@kaiyuanshe.org"]

  spec.summary = "Ultra-lightweight personal AI assistant in Ruby"
  spec.description = "A Ruby-based AI assistant framework inspired by nanobot"
  spec.homepage = "https://github.com/yourname/smart_bot"
  spec.license = "MIT"

  spec.files = Dir["lib/**/*", "LICENSE", "README.md", "bin/*"]
  spec.bindir = "bin"
  spec.executables = ["smart_bot"]
  spec.require_paths = ["lib"]

  spec.required_ruby_version = ">= 3.2.0"

  # Runtime dependencies
  spec.add_runtime_dependency "thor", "~> 1.3"
  spec.add_runtime_dependency "base64", "~> 0.2"
  spec.add_runtime_dependency "mime-types", "~> 3.0"
  # Note: smart_agent, smart_prompt, smart_rag, ruby_rich are loaded from local paths in Gemfile

  # Development dependencies
  spec.add_development_dependency "rspec", "~> 3.0"
  spec.add_development_dependency "rake", "~> 13.0"
end
