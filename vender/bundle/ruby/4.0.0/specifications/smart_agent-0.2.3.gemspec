# -*- encoding: utf-8 -*-
# stub: smart_agent 0.2.3 ruby lib

Gem::Specification.new do |s|
  s.name = "smart_agent".freeze
  s.version = "0.2.3".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["Zhuang Biaowei".freeze]
  s.date = "1980-01-02"
  s.description = "Build AI agents with declarative DSL and Model Context Protocol support".freeze
  s.email = ["zbw@kaiyuanshe.org".freeze]
  s.homepage = "https://github.com/yourname/smart_agent".freeze
  s.licenses = ["MIT".freeze]
  s.required_ruby_version = Gem::Requirement.new(">= 3.2.0".freeze)
  s.rubygems_version = "3.7.1".freeze
  s.summary = "Intelligent agent framework with DSL and MCP integration".freeze

  s.installed_by_version = "4.0.6".freeze

  s.specification_version = 4

  s.add_runtime_dependency(%q<smart_prompt>.freeze, [">= 0".freeze])
  s.add_runtime_dependency(%q<mcp-sdk.rb>.freeze, [">= 0".freeze])
  s.add_development_dependency(%q<rspec>.freeze, ["~> 3.0".freeze])
end
