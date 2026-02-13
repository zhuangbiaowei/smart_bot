# -*- encoding: utf-8 -*-
# stub: mcp-sdk.rb 0.1.4 ruby lib

Gem::Specification.new do |s|
  s.name = "mcp-sdk.rb".freeze
  s.version = "0.1.4".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["Zhuang Biaowei".freeze]
  s.date = "1980-01-02"
  s.description = "A Ruby SDK for the MCP (Model Context Protocol) implementation".freeze
  s.email = ["zbw@kaiyuanshe.org".freeze]
  s.homepage = "https://github.com/zhuangbiaowei/mcp-sdk.rb".freeze
  s.licenses = ["MIT".freeze]
  s.required_ruby_version = Gem::Requirement.new(">= 3.0.0".freeze)
  s.rubygems_version = "3.7.0".freeze
  s.summary = "MCP SDK for Ruby".freeze

  s.installed_by_version = "4.0.6".freeze

  s.specification_version = 4

  s.add_runtime_dependency(%q<json>.freeze, [">= 2.7.1".freeze])
  s.add_runtime_dependency(%q<faraday>.freeze, [">= 2.0.0".freeze])
  s.add_development_dependency(%q<bundler>.freeze, [">= 2.0".freeze])
  s.add_development_dependency(%q<rake>.freeze, [">= 13.0".freeze])
  s.add_development_dependency(%q<rspec>.freeze, [">= 3.0".freeze])
  s.add_development_dependency(%q<rubocop>.freeze, [">= 1.0".freeze])
end
