# -*- encoding: utf-8 -*-
# stub: smart_prompt 0.3.4 ruby lib

Gem::Specification.new do |s|
  s.name = "smart_prompt".freeze
  s.version = "0.3.4".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.metadata = { "changelog_uri" => "https://github.com/zhuangbiaowei/smart_prompt/blob/master/CHANGELOG.md" } if s.respond_to? :metadata=
  s.require_paths = ["lib".freeze]
  s.authors = ["zhuang biaowei".freeze]
  s.bindir = "exe".freeze
  s.date = "1980-01-02"
  s.description = "SmartPrompt provides a flexible DSL for managing prompts, interacting with multiple LLMs, and creating composable task workers.".freeze
  s.email = ["zbw@kaiyuanshe.org".freeze]
  s.homepage = "https://github.com/zhuangbiaowei/smart_prompt".freeze
  s.licenses = ["MIT".freeze]
  s.required_ruby_version = Gem::Requirement.new(">= 3.0.0".freeze)
  s.rubygems_version = "3.7.1".freeze
  s.summary = "A smart prompt management and LLM interaction gem".freeze

  s.installed_by_version = "4.0.6".freeze

  s.specification_version = 4

  s.add_runtime_dependency(%q<yaml>.freeze, ["~> 0.4.0".freeze])
  s.add_runtime_dependency(%q<ruby-openai>.freeze, ["~> 8.1.0".freeze])
  s.add_runtime_dependency(%q<json>.freeze, ["~> 2.12.0".freeze])
  s.add_runtime_dependency(%q<safe_ruby>.freeze, ["~> 1.0.5".freeze])
  s.add_runtime_dependency(%q<retriable>.freeze, ["~> 3.1.2".freeze])
  s.add_runtime_dependency(%q<numo-narray>.freeze, ["~> 0.9.2.1".freeze])
  s.add_runtime_dependency(%q<better_prompt>.freeze, ["~> 0.2.1".freeze])
end
