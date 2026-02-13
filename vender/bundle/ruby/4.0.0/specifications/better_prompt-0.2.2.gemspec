# -*- encoding: utf-8 -*-
# stub: better_prompt 0.2.2 ruby lib

Gem::Specification.new do |s|
  s.name = "better_prompt".freeze
  s.version = "0.2.2".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["zhuang biaowei".freeze]
  s.date = "1980-01-02"
  s.description = "Provides enhanced command line prompt functionality with database storage".freeze
  s.email = ["zbw@kaiyuanshe.org".freeze]
  s.executables = ["bp".freeze]
  s.files = ["bin/bp".freeze]
  s.homepage = "https://github.com/zhuangbiaowei/better_prompt".freeze
  s.licenses = ["MIT".freeze]
  s.required_ruby_version = Gem::Requirement.new(">= 2.6.0".freeze)
  s.rubygems_version = "4.0.0".freeze
  s.summary = "A better command line prompt utility".freeze

  s.installed_by_version = "4.0.6".freeze

  s.specification_version = 4

  s.add_runtime_dependency(%q<ruby_rich>.freeze, ["~> 0.4.0".freeze])
  s.add_runtime_dependency(%q<sqlite3>.freeze, ["~> 1.4".freeze])
  s.add_development_dependency(%q<bundler>.freeze, ["~> 2.0".freeze])
  s.add_development_dependency(%q<rake>.freeze, ["~> 13.0".freeze])
  s.add_development_dependency(%q<rspec>.freeze, ["~> 3.0".freeze])
end
