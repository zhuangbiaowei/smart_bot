# -*- encoding: utf-8 -*-
# stub: ruby_rich 0.4.0 ruby lib

Gem::Specification.new do |s|
  s.name = "ruby_rich".freeze
  s.version = "0.4.0".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["zhuang biaowei".freeze]
  s.date = "1980-01-02"
  s.description = "A Ruby gem providing rich text formatting, progress bars, tables and other console output enhancements".freeze
  s.email = "zbw@kaiyuanshe.org".freeze
  s.homepage = "https://github.com/zhuangbiaowei/ruby_rich".freeze
  s.licenses = ["MIT".freeze]
  s.required_ruby_version = Gem::Requirement.new(">= 2.7.0".freeze)
  s.rubygems_version = "3.6.9".freeze
  s.summary = "Rich text formatting and console output for Ruby".freeze

  s.installed_by_version = "4.0.6".freeze

  s.specification_version = 4

  s.add_development_dependency(%q<rake>.freeze, ["~> 13.0".freeze])
  s.add_development_dependency(%q<minitest>.freeze, ["~> 5.0".freeze])
  s.add_runtime_dependency(%q<rouge>.freeze, ["~> 4.5.2".freeze])
  s.add_runtime_dependency(%q<unicode-display_width>.freeze, ["~> 3.1.4".freeze])
  s.add_runtime_dependency(%q<tty-cursor>.freeze, ["~> 0.7.1".freeze])
  s.add_runtime_dependency(%q<tty-screen>.freeze, ["~> 0.8.2".freeze])
  s.add_runtime_dependency(%q<tty-reader>.freeze, ["~> 0.9.0".freeze])
  s.add_runtime_dependency(%q<redcarpet>.freeze, ["~> 3.6.1".freeze])
end
