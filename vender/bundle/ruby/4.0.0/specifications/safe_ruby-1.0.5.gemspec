# -*- encoding: utf-8 -*-
# stub: safe_ruby 1.0.5 ruby lib

Gem::Specification.new do |s|
  s.name = "safe_ruby".freeze
  s.version = "1.0.5".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.metadata = { "rubygems_mfa_required" => "true" } if s.respond_to? :metadata=
  s.require_paths = ["lib".freeze]
  s.authors = ["J\u00E9r\u00F4me Arbez-Gindre".freeze, "Uku Taht".freeze]
  s.date = "2024-10-04"
  s.description = "Evaluates ruby code by writing it to a tempfile and spawning a child process. Uses a allowlist of methods and constants to keep, for example one cannot run system commands in the environment created by this gem. The environment created by the untrusted code does not leak out into the parent process.".freeze
  s.email = "jeromearbezgindre@gmail.com".freeze
  s.homepage = "https://gitlab.com/defmastership/safe_ruby/".freeze
  s.licenses = ["MIT".freeze]
  s.required_ruby_version = Gem::Requirement.new(">= 2.7".freeze)
  s.rubygems_version = "3.5.16".freeze
  s.summary = "Run untrusted ruby code in a safe environment".freeze

  s.installed_by_version = "4.0.6".freeze

  s.specification_version = 4

  s.add_runtime_dependency(%q<childprocess>.freeze, ["~> 5".freeze])
end
