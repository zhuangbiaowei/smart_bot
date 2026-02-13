# Copyright (c) 2023 Jerome Arbez-Gindre
# frozen_string_literal: true

require('rspec/core/rake_task')

namespace 'test' do
  RSpec::Core::RakeTask.new(:spec) do |t|
    t.rspec_opts = ['--options config/rspec']
  end

  desc 'Runs all unit tests and acceptance tests'
  task(all: ['test:spec'])
end

desc 'Synonym for test:spec'
task(spec: 'test:spec')

desc 'Synonym for test:all'
task(test: 'test:all')
