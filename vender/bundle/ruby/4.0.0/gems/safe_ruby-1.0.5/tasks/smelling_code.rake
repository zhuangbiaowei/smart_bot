# Copyright (c) 2023 Jerome Arbez-Gindre
# frozen_string_literal: true

namespace 'quality' do
  begin
    require('rubocop/rake_task')

    RuboCop::RakeTask.new do |task|
      task.options << '--display-cop-names'
      task.options << '--config=config/rubocop.yml'
    end
  rescue LoadError
    task(:rubocop) do
      puts('Install rubocop to run its rake tasks')
    end
  end

  desc 'Runs all quality code check'
  task(all: ['quality:rubocop'])
end

desc 'Synonym for quality:rubocop'
task(rubocop: 'quality:rubocop')
