# Copyright (c) 2023 Jerome Arbez-Gindre
# frozen_string_literal: true

desc 'Starts the interactive console'
task :console do
  require 'pry'
  Pry.start
end
