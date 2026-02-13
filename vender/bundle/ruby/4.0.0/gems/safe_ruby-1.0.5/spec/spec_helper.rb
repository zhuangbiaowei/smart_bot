# Copyright (c) 2018 Uku Taht
# frozen_string_literal: true

require 'benchmark'
require 'safe_ruby'

RSpec.configure do |config|
  config.run_all_when_everything_filtered = true
  config.filter_run(:focus)

  config.order = 'random'
end
