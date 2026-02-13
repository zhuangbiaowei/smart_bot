# Copyright (c) 2018 Uku Taht
# frozen_string_literal: true

# main class
class SafeRuby
  MAJOR_VERSION = 1
  MINOR_VERSION = 0
  RELEASE_VERSION = 5

  private_constant :MAJOR_VERSION
  private_constant :MINOR_VERSION
  private_constant :RELEASE_VERSION

  VERSION = [MAJOR_VERSION, MINOR_VERSION, RELEASE_VERSION].join('.')
  public_constant :VERSION
end
