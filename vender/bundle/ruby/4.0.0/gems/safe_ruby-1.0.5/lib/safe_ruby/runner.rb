# Copyright (c) 2018 Uku Taht
# frozen_string_literal: true

require('childprocess')
require('tempfile')

class EvalError < StandardError
end

# main class
class SafeRuby
  DEFAULTS = {
    timeout: 5,
    raise_errors: true
  }.freeze
  private_constant :DEFAULTS

  # rubocop:disable Style/OptionHash
  def self.eval(code, options = {})
    new(code, options).eval
  end
  # rubocop:enable Style/OptionHash

  def self.check(code, expected)
    # rubocop:disable Security/Eval
    eval(code) == eval(expected)
    # rubocop:enable Security/Eval
  end

  # rubocop:disable Style/OptionHash
  def initialize(code, options = {})
    options = DEFAULTS.merge(options)

    @code         = code
    @raise_errors = options[:raise_errors]
    @timeout      = options[:timeout]
  end
  # rubocop:enable Style/OptionHash

  # rubocop:disable Metrics/AbcSize
  # rubocop:disable Metrics/MethodLength
  def eval
    temp = build_tempfile
    read, write = IO.pipe
    ChildProcess.build('ruby', temp.path).tap do |process|
      process.io.stdout = write
      process.io.stderr = write
      process.start
      begin
        process.poll_for_exit(@timeout)
      rescue ChildProcess::TimeoutError => e
        # tries increasingly harsher methods to kill the process.
        process.stop
        return e.message
      end
      write.close
      temp.unlink
    end

    data = read.read
    begin
      # rubocop:disable Security/MarshalLoad
      Marshal.load(data)
      # rubocop:enable Security/MarshalLoad
    rescue StandardError
      raise(data) if @raise_errors

      data
    end
  end
  # rubocop:enable Metrics/AbcSize
  # rubocop:enable Metrics/MethodLength

  private

  def build_tempfile
    file = Tempfile.new('saferuby')
    file.write(MAKE_SAFE_CODE)
    file.write(
      <<~STRING
        result = eval(%q(#{@code}))
        print Marshal.dump(result)
      STRING
    )
    file.rewind
    file
  end
end
