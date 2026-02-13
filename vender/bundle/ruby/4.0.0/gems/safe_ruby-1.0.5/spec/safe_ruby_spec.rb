# Copyright (c) 2018 Uku Taht
# frozen_string_literal: true

require 'spec_helper'

MALICIOUS_OPERATIONS = [
  "system('rm *')",
  '`rm *`',
  'Kernel.abort',
  'cat spec/spec_helper.rb',
  'File.class_eval { `echo Hello` }',
  'FileUtils.class_eval { `echo Hello` }',
  'Dir.class_eval { `echo Hello` }',
  'FileTest.class_eval { `echo Hello` }',
  'File.eval "`echo Hello`"',
  'FileUtils.eval "`echo Hello`"',
  'Dir.eval "`echo Hello`"',
  'FileTest.eval "`echo Hello`"',
  'File.instance_eval { `echo Hello` }',
  'FileUtils.instance_eval { `echo Hello` }',
  'Dir.instance_eval { `echo Hello` }',
  'FileTest.instance_eval { `echo Hello` }',
  "f=IO.popen('uname'); f.readlines; f.close",
  "IO.binread('/etc/passwd')",
  "IO.read('/etc/passwd')"
].freeze

describe SafeRuby do
  describe '#eval' do
    it 'allows basic operations' do
      expect(described_class.eval('4 + 5')).to(eq(9))
    end

    it 'allows basic operations with map' do
      expect(described_class.eval('[4, 5].map{|n| n+1}')).to(eq([5, 6]))
    end

    it 'returns correct object' do
      expect(described_class.eval('[1,2,3]')).to(eq([1, 2, 3]))
    end

    MALICIOUS_OPERATIONS.each do |op|
      it "protects from malicious operations like (#{op})" do
        expect { described_class.eval(op) }
          .to(raise_error(RuntimeError))
      end
    end

    describe 'options' do
      describe 'timeout' do
        it 'defaults to a 5 second timeout' do
          time =
            Benchmark.realtime do
              described_class.eval('(1..100000).map {|n| n**100}')
            end
          expect(time).to(be_within(0.5).of(5))
        end

        it 'allows custom timeout' do
          time =
            Benchmark.realtime do
              described_class.eval('(1..100000).map {|n| n**100}', timeout: 1)
            end
          expect(time).to(be_within(0.5).of(1))
        end
      end

      describe 'raising errors' do
        it 'defaults to raising errors' do
          expect { described_class.eval('asdasdasd') }
            .to(raise_error(RuntimeError))
        end

        it 'allows not raising errors' do
          expect { described_class.eval('asdasd', raise_errors: false) }
            .not_to(raise_error)
        end
      end
    end
  end
end
