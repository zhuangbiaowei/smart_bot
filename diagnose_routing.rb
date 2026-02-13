#!/usr/bin/env ruby
# frozen_string_literal: true

# Diagnose why youtube-summarizer is not being routed

$LOAD_PATH.unshift File.expand_path("lib", __dir__)

puts "=" * 70
puts "Diagnose youtube-summarizer Routing Issue"
puts "=" * 70

# Step 1: Check if skill files exist
skill_path = File.expand_path("~/smart_ai/smart_bot/skills/youtube-summarizer")
puts "\n[1] Checking skill files..."
puts "  Path: #{skill_path}"
puts "  Exists: #{File.directory?(skill_path)}"
puts "  SKILL.md: #{File.exist?(File.join(skill_path, 'SKILL.md'))}"

# Step 2: Parse the skill
require "yaml"

skill_md = File.join(skill_path, "SKILL.md")
content = File.read(skill_md, encoding: "UTF-8")

if content =~ /\A---\s*\n(.+?)\n---\s*\n/m
  yaml_content = $1
  frontmatter = YAML.safe_load(yaml_content, aliases: true)
  puts "\n[2] Frontmatter parsed:"
  puts "  name: #{frontmatter['name']}"
  puts "  has metadata.openclaw: #{!frontmatter.dig('metadata', 'openclaw').nil?}"
else
  puts "\n[2] ✗ Failed to parse frontmatter"
  exit 1
end

# Step 3: Check OpenClawAdapter
puts "\n[3] Testing OpenClawAdapter.can_parse?"

module ::SmartBot
  module SkillSystem
    module Adapters
      class OpenClawAdapter
        OPENCLAW_METADATA_KEYS = %w[openclaw claude].freeze
        
        def self.can_parse?(frontmatter)
          return false unless frontmatter.is_a?(Hash)
          metadata = frontmatter["metadata"]
          return false unless metadata.is_a?(Hash)
          OPENCLAW_METADATA_KEYS.any? { |key| metadata.key?(key) }
        end
      end
    end
  end
end

if ::SmartBot::SkillSystem::Adapters::OpenClawAdapter.can_parse?(frontmatter)
  puts "  ✓ Recognized as OpenClaw format"
else
  puts "  ✗ NOT recognized as OpenClaw format"
  puts "  This is the problem!"
end

# Step 4: Check metadata structure
puts "\n[4] Checking metadata structure:"
puts "  frontmatter.keys: #{frontmatter.keys.inspect}"
if frontmatter["metadata"]
  puts "  frontmatter['metadata'].keys: #{frontmatter['metadata'].keys.inspect}"
  if frontmatter.dig("metadata", "openclaw")
    puts "  frontmatter['metadata']['openclaw'].keys: #{frontmatter['metadata']['openclaw'].keys.inspect}"
  end
end

# Step 5: Simulate UnifiedLoader.load_openclaw_skill
puts "\n[5] Simulating UnifiedLoader.load_openclaw_skill..."

# Check the exact condition
fm = frontmatter
metadata_section = fm["metadata"]
puts "  metadata_section: #{metadata_section.inspect}"
puts "  metadata_section.class: #{metadata_section.class}"
puts "  metadata_section.is_a?(Hash): #{metadata_section.is_a?(Hash)}"

if metadata_section.is_a?(Hash)
  puts "  metadata_section.key?('openclaw'): #{metadata_section.key?('openclaw')}"
  puts "  metadata_section['openclaw']: #{metadata_section['openclaw'].inspect}"
end

# The actual check in OpenClawAdapter
puts "\n[6] Detailed OpenClawAdapter check:"
puts "  frontmatter.is_a?(Hash): #{frontmatter.is_a?(Hash)}"
puts "  frontmatter['metadata']: #{frontmatter['metadata'].inspect}"
puts "  frontmatter['metadata'].is_a?(Hash): #{frontmatter['metadata'].is_a?(Hash)}"

metadata = frontmatter["metadata"]
if metadata.is_a?(Hash)
  has_openclaw = metadata.key?("openclaw")
  has_claude = metadata.key?("claude")
  puts "  metadata.key?('openclaw'): #{has_openclaw}"
  puts "  metadata.key?('claude'): #{has_claude}"
  puts "  Should return: #{has_openclaw || has_claude}"
end

puts "\n" + "=" * 70
puts "Diagnosis complete"
puts "=" * 70
