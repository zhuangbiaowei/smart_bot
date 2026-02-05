# frozen_string_literal: true

# Skill Adapters - 支持多种 Skill 格式
# 包括：
# - 原生 Ruby Skill (skill.rb)
# - awesome-claude-skills (SKILL.md)
# - ClawdHub Skills (SKILL.md)

require_relative "skill_adapters/claude_skill_adapter"
require_relative "skill_adapters/unified_skill_loader"
