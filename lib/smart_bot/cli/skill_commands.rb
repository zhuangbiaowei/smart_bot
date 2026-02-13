# frozen_string_literal: true

require "thor"
require_relative "../skill_system/installer"

module SmartBot
  module CLI
    class SkillCommands < Thor
      desc "list", "List all available skills"
      def list
        require_relative "../skill_system"

        SmartBot::SkillSystem.load_all

        say "🛠️  Available Skills\n\n"

        registry = SmartBot::SkillSystem.registry

        if registry.empty?
          say "No skills found.", :yellow
          return
        end

        available = registry.list_available
        unavailable = registry.reject(&:available?)

        if available.any?
          say "Available (#{available.size}):", :green
          available.each { |skill| display_skill(skill) }
          say ""
        end

        if unavailable.any?
          say "Unavailable (#{unavailable.size}):", :yellow
          unavailable.each { |skill| display_skill(skill, available: false) }
        end

        say "\nStats: #{registry.stats}"
      end

      desc "info SKILL_NAME", "Show detailed information about a skill"
      def info(name)
        require_relative "../skill_system"

        SmartBot::SkillSystem.load_all
        skill = SmartBot::SkillSystem.registry.find(name)

        unless skill
          say "❌ Skill not found: #{name}", :red
          return
        end

        say "📋 Skill: #{skill.name}\n"
        say "Description: #{skill.description}"
        say "Version: #{skill.version}"
        say "Type: #{skill.type}"
        say "Available: #{skill.available? ? '✓' : '✗'}"
        say "Always Load: #{skill.always_load? ? '✓' : '✗'}"
        say "Parallel Safe: #{skill.parallel_safe? ? '✓' : '✗'}"
        say "\nPath: #{skill.source_path}"

        metadata = skill.metadata
        say "\nTriggers: #{metadata.triggers.join(', ')}" if metadata.triggers.any?
        say "Anti-triggers: #{metadata.anti_triggers.join(', ')}" if metadata.anti_triggers.any?
        say "Cost: #{metadata.cost_hint}"

        if skill.scripts.any?
          say "\nScripts:"
          skill.scripts.each { |s| say "  - #{File.basename(s)}" }
        end

        if skill.references.any?
          say "\nReferences:"
          skill.references.each { |r| say "  - #{File.basename(r)}" }
        end
      end

      desc "exec SKILL_NAME TASK", "Execute a specific skill with a task"
      def exec(name, *task_parts)
        require_relative "../skill_system"

        task = task_parts.join(" ")

        if task.empty?
          say "❌ Task is required", :red
          say "Usage: smart_bot skill run SKILL_NAME 'your task here'"
          return
        end

        SmartBot::SkillSystem.load_all

        say "🎯 Running skill: #{name}", :cyan
        say "Task: #{task}\n"

        result = SmartBot::SkillSystem.run("$#{name} #{task}")

        if result.success?
          say "✅ Success", :green
          say result.value.to_s
        else
          say "❌ Failed", :red
          say result.error.to_s
        end
      end

      desc "route QUERY", "Test skill routing for a query"
      def route(query)
        require_relative "../skill_system"

        SmartBot::SkillSystem.load_all

        say "🔍 Testing route for: #{query}\n"

        plan = SmartBot::SkillSystem.route(query)

        if plan.empty?
          say "No matching skills found", :yellow
          return
        end

        say "Matched Skills:"
        plan.skills.each_with_index do |skill, idx|
          marker = idx == 0 ? "→" : "  "
          say "#{marker} #{skill.name} (#{skill.description})"
        end

        say "\nPrimary: #{plan.primary_skill.name}"
        say "Parallelizable: #{plan.parallelizable? ? 'Yes' : 'No'}"
        say "Estimated Cost: #{plan.estimated_cost}"

        if plan.fallback_chain.any?
          say "\nFallback Chain:"
          plan.fallback_chain.each { |f| say "  - #{f.is_a?(Symbol) ? f : f.name}" }
        end
      end

      desc "install SOURCE", "Install a skill from various sources"
      long_desc "Install skills from Git, GitHub, NPM, PyPI, or local path\n\n" \
                "Examples:\n" \
                "  smart_bot skill install https://github.com/user/skill.git\n" \
                "  smart_bot skill install user/repo  # GitHub shorthand\n" \
                "  smart_bot skill install ./local/skill/path\n" \
                "  smart_bot skill install npm:package-name\n" \
                "  smart_bot skill install pypi:package-name\n" \
                "  smart_bot skill install https://example.com/skill.zip"
      option :name, aliases: "-n", desc: "Custom name for the skill"
      option :version, aliases: "-v", desc: "Specific version to install"
      option :force, aliases: "-f", type: :boolean, desc: "Overwrite existing skill"
      def install(source)
        require_relative "../skill_system/installer"

        installer = SmartBot::SkillSystem::SkillInstaller.new

        result = installer.install(
          source,
          name: options[:name],
          version: options[:version],
          force: options[:force]
        )

        if result.success?
          say "✅ #{result.message}", :green
          say "   Location: #{result.path}" if result.path
        else
          say "❌ #{result.message}", :red
        end
      end

      desc "uninstall SKILL_NAME", "Remove an installed skill"
      def uninstall(name)
        require_relative "../skill_system/installer"

        installer = SmartBot::SkillSystem::SkillInstaller.new
        result = installer.uninstall(name)

        if result.success?
          say "✅ #{result.message}", :green
        else
          say "❌ #{result.message}", :red
        end
      end

      desc "update SKILL_NAME", "Update an installed skill"
      def update(name)
        require_relative "../skill_system/installer"

        installer = SmartBot::SkillSystem::SkillInstaller.new
        result = installer.update(name)

        if result.success?
          say "✅ #{result.message}", :green
        else
          say "❌ #{result.message}", :red
        end
      end

      desc "installed", "List all installed skills"
      def installed
        require_relative "../skill_system/installer"

        installer = SmartBot::SkillSystem::SkillInstaller.new
        skills = installer.list_installed

        if skills.empty?
          say "No skills installed.", :yellow
          return
        end

        say "📦 Installed Skills (#{skills.size}):\n\n"

        skills.each do |skill|
          say "  #{skill[:name]}"
          say "    Version: #{skill[:metadata]['metadata']&.dig('version') || 'unknown'}"
          say "    Path: #{skill[:path]}"
          say "    Installed: #{skill[:installed_at].strftime('%Y-%m-%d %H:%M')}"
          say ""
        end
      end

      desc "search QUERY", "Search for skills in the registry"
      def search(query)
        require_relative "../skill_system"

        SmartBot::SkillSystem.load_all

        matches = SmartBot::SkillSystem.registry.find_by_trigger(query)

        if matches.empty?
          say "No skills found matching '#{query}'", :yellow
          return
        end

        say "🔍 Found #{matches.size} skill(s) matching '#{query}':\n\n"
        matches.each { |skill| display_skill(skill) }
      end

      private

      def display_skill(skill, available: true)
        status = available ? "✓" : "✗"
        color = available ? :green : :yellow
        say "  #{status} #{skill.name} - #{skill.description}", color
      end
    end
  end
end
