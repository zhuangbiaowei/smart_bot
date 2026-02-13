# frozen_string_literal: true

module SmartBot
  module SkillSystem
    # Central registry for all skills with indexing capabilities
    class SkillRegistry
      include Enumerable

      def initialize
        @skills = {}
        @index = SkillIndex.new
        @lock = Mutex.new
      end

      def register(skill_package)
        raise ArgumentError, "Must be SkillPackage" unless skill_package.is_a?(SkillPackage)

        @lock.synchronize do
          @skills[skill_package.name] = skill_package
          @index.add(skill_package)
        end

        skill_package
      end

      def unregister(name)
        @lock.synchronize do
          skill = @skills.delete(normalize_name(name))
          @index.remove(skill) if skill
          skill
        end
      end

      def find(name)
        @skills[normalize_name(name)]
      end

      def find_by_trigger(query)
        return [] unless query.is_a?(String)

        @index.find_by_trigger(query.downcase)
      end

      def list_available
        @skills.values.select(&:available?)
      end

      def list_always
        @skills.values.select(&:always_load?)
      end

      def list_by_type(type)
        @skills.values.select { |s| s.type == type.to_sym }
      end

      def empty?
        @skills.empty?
      end

      def size
        @skills.size
      end

      def stats
        {
          total: @skills.size,
          available: list_available.size,
          always: list_always.size,
          by_type: %i[instruction script ruby_native openclaw_instruction].to_h { |t| [t, list_by_type(t).size] }
        }
      end

      def each(&block)
        @skills.values.each(&block)
      end

      def clear
        @lock.synchronize do
          @skills.clear
          @index.clear
        end
      end

      def to_h
        {
          skills: @skills.transform_values(&:to_h),
          stats: stats
        }
      end

      private

      def normalize_name(name)
        name.to_s.downcase.gsub(/[^a-z0-9_]/, "_").gsub(/_+/, "_").gsub(/^_+|_$/, "")
      end
    end

    # Inverted index for fast skill lookup
    class SkillIndex
      def initialize
        @trigger_index = Hash.new { |h, k| h[k] = [] }
        @name_index = {}
      end

      def add(skill)
        @name_index[skill.name] = skill

        skill.metadata.triggers.each do |trigger|
          normalized = trigger.to_s.downcase
          @trigger_index[normalized] << skill
          @trigger_index[normalized].uniq!
        end
      end

      def remove(skill)
        @name_index.delete(skill.name)

        skill.metadata.triggers.each do |trigger|
          normalized = trigger.to_s.downcase
          @trigger_index[normalized].delete(skill)
        end
      end

      def find_by_trigger(query)
        normalized = query.to_s.downcase
        matches = []

        @trigger_index.each do |trigger, skills|
          matches.concat(skills) if normalized.include?(trigger)
        end

        matches.uniq
      end

      def find_by_name(name)
        @name_index[name.to_s.downcase]
      end

      def clear
        @trigger_index.clear
        @name_index.clear
      end
    end
  end
end
