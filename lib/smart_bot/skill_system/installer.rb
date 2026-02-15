# frozen_string_literal: true

require "fileutils"
require "open-uri"
require "json"
require "yaml"
require "ostruct"
require "pathname"

module SmartBot
  module SkillSystem
    # Installer for skills from various sources
    class SkillInstaller
      INSTALL_SOURCES = {
        git: :install_from_git,
        github: :install_from_github,
        local: :install_from_local,
        npm: :install_from_npm,
        pypi: :install_from_pypi,
        url: :install_from_url
      }.freeze

      DEFAULT_INSTALL_DIR = File.expand_path("~/smart_ai/smart_bot/skills")

      attr_reader :target_dir, :observer

      def initialize(target_dir: DEFAULT_INSTALL_DIR, observer: nil)
        @target_dir = target_dir
        @observer = observer || InstallObserver.new
        FileUtils.mkdir_p(@target_dir)
      end

      # Main install method - auto-detects source type
      def install(source, name: nil, version: nil, force: false)
        @observer.install_started(source, name)

        source_type = detect_source_type(source)
        install_method = INSTALL_SOURCES[source_type]

        unless install_method
          return install_result(false, "Unknown source type: #{source}")
        end

        result = send(install_method, source, name: name, version: version, force: force)
        @observer.install_completed(source, result)
        result
      rescue => e
        @observer.install_failed(source, e)
        install_result(false, e.message)
      end

      # Install from Git repository
      def install_from_git(url, name: nil, version: nil, force: false)
        validate_git_url(url)

        @observer.step_started("Cloning from #{url}")

        Dir.mktmpdir do |tmpdir|
          clone_cmd = version ? "git clone --depth 1 --branch #{version} #{url} #{tmpdir}" : "git clone --depth 1 #{url} #{tmpdir}"

          success = system(clone_cmd, out: File::NULL, err: File::NULL)
          unless success
            return install_result(false, "Failed to clone repository")
          end

          inferred_name = name || extract_name_from_git_url(url)
          return install_from_directory_root(tmpdir, name: inferred_name, force: force)
        end
      end

      # Install from GitHub (shorthand)
      def install_from_github(repo, name: nil, version: nil, force: false)
        url = "https://github.com/#{repo}.git"
        install_from_git(url, name: name, version: version, force: force)
      end

      # Install from local directory
      def install_from_local(path, name: nil, force: false, **_options)
        unless File.directory?(path)
          return install_result(false, "Not a directory: #{path}")
        end

        install_from_directory_root(path, name: name, force: force)
      end

      # Install from NPM package
      def install_from_npm(package_name, name: nil, version: nil, **_options)
        skill_name = name || package_name.gsub("@", "").gsub("/", "_")
        target_path = File.join(@target_dir, skill_name)

        Dir.mktmpdir do |tmpdir|
          npm_cmd = version ? "npm pack #{package_name}@#{version} --pack-destination #{tmpdir}" : "npm pack #{package_name} --pack-destination #{tmpdir}"

          @observer.step_started("Downloading NPM package #{package_name}")
          success = system(npm_cmd, out: File::NULL, err: File::NULL)
          unless success
            return install_result(false, "Failed to download NPM package")
          end

          tarball = Dir.glob(File.join(tmpdir, "*.tgz")).first
          unless tarball
            return install_result(false, "NPM package tarball not found")
          end

          extract_dir = File.join(tmpdir, "extracted")
          FileUtils.mkdir_p(extract_dir)
          system("tar -xzf #{tarball} -C #{extract_dir}")

          package_dir = File.join(extract_dir, "package")
          validate_skill_structure(package_dir)

          convert_npm_to_skill(package_dir, skill_name)

          FileUtils.cp_r(package_dir, target_path)
        end

        install_result(true, "Successfully installed NPM package '#{package_name}' as '#{skill_name}'", skill_name: skill_name, path: target_path)
      end

      # Install from PyPI package
      def install_from_pypi(package_name, name: nil, version: nil, **_options)
        skill_name = name || package_name.gsub("-", "_")
        target_path = File.join(@target_dir, skill_name)

        Dir.mktmpdir do |tmpdir|
          pip_cmd = version ? "pip download #{package_name}==#{version} -d #{tmpdir} --no-deps" : "pip download #{package_name} -d #{tmpdir} --no-deps"

          @observer.step_started("Downloading PyPI package #{package_name}")
          success = system(pip_cmd, out: File::NULL, err: File::NULL)
          unless success
            return install_result(false, "Failed to download PyPI package")
          end

          wheel = Dir.glob(File.join(tmpdir, "*.whl")).first
          sdist = Dir.glob(File.join(tmpdir, "*.tar.gz")).first

          package_dir = File.join(tmpdir, "package")
          FileUtils.mkdir_p(package_dir)

          if wheel
            system("unzip -q #{wheel} -d #{package_dir}")
          elsif sdist
            system("tar -xzf #{sdist} -C #{package_dir}")
          else
            return install_result(false, "PyPI package archive not found")
          end

          validate_skill_structure(package_dir)
          convert_pypi_to_skill(package_dir, skill_name)

          FileUtils.cp_r(package_dir, target_path)
        end

        install_result(true, "Successfully installed PyPI package '#{package_name}' as '#{skill_name}'", skill_name: skill_name, path: target_path)
      end

      # Install from direct URL
      def install_from_url(url, name: nil, **_options)
        unless name
          return install_result(false, "Name is required for URL installs")
        end

        target_path = File.join(@target_dir, name)

        if File.exist?(target_path)
          return install_result(false, "Skill '#{name}' already exists")
        end

        @observer.step_started("Downloading from #{url}")

        Dir.mktmpdir do |tmpdir|
          archive_path = File.join(tmpdir, "archive")
          URI.open(url) do |uri|
            File.write(archive_path, uri.read)
          end

          extract_dir = File.join(tmpdir, "extracted")
          FileUtils.mkdir_p(extract_dir)

          if url.end_with?(".zip")
            system("unzip -q #{archive_path} -d #{extract_dir}")
          else
            system("tar -xzf #{archive_path} -C #{extract_dir} 2>/dev/null || tar -xf #{archive_path} -C #{extract_dir}")
          end

          source_dir = Dir.glob(File.join(extract_dir, "*")).find { |f| File.directory?(f) }
          unless source_dir
            return install_result(false, "Could not find extracted directory")
          end

          validate_skill_structure(source_dir)
          FileUtils.cp_r(source_dir, target_path)
        end

        install_result(true, "Successfully installed '#{name}' from URL", skill_name: name, path: target_path)
      end

      # List installed skills
      def list_installed
        return [] unless File.directory?(@target_dir)

        Dir.glob(File.join(@target_dir, "*/")).map do |path|
          skill_name = File.basename(path)
          metadata = load_skill_metadata(path)

          {
            name: skill_name,
            path: path,
            metadata: metadata,
            installed_at: File.mtime(path)
          }
        end
      end

      # Remove installed skill
      def uninstall(skill_name)
        target_path = resolve_installed_skill_path(skill_name)

        unless target_path && File.exist?(target_path)
          return install_result(false, "Skill '#{skill_name}' not found")
        end

        removed_name = File.basename(target_path)
        FileUtils.rm_rf(target_path)
        install_result(true, "Successfully uninstalled '#{removed_name}'")
      end

      # Update installed skill
      def update(skill_name)
        target_path = File.join(@target_dir, skill_name)

        unless File.exist?(target_path)
          return install_result(false, "Skill '#{skill_name}' not found")
        end

        metadata_path = File.join(target_path, "skill.yaml")
        return install_result(false, "No metadata found for '#{skill_name}'") unless File.exist?(metadata_path)

        metadata = YAML.load_file(metadata_path)
        source = metadata.dig("install", "source")

        unless source
          return install_result(false, "No source information found for '#{skill_name}'")
        end

        install(source, name: skill_name, force: true)
      end

      private

      def detect_source_type(source)
        return :github if source.match?(%r{^[\w-]+/[\w-]+$})
        return :git if source.start_with?("git@") || source.end_with?(".git") || source.include?("github.com") || source.include?("gitlab.com")
        return :local if File.directory?(source)
        return :npm if source.start_with?("npm:")
        return :pypi if source.start_with?("pypi:") || source.start_with?("pip:")
        return :url if source.start_with?("http://") || source.start_with?("https://")

        :unknown
      end

      def validate_git_url(url)
        unless url.match?(%r{^(https?://|git@)})
          raise ArgumentError, "Invalid Git URL: #{url}"
        end
      end

      def extract_name_from_git_url(url)
        url.gsub(/\.git$/, "").split("/").last
      end

      def validate_skill_structure(path)
        skill_md = File.join(path, "SKILL.md")
        skill_yaml = File.join(path, "skill.yaml")
        skill_rb = File.join(path, "skill.rb")

        unless File.exist?(skill_md) || File.exist?(skill_yaml) || File.exist?(skill_rb)
          raise ArgumentError, "Invalid skill structure: missing SKILL.md, skill.yaml, or skill.rb"
        end
      end

      def load_skill_metadata(path)
        skill_yaml = File.join(path, "skill.yaml")
        skill_md = File.join(path, "SKILL.md")

        if File.exist?(skill_yaml)
          YAML.load_file(skill_yaml)
        elsif File.exist?(skill_md)
          { name: File.basename(path), type: "instruction" }
        else
          {}
        end
      end

      def convert_npm_to_skill(path, skill_name)
        package_json = File.join(path, "package.json")
        return unless File.exist?(package_json)

        package = JSON.parse(File.read(package_json))

        skill_yaml = {
          "metadata" => {
            "name" => skill_name,
            "description" => package["description"] || "NPM package: #{package["name"]}",
            "version" => package["version"] || "0.1.0",
            "author" => package["author"] || "Unknown"
          },
          "spec" => {
            "type" => "script",
            "triggers" => [skill_name, package["name"]].compact,
            "entrypoints" => [
              {
                "name" => "default",
                "runtime" => "node",
                "command" => package["main"] || "index.js"
              }
            ]
          },
          "install" => {
            "source" => "npm:#{package["name"]}",
            "converted" => true
          }
        }

        File.write(File.join(path, "skill.yaml"), YAML.dump(skill_yaml))
      end

      def convert_pypi_to_skill(path, skill_name)
        setup_py = File.join(path, "setup.py")
        pyproject_toml = File.join(path, "pyproject.toml")

        metadata = { name: skill_name, version: "0.1.0" }

        if File.exist?(pyproject_toml)
          begin
            require "tomlrb"
            pyproject = Tomlrb.load_file(pyproject_toml)
            metadata[:version] = pyproject.dig("project", "version") || "0.1.0"
            metadata[:description] = pyproject.dig("project", "description")
          rescue LoadError
            # TOML parser not available
          end
        end

        skill_yaml = {
          "metadata" => {
            "name" => skill_name,
            "description" => metadata[:description] || "PyPI package: #{skill_name}",
            "version" => metadata[:version],
            "author" => "Unknown"
          },
          "spec" => {
            "type" => "script",
            "triggers" => [skill_name],
            "entrypoints" => [
              {
                "name" => "default",
                "runtime" => "python",
                "command" => "__main__.py"
              }
            ]
          },
          "install" => {
            "source" => "pypi:#{skill_name}",
            "converted" => true
          }
        }

        File.write(File.join(path, "skill.yaml"), YAML.dump(skill_yaml))
      end

      def install_result(success, message, **data)
        OpenStruct.new(
          success?: success,
          message: message,
          **data
        )
      end

      def install_from_directory_root(root_path, name: nil, force: false)
        root_path = File.expand_path(root_path)
        skill_dirs = discover_skill_dirs(root_path)

        if skill_dirs.empty?
          return install_result(false, "No valid skills found under: #{root_path}")
        end

        if skill_dirs.size == 1 && skill_dirs.first == root_path
          return install_single_skill_dir(root_path, skill_name: (name || File.basename(root_path)), force: force)
        end

        if name
          return install_result(false, "--name can only be used when installing a single skill")
        end

        install_multiple_skill_dirs(root_path, skill_dirs, force: force)
      end

      def install_single_skill_dir(path, skill_name:, force:)
        validate_skill_structure(path)
        target_path = File.join(@target_dir, skill_name)

        if File.exist?(target_path)
          return install_result(false, "Skill '#{skill_name}' already exists. Use --force to overwrite.") unless force
          FileUtils.rm_rf(target_path)
        end

        FileUtils.cp_r(path, target_path)
        install_result(true, "Successfully installed '#{skill_name}'", skill_name: skill_name, path: target_path)
      end

      def install_multiple_skill_dirs(root_path, skill_dirs, force:)
        names = generate_unique_skill_names(root_path, skill_dirs)
        installed = []
        skipped = []

        skill_dirs.each do |dir|
          skill_name = names.fetch(dir)
          target_path = File.join(@target_dir, skill_name)

          if File.exist?(target_path)
            unless force
              skipped << "#{skill_name} (already exists)"
              next
            end
            FileUtils.rm_rf(target_path)
          end

          FileUtils.cp_r(dir, target_path)
          installed << { name: skill_name, path: target_path, source: dir }
        end

        return install_result(false, "No skills installed. #{skipped.join(', ')}") if installed.empty?

        message = "Successfully installed #{installed.size} skill(s)"
        message += "; skipped #{skipped.size} (use --force to overwrite)" if skipped.any?
        install_result(true, message, installed: installed, skipped: skipped)
      end

      def discover_skill_dirs(root_path)
        markers = %w[SKILL.md skill.yaml skill.rb]
        dirs = markers.flat_map do |marker|
          Dir.glob(File.join(root_path, "**", marker)).map { |f| File.dirname(f) }
        end

        dirs.uniq.sort
      end

      def generate_unique_skill_names(root_path, skill_dirs)
        basename_counts = Hash.new(0)
        skill_dirs.each { |dir| basename_counts[File.basename(dir)] += 1 }

        skill_dirs.each_with_object({}) do |dir, names|
          base = File.basename(dir)
          if basename_counts[base] == 1
            names[dir] = base
            next
          end

          rel = Pathname.new(dir).relative_path_from(Pathname.new(root_path)).to_s
          names[dir] = rel.tr(File::SEPARATOR, "_")
        end
      end

      def resolve_installed_skill_path(skill_name)
        exact_path = File.join(@target_dir, skill_name)
        return exact_path if File.directory?(exact_path)

        normalized_target = normalize_identifier(skill_name)
        return nil if normalized_target.empty?

        installed_dirs = Dir.glob(File.join(@target_dir, "*/")).select { |path| File.directory?(path) }

        installed_dirs.find do |path|
          matches_identifier?(File.basename(path), normalized_target) ||
            installed_skill_identifiers(path).any? { |identifier| matches_identifier?(identifier, normalized_target) }
        end
      end

      def installed_skill_identifiers(path)
        identifiers = []
        skill_yaml = File.join(path, "skill.yaml")
        skill_md = File.join(path, "SKILL.md")

        if File.exist?(skill_yaml)
          begin
            yaml = YAML.load_file(skill_yaml)
            if yaml.is_a?(Hash)
              metadata_name = yaml.dig("metadata", "name")
              identifiers << metadata_name if metadata_name
            end
          rescue Psych::SyntaxError
            # Ignore malformed metadata file; fallback to other identifiers.
          end
        end

        if File.exist?(skill_md)
          begin
            content = File.read(skill_md, encoding: "UTF-8")
            if content =~ /\A---\s*\n(.+?)\n---/m
              frontmatter = YAML.safe_load($1, permitted_classes: [Date, Time], aliases: true)
              if frontmatter.is_a?(Hash) && frontmatter["name"]
                identifiers << frontmatter["name"]
              end
            end
          rescue Psych::SyntaxError
            # Ignore malformed frontmatter; uninstall can still match by path.
          end
        end

        identifiers
      end

      def matches_identifier?(candidate, normalized_target)
        normalize_identifier(candidate) == normalized_target
      end

      def normalize_identifier(value)
        value.to_s.downcase.gsub(/[^a-z0-9]+/, "_").gsub(/^_+|_+$/, "").gsub(/_+/, "_")
      end
    end

    # Observer for installation events
    class InstallObserver
      def install_started(source, name)
        puts "📦 Installing skill#{name ? " '#{name}'" : ""} from #{source}..."
      end

      def install_completed(source, result)
        if result.success?
          puts "✅ #{result.message}"
        else
          puts "❌ #{result.message}"
        end
      end

      def install_failed(source, error)
        puts "💥 Installation failed: #{error.message}"
      end

      def step_started(description)
        puts "   #{description}..."
      end
    end
  end
end
