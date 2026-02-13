# SmartBot Skill System Architecture

## 1. 架构概述

### 1.1 设计目标

整合 `skill_spec.md` 和 `skill_routing_spec.md`，构建一个统一的 Skill 系统，实现：

- **统一元数据模型**: 兼容 SKILL.md + skill.yaml 格式
- **智能路由**: 两阶段路由（召回 + 打分）自动匹配 Skill
- **安全执行**: 权限控制和沙箱执行
- **自我修复**: 失败检测和自动修复机制
- **渐进加载**: 按需加载减少上下文开销
- **向后兼容**: 支持现有 Ruby Skills 和 Markdown Skills

### 1.2 核心原则

1. **关注点分离**: 路由、执行、适配独立
2. **接口优先**: 明确定义组件间契约
3. **渐进增强**: 从现有代码逐步迁移
4. **可观测性**: 全链路事件追踪

---

## 2. 架构分层

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / API Layer                          │
│  (skill install/list/remove, /run_skill, agent integration)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Routing Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Recall    │  │    Score    │  │    Activation Plan      │ │
│  │  (Phase 1)  │→ │  (Phase 2)  │→ │       Builder           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Execution Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Sandbox   │  │   Repair    │  │    Fallback State       │ │
│  │  Enforcer   │  │    Loop     │  │        Machine          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Layer                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Skill     │  │   Unified   │  │      Registry &         │ │
│  │   Package   │  │   Loader    │  │        Index            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Adapter Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    Ruby     │  │  Markdown   │  │    Claude Format        │ │
│  │   Adapter   │  │   Adapter   │  │       Adapter           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块接口

### 3.1 Core Layer

#### 3.1.1 SkillPackage (统一 Skill 表示)

```ruby
module SmartBot
  module SkillSystem
    class SkillPackage
      attr_reader :name, :source_path, :metadata, :content, :type
      
      # Types: :instruction, :script, :ruby_native
      def initialize(name:, source_path:, metadata:, content:, type:)
        @name = name
        @source_path = source_path
        @metadata = metadata  # SkillMetadata instance
        @content = content    # Lazy loaded content
        @type = type
      end
      
      # Progressive loading
      def load_full_content
        @content ||= ContentLoader.load(@source_path)
      end
      
      def scripts
        @scripts ||= ScriptDiscovery.discover(@source_path)
      end
      
      def references
        @references ||= ReferenceLoader.load(@source_path)
      end
      
      # Routing helpers
      def matches_trigger?(query)
        metadata.triggers.any? { |t| query.include?(t) }
      end
      
      def matches_anti_trigger?(query)
        metadata.anti_triggers.any? { |t| query.include?(t) }
      end
      
      def available?
        metadata.prerequisites.all? { |p| p.satisfied? }
      end
      
      # Execution helpers
      def entrypoint_for(action)
        metadata.entrypoints.find { |e| e.name == action }
      end
      
      def permissions
        metadata.permissions
      end
      
      def to_h
        {
          name: @name,
          type: @type,
          description: metadata.description,
          version: metadata.version,
          available: available?
        }
      end
    end
  end
end
```

#### 3.1.2 SkillMetadata (元数据模型)

```ruby
module SmartBot
  module SkillSystem
    class SkillMetadata
      attr_reader :name, :description, :version, :license, :author
      attr_reader :triggers, :anti_triggers, :cost_hint
      attr_reader :prerequisites, :permissions, :execution_policy
      attr_reader :entrypoints, :parallel_safe, :always
      
      def self.from_yaml(yaml_content)
        # Parse skill.yaml format
      end
      
      def self.from_frontmatter(frontmatter, remaining_content)
        # Parse SKILL.md frontmatter
      end
      
      def initialize(attrs = {})
        @name = attrs[:name]
        @description = attrs[:description]
        @version = attrs[:version] || "0.1.0"
        @license = attrs[:license]
        @author = attrs[:author] || "Unknown"
        
        # Routing fields
        @triggers = attrs[:triggers] || []
        @anti_triggers = attrs[:anti_triggers] || []
        @cost_hint = attrs[:cost_hint] || :medium
        @always = attrs[:always] || false
        @parallel_safe = attrs[:parallel_safe] || false
        
        # Execution fields
        @prerequisites = attrs[:prerequisites] || []
        @permissions = attrs[:permissions] || default_permissions
        @execution_policy = attrs[:execution_policy] || default_execution_policy
        @entrypoints = attrs[:entrypoints] || []
      end
      
      def cost_penalty
        case @cost_hint
        when :low then 0.0
        when :medium then -0.05
        when :high then -0.10
        else 0.0
        end
      end
      
      private
      
      def default_permissions
        PermissionSet.new(
          filesystem: { read: [], write: [] },
          network: { outbound: false },
          environment: { allow: [] }
        )
      end
      
      def default_execution_policy
        ExecutionPolicy.new(
          sandbox: :process,
          approval: :ask,
          timeout: 120
        )
      end
    end
  end
end
```

#### 3.1.3 UnifiedLoader (统一加载器)

```ruby
module SmartBot
  module SkillSystem
    class UnifiedLoader
      include Enumerable
      
      DISCOVERY_PATHS = [
        "%{workspace}/skills",
        "%{repo_root}/.agents/skills",
        "%{repo_root}/.claude/skills",
        "%{home}/.agents/skills",
        "%{home}/.claude/skills",
        "/etc/agent/skills"
      ].freeze
      
      def initialize(workspace:, repo_root: nil, home: nil)
        @workspace = workspace
        @repo_root = repo_root || Dir.pwd
        @home = home || Dir.home
        @adapters = {
          ruby: RubySkillAdapter.new,
          markdown: MarkdownSkillAdapter.new,
          claude: ClaudeSkillAdapter.new
        }
      end
      
      # Load all skills from discovery paths
      def load_all
        skills = []
        
        discovery_paths.each do |path|
          next unless File.directory?(path)
          skills.concat(load_from_directory(path))
        end
        
        skills
      end
      
      # Load specific skill by name
      def load_skill(name)
        discovery_paths.each do |path|
          skill = load_skill_from_path(File.join(path, name))
          return skill if skill
        end
        nil
      end
      
      private
      
      def discovery_paths
        DISCOVERY_PATHS.map do |template|
          template % {
            workspace: @workspace,
            repo_root: @repo_root,
            home: @home
          }
        end
      end
      
      def load_from_directory(dir)
        skills = []
        
        Dir.glob(File.join(dir, "*/")).each do |skill_dir|
          skill = load_skill_from_path(skill_dir)
          skills << skill if skill
        end
        
        skills
      end
      
      def load_skill_from_path(path)
        # Try adapters in order of specificity
        [:ruby, :markdown, :claude].each do |adapter_type|
          adapter = @adapters[adapter_type]
          skill = adapter.load(path)
          return skill if skill
        end
        
        nil
      end
    end
  end
end
```

#### 3.1.4 SkillRegistry (注册表与索引)

```ruby
module SmartBot
  module SkillSystem
    class SkillRegistry
      include Enumerable
      
      def initialize
        @skills = {}  # name -> SkillPackage
        @index = SkillIndex.new
        @lock = Mutex.new
      end
      
      # Registration
      def register(skill_package)
        @lock.synchronize do
          @skills[skill_package.name] = skill_package
          @index.add(skill_package)
        end
      end
      
      def unregister(name)
        @lock.synchronize do
          skill = @skills.delete(name)
          @index.remove(skill) if skill
        end
      end
      
      # Query
      def find(name)
        @skills[name.to_s.downcase.gsub(/[^a-z0-9_]/, "_")]
      end
      
      def find_by_trigger(query)
        @index.find_by_trigger(query)
      end
      
      def semantic_search(query, top_k: 3)
        @index.semantic_search(query, top_k: top_k)
      end
      
      def list_available
        @skills.values.select(&:available?)
      end
      
      def list_always
        @skills.values.select { |s| s.metadata.always }
      end
      
      # Statistics for routing
      def stats
        {
          total: @skills.size,
          available: list_available.size,
          always: list_always.size,
          by_type: @skills.group_by { |_, s| s.type }.transform_values(&:size)
        }
      end
      
      def each(&block)
        @skills.values.each(&block)
      end
    end
    
    # Inverted index for fast trigger matching
    class SkillIndex
      def initialize
        @trigger_index = Hash.new { |h, k| h[k] = [] }
        @semantic_index = nil  # Lazy loaded
      end
      
      def add(skill)
        skill.metadata.triggers.each do |trigger|
          @trigger_index[trigger.downcase] << skill
        end
      end
      
      def remove(skill)
        skill.metadata.triggers.each do |trigger|
          @trigger_index[trigger.downcase].delete(skill)
        end
      end
      
      def find_by_trigger(query)
        normalized = query.downcase
        @trigger_index.select { |k, _| normalized.include?(k) }
                      .flat_map { |_, skills| skills }
                      .uniq
      end
      
      def semantic_search(query, top_k: 3)
        # Placeholder for semantic search implementation
        # Could use embeddings or simple TF-IDF
      end
    end
  end
end
```

---

### 3.2 Routing Layer

#### 3.2.1 Router (两阶段路由)

```ruby
module SmartBot
  module SkillSystem
    class Router
      DEFAULT_THRESHOLD = 0.65
      MAX_PARALLEL_SKILLS = 2
      
      def initialize(registry:, scorer: nil)
        @registry = registry
        @scorer = scorer || SkillScorer.new
      end
      
      # Main entry point
      def route(query:, context: {}, history: [], stats: {})
        # Phase 1: Recall
        candidates = recall_candidates(query, context)
        
        # Phase 2: Score and filter
        scored = score_candidates(candidates, query, context, stats)
        selected = select_candidates(scored)
        
        # Build activation plan
        build_activation_plan(selected, query, context)
      end
      
      private
      
      # Phase 1: Recall candidates
      def recall_candidates(query, context)
        candidates = []
        
        # 1. Hard triggers (explicit skill invocation)
        if (match = query.match(/\$(\w+)/) || query.match(/使用\s+(\w+)\s+skill/i))
          skill_name = match[1].downcase
          skill = @registry.find(skill_name)
          candidates << SkillCandidate.new(skill, source: :forced) if skill
        end
        
        # 2. Rule-based recall (triggers)
        trigger_matches = @registry.find_by_trigger(query)
        trigger_matches.each do |skill|
          next if candidates.any? { |c| c.skill.name == skill.name }
          candidates << SkillCandidate.new(skill, source: :rule)
        end
        
        # 3. Semantic recall
        semantic_matches = @registry.semantic_search(query, top_k: 3)
        semantic_matches.each do |skill|
          next if candidates.any? { |c| c.skill.name == skill.name }
          candidates << SkillCandidate.new(skill, source: :semantic)
        end
        
        candidates
      end
      
      # Phase 2: Score candidates
      def score_candidates(candidates, query, context, stats)
        candidates.map do |candidate|
          score = @scorer.score(
            candidate: candidate,
            query: query,
            context: context,
            stats: stats
          )
          ScoredCandidate.new(candidate, score)
        end
      end
      
      # Select candidates above threshold
      def select_candidates(scored)
        forced = scored.select { |s| s.candidate.source == :forced }
        return forced if forced.any?
        
        regular = scored.select { |s| s.score >= DEFAULT_THRESHOLD }
        
        # If no candidates pass threshold but forced exists
        if regular.empty? && forced.any?
          forced
        else
          regular.sort_by(&:score).reverse
        end
      end
      
      def build_activation_plan(selected, query, context)
        ActivationPlan.new(
          skills: selected.map(&:candidate),
          parameters: extract_parameters(query, selected),
          primary_skill: selected.first&.candidate&.skill,
          fallback_chain: build_fallback_chain(selected),
          parallel_groups: group_parallelizable(selected),
          estimated_cost: estimate_cost(selected)
        )
      end
      
      def extract_parameters(query, selected)
        # Extract task parameters from query
        # This could use simple regex or more sophisticated NLP
        { task: query }
      end
      
      def build_fallback_chain(selected)
        # Build ordered list of fallback options
        selected.drop(1).map(&:candidate) + [:generic_tools]
      end
      
      def group_parallelizable(selected)
        # Group skills that can run in parallel
        groups = []
        current_group = []
        
        selected.each do |scored|
          skill = scored.candidate.skill
          if skill.metadata.parallel_safe && current_group.size < MAX_PARALLEL_SKILLS
            current_group << skill
          else
            groups << current_group unless current_group.empty?
            current_group = [skill]
          end
        end
        
        groups << current_group unless current_group.empty?
        groups
      end
      
      def estimate_cost(selected)
        selected.sum { |s| cost_weight(s.candidate.skill.metadata.cost_hint) }
      end
      
      def cost_weight(hint)
        case hint
        when :low then 1
        when :medium then 2
        when :high then 3
        else 2
        end
      end
    end
    
    # Value objects
    SkillCandidate = Struct.new(:skill, :source, keyword_init: true)
    ScoredCandidate = Struct.new(:candidate, :score, keyword_init: true)
  end
end
```

#### 3.2.2 SkillScorer (打分器)

```ruby
module SmartBot
  module SkillSystem
    class SkillScorer
      # Weights from routing spec
      WEIGHTS = {
        intent_match: 0.40,
        trigger_match: 0.20,
        success_rate: 0.15,
        context_readiness: 0.10,
        cost_penalty: 0.10,
        conflict_penalty: 0.05
      }.freeze
      
      def score(candidate:, query:, context:, stats:)
        skill = candidate.skill
        
        scores = {
          intent_match: calculate_intent_match(candidate, query),
          trigger_match: calculate_trigger_match(candidate, query),
          success_rate: calculate_success_rate(skill, stats),
          context_readiness: calculate_context_readiness(skill, context),
          cost_penalty: skill.metadata.cost_penalty,
          conflict_penalty: calculate_conflict_penalty(candidate, query)
        }
        
        # Weighted sum
        total = scores.sum { |key, value| WEIGHTS[key] * value }
        
        # Apply availability penalty
        total *= 0.5 unless skill.available?
        
        total.clamp(0.0, 1.0)
      end
      
      private
      
      def calculate_intent_match(candidate, query)
        # Semantic similarity between query and skill description
        # Placeholder: could use embeddings
        0.5
      end
      
      def calculate_trigger_match(candidate, query)
        return 1.0 if candidate.source == :forced
        return 0.9 if candidate.source == :rule
        0.6  # semantic match baseline
      end
      
      def calculate_success_rate(skill, stats)
        skill_stats = stats[skill.name]
        return 0.5 unless skill_stats
        
        success_rate = skill_stats[:successes].to_f / skill_stats[:total]
        success_rate.clamp(0.0, 1.0)
      end
      
      def calculate_context_readiness(skill, context)
        # Check if required context is available
        return 1.0 if context.empty?
        
        # Check prerequisites
        skill.metadata.prerequisites.count(&:satisfied?).to_f /
          [skill.metadata.prerequisites.size, 1].max
      end
      
      def calculate_conflict_penalty(candidate, query)
        skill = candidate.skill
        
        # Check anti-triggers
        if skill.matches_anti_trigger?(query)
          -1.0  # Strong penalty
        else
          0.0
        end
      end
    end
  end
end
```

#### 3.2.3 ActivationPlan (激活计划)

```ruby
module SmartBot
  module SkillSystem
    class ActivationPlan
      attr_reader :skills, :parameters, :primary_skill, 
                  :fallback_chain, :parallel_groups, :estimated_cost
      
      def initialize(skills:, parameters:, primary_skill:, 
                     fallback_chain:, parallel_groups:, estimated_cost:)
        @skills = skills
        @parameters = parameters
        @primary_skill = primary_skill
        @fallback_chain = fallback_chain
        @parallel_groups = parallel_groups
        @estimated_cost = estimated_cost
      end
      
      def empty?
        @skills.empty?
      end
      
      def single?
        @skills.size == 1
      end
      
      def parallelizable?
        @parallel_groups.size > 1 || 
          (@parallel_groups.first&.size || 0) > 1
      end
      
      def to_h
        {
          skills: @skills.map(&:name),
          primary: @primary_skill&.name,
          parameters: @parameters,
          fallback_chain: @fallback_chain.map { |f| f.is_a?(Symbol) ? f : f.name },
          parallel_groups: @parallel_groups.map { |g| g.map(&:name) },
          estimated_cost: @estimated_cost
        }
      end
    end
  end
end
```

---

### 3.3 Execution Layer

#### 3.3.1 SkillExecutor (执行器)

```ruby
module SmartBot
  module SkillSystem
    class SkillExecutor
      def initialize(sandbox: nil, observer: nil)
        @sandbox = sandbox || Sandbox.new
        @observer = observer || ExecutionObserver.new
      end
      
      # Execute activation plan
      def execute(plan, context: {})
        @observer.plan_started(plan)
        
        result = execute_plan(plan, context)
        
        @observer.plan_completed(plan, result)
        result
      rescue => e
        @observer.plan_failed(plan, e)
        raise
      end
      
      private
      
      def execute_plan(plan, context)
        # Execute parallel groups sequentially
        results = []
        
        plan.parallel_groups.each do |group|
          if group.size == 1
            # Single skill execution
            result = execute_skill(group.first, plan.parameters, context)
            results << result
            
            # Check if we should continue
            break unless result.success?
          else
            # Parallel execution
            group_results = execute_parallel(group, plan.parameters, context)
            results.concat(group_results)
            
            # Check if all succeeded
            break if group_results.any?(&:failure?)
          end
        end
        
        ExecutionResult.new(
          success: results.all?(&:success?),
          results: results,
          primary_result: results.first
        )
      end
      
      def execute_skill(skill, parameters, context)
        @observer.skill_started(skill, parameters)
        
        # Check permissions
        unless @sandbox.check_permissions(skill.permissions)
          return ExecutionResult.failure(
            skill: skill,
            error: "Permission check failed"
          )
        end
        
        # Execute in sandbox
        result = @sandbox.execute(skill, parameters, context)
        
        @observer.skill_completed(skill, result)
        result
      rescue => e
        @observer.skill_failed(skill, e)
        ExecutionResult.failure(skill: skill, error: e.message)
      end
      
      def execute_parallel(skills, parameters, context)
        # Use fibers or threads for parallel execution
        fibers = skills.map do |skill|
          Fiber.new { execute_skill(skill, parameters, context) }
        end
        
        fibers.map(&:resume)
      end
    end
    
    # Result object
    class ExecutionResult
      attr_reader :skill, :value, :error, :metadata
      
      def self.success(skill:, value:, metadata: {})
        new(skill: skill, value: value, metadata: metadata, success: true)
      end
      
      def self.failure(skill:, error:, metadata: {})
        new(skill: skill, error: error, metadata: metadata, success: false)
      end
      
      def initialize(skill:, value: nil, error: nil, metadata: {}, success: false)
        @skill = skill
        @value = value
        @error = error
        @metadata = metadata
        @success = success
      end
      
      def success?
        @success
      end
      
      def failure?
        !@success
      end
    end
  end
end
```

#### 3.3.2 Sandbox (沙箱)

```ruby
module SmartBot
  module SkillSystem
    class Sandbox
      def check_permissions(permissions)
        # Verify filesystem permissions
        permissions.filesystem[:read].each do |path|
          return false unless readable?(path)
        end
        
        permissions.filesystem[:write].each do |path|
          return false unless writable?(path)
        end
        
        # Verify network permissions
        if permissions.network[:outbound]
          return false unless network_allowed?
        end
        
        # Verify environment variables
        permissions.environment[:allow].each do |var|
          return false unless ENV.key?(var)
        end
        
        true
      end
      
      def execute(skill, parameters, context)
        policy = skill.metadata.execution_policy
        
        case policy.sandbox
        when :none
          execute_unrestricted(skill, parameters, context)
        when :process
          execute_in_process(skill, parameters, context)
        when :container
          execute_in_container(skill, parameters, context)
        else
          execute_in_process(skill, parameters, context)
        end
      end
      
      private
      
      def execute_unrestricted(skill, parameters, context)
        # Direct execution for trusted skills
        # This is the current behavior
        invoke_skill(skill, parameters, context)
      end
      
      def execute_in_process(skill, parameters, context)
        # Execute with resource limits
        # Could use Timeout, chroot, or seccomp
        Timeout.timeout(skill.metadata.execution_policy.timeout) do
          invoke_skill(skill, parameters, context)
        end
      rescue Timeout::Error
        ExecutionResult.failure(
          skill: skill,
          error: "Execution timeout"
        )
      end
      
      def execute_in_container(skill, parameters, context)
        # Future: Docker/containerd integration
        raise NotImplementedError, "Container sandbox not yet implemented"
      end
      
      def invoke_skill(skill, parameters, context)
        # Dispatch to appropriate adapter
        case skill.type
        when :ruby_native
          invoke_ruby_skill(skill, parameters, context)
        when :instruction
          invoke_instruction_skill(skill, parameters, context)
        when :script
          invoke_script_skill(skill, parameters, context)
        else
          ExecutionResult.failure(
            skill: skill,
            error: "Unknown skill type: #{skill.type}"
          )
        end
      end
      
      def invoke_ruby_skill(skill, parameters, context)
        definition = SmartBot::Skill.find(skill.name)
        return ExecutionResult.failure(
          skill: skill,
          error: "Ruby skill not registered: #{skill.name}"
        ) unless definition
        
        # Execute through SmartAgent tool system
        result = SmartAgent::Tool.call(
          "#{skill.name}_agent",
          parameters.transform_keys(&:to_s)
        )
        
        ExecutionResult.success(skill: skill, value: result)
      end
      
      def invoke_instruction_skill(skill, parameters, context)
        # Load full content and execute via LLM
        skill.load_full_content
        
        # Build system prompt from SKILL.md
        system_prompt = skill.content
        
        # Execute via LLM
        result = execute_via_llm(system_prompt, parameters[:task])
        
        ExecutionResult.success(skill: skill, value: result)
      end
      
      def invoke_script_skill(skill, parameters, context)
        entrypoint = skill.entrypoint_for(parameters[:action] || "default")
        return ExecutionResult.failure(
          skill: skill,
          error: "No entrypoint found"
        ) unless entrypoint
        
        script_path = File.join(skill.source_path, entrypoint.command)
        result = execute_script(script_path, parameters)
        
        ExecutionResult.success(skill: skill, value: result)
      end
      
      def execute_via_llm(system_prompt, user_prompt)
        # Use SmartPrompt to execute
        config_path = File.expand_path("~/.smart_bot/smart_prompt.yml")
        return "Error: SmartPrompt config not found" unless File.exist?(config_path)
        
        config = YAML.load_file(config_path)
        engine = SmartPrompt::Engine.new(config_path)
        llm_name = config["default_llm"] || "deepseek"
        
        worker_name = :"skill_execution_#{Time.now.to_i}"
        SmartPrompt.define_worker worker_name do
          use llm_name
          sys_msg system_prompt
          prompt user_prompt
          send_msg
        end
        
        engine.call_worker(worker_name, {})
      end
      
      def execute_script(script_path, parameters)
        # Execute script with parameters
        # Similar to existing ClaudeSkillAdapter implementation
        ext = File.extname(script_path).downcase
        
        command = case ext
                  when ".py" then "python3 #{script_path}"
                  when ".rb" then "ruby #{script_path}"
                  when ".sh" then "bash #{script_path}"
                  when ".js" then "node #{script_path}"
                  else script_path
                  end
        
        output = `#{command} 2>&1`
        exit_status = $?.exitstatus
        
        if exit_status == 0
          { success: true, output: output }
        else
          { success: false, error: output, exit_code: exit_status }
        end
      end
      
      def readable?(path)
        File.readable?(File.expand_path(path))
      end
      
      def writable?(path)
        dir = File.expand_path(path)
        File.directory?(dir) ? File.writable?(dir) : File.writable?(File.dirname(dir))
      end
      
      def network_allowed?
        # Check if network is allowed in this environment
        true
      end
    end
  end
end
```

#### 3.3.3 RepairLoop (修复循环)

```ruby
module SmartBot
  module SkillSystem
    class RepairLoop
      MAX_REPAIR_ATTEMPTS = 2
      PROGRESS_THRESHOLD = 0.70
      
      def initialize(executor:, observer: nil)
        @executor = executor
        @observer = observer
        @budget = RepairBudget.new
      end
      
      def execute_with_repair(skill, parameters, context)
        result = @executor.execute_skill(skill, parameters, context)
        
        return result if result.success?
        
        # Check if repairable
        unless repairable?(result)
          @observer.repair_skipped(skill, result, "Not repairable")
          return result
        end
        
        # Attempt repair
        attempt_repair(skill, parameters, context, result)
      end
      
      private
      
      def repairable?(result)
        return false unless result.failure?
        
        # Check failure type
        error = result.error.to_s.downcase
        
        # Repairable patterns
        repairable_patterns = [
          /parameter/i,
          /missing.*field/i,
          /not found/i,
          /path.*error/i,
          /template/i,
          /reference/i
        ]
        
        repairable_patterns.any? { |p| error =~ p }
      end
      
      def attempt_repair(skill, parameters, context, original_result)
        attempt = 0
        current_result = original_result
        
        while attempt < MAX_REPAIR_ATTEMPTS && @budget.has_budget?
          attempt += 1
          @observer.repair_attempted(skill, attempt)
          
          # Analyze failure
          diagnosis = diagnose_failure(skill, current_result)
          
          # Generate repair plan
          repair_plan = generate_repair_plan(skill, diagnosis)
          
          # Apply patches
          patches_applied = apply_patches(skill, repair_plan)
          
          unless patches_applied
            @observer.repair_failed(skill, attempt, "No patches applied")
            break
          end
          
          # Retry execution
          new_result = @executor.execute_skill(skill, parameters, context)
          
          # Evaluate improvement
          if improved?(current_result, new_result)
            current_result = new_result
            
            if new_result.success?
              @observer.repair_succeeded(skill, attempt)
              return new_result
            end
          else
            @observer.repair_no_improvement(skill, attempt)
            break
          end
          
          @budget.consume_attempt
        end
        
        current_result
      end
      
      def diagnose_failure(skill, result)
        # Analyze error to determine root cause
        {
          error_type: classify_error(result.error),
          error_location: locate_error(skill, result.error),
          skill_files: affected_files(skill, result.error)
        }
      end
      
      def classify_error(error)
        error_str = error.to_s.downcase
        
        if error_str.include?("parameter") || error_str.include?("argument")
          :parameter_error
        elsif error_str.include?("file") || error_str.include?("path")
          :file_error
        elsif error_str.include?("template") || error_str.include?("reference")
          :template_error
        else
          :unknown_error
        end
      end
      
      def locate_error(skill, error)
        # Extract file/line info from error
        # Placeholder implementation
        { file: nil, line: nil }
      end
      
      def affected_files(skill, error)
        # Determine which skill files need patching
        [File.join(skill.source_path, "SKILL.md")]
      end
      
      def generate_repair_plan(skill, diagnosis)
        case diagnosis[:error_type]
        when :parameter_error
          { action: :update_parameters, files: ["SKILL.md"] }
        when :file_error
          { action: :fix_paths, files: diagnosis[:skill_files] }
        when :template_error
          { action: :fix_templates, files: ["SKILL.md"] }
        else
          { action: :unknown, files: [] }
        end
      end
      
      def apply_patches(skill, plan)
        # Apply patches to skill files
        # This is a simplified version - real implementation would be more sophisticated
        case plan[:action]
        when :update_parameters
          # Update SKILL.md with parameter info
          true
        when :fix_paths
          # Fix path references
          true
        when :fix_templates
          # Fix template references
          true
        else
          false
        end
      end
      
      def improved?(old_result, new_result)
        return true if new_result.success? && !old_result.success?
        
        # Check if error changed from fatal to retryable
        old_fatal = old_result.error.to_s.include?("fatal")
        new_fatal = new_result.error.to_s.include?("fatal")
        
        return true if old_fatal && !new_fatal
        
        # Check if we made progress
        false
      end
    end
    
    class RepairBudget
      MAX_ATTEMPTS = 2
      MAX_PATCHED_FILES = 3
      MAX_PATCH_HUNKS = 8
      
      def initialize
        @attempts = 0
        @patched_files = 0
        @patched_hunks = 0
      end
      
      def has_budget?
        @attempts < MAX_ATTEMPTS &&
          @patched_files < MAX_PATCHED_FILES &&
          @patched_hunks < MAX_PATCH_HUNKS
      end
      
      def consume_attempt
        @attempts += 1
      end
      
      def consume_patch(files:, hunks:)
        @patched_files += files
        @patched_hunks += hunks
      end
    end
  end
end
```

#### 3.3.4 FallbackStateMachine (回退状态机)

```ruby
module SmartBot
  module SkillSystem
    class FallbackStateMachine
      STATES = %i[selected running success retryable_failure 
                  fatal_failure fallback exit].freeze
      
      def initialize(plan:, executor:, observer: nil)
        @plan = plan
        @executor = executor
        @observer = observer
        @state = :selected
        @retry_count = 0
        @max_retries = 1
        @current_skill_index = 0
      end
      
      def run(context: {})
        transition_to(:running)
        
        loop do
          case @state
          when :running
            execute_current_skill(context)
          when :retryable_failure
            handle_retryable_failure
          when :fatal_failure
            handle_fatal_failure
          when :fallback
            execute_fallback(context)
          when :success, :exit
            break
          end
        end
        
        @result
      end
      
      private
      
      def execute_current_skill(context)
        skill = current_skill
        return transition_to(:fallback) unless skill
        
        @observer.skill_started(skill)
        result = @executor.execute_skill(skill, @plan.parameters, context)
        
        if result.success?
          @result = result
          transition_to(:success)
        elsif retryable?(result)
          @last_error = result.error
          transition_to(:retryable_failure)
        else
          @last_error = result.error
          transition_to(:fatal_failure)
        end
      end
      
      def handle_retryable_failure
        if @retry_count < @max_retries
          @retry_count += 1
          @observer.retry_attempted(current_skill, @retry_count)
          transition_to(:running)
        else
          transition_to(:fallback)
        end
      end
      
      def handle_fatal_failure
        # Skip retry for fatal failures
        transition_to(:fallback)
      end
      
      def execute_fallback(context)
        fallback_skill = next_fallback_skill
        
        if fallback_skill == :generic_tools
          @result = execute_generic_tools(context)
          transition_to(@result.success? ? :success : :exit)
        elsif fallback_skill
          @current_skill_index = @plan.skills.index(fallback_skill) || @current_skill_index + 1
          @retry_count = 0
          transition_to(:running)
        else
          @result = ExecutionResult.failure(
            skill: nil,
            error: "All fallback options exhausted. Last error: #{@last_error}"
          )
          transition_to(:exit)
        end
      end
      
      def current_skill
        @plan.skills[@current_skill_index]
      end
      
      def next_fallback_skill
        fallback_index = @plan.fallback_chain.index { |s| 
          s.is_a?(Symbol) || @plan.skills.index(s) > @current_skill_index 
        }
        
        return nil unless fallback_index
        
        @plan.fallback_chain[fallback_index..].find do |s|
          s == :generic_tools || !@plan.skills[0..@current_skill_index].include?(s)
        end
      end
      
      def retryable?(result)
        error = result.error.to_s.downcase
        
        # Non-retryable patterns
        non_retryable = [
          /permission denied/i,
          /not found/i,
          /invalid.*format/i,
          /capability.*not.*available/i
        ]
        
        !non_retryable.any? { |p| error =~ p }
      end
      
      def execute_generic_tools(context)
        # Fall back to generic tool execution
        # This could use the existing SmartAgent tool system
        ExecutionResult.success(
          skill: nil,
          value: { message: "Executed using generic tools" }
        )
      end
      
      def transition_to(new_state)
        old_state = @state
        @state = new_state
        @observer.state_transition(old_state, new_state) if @observer
      end
    end
  end
end
```

---

### 3.4 Adapter Layer

#### 3.4.1 RubySkillAdapter

```ruby
module SmartBot
  module SkillSystem
    class RubySkillAdapter
      def load(skill_path)
        skill_rb = File.join(skill_path, "skill.rb")
        return nil unless File.exist?(skill_rb)
        
        # Load the Ruby skill definition
        begin
          load skill_rb
          
          # Extract metadata from registered skill
          skill_name = File.basename(skill_path)
          definition = SmartBot::Skill.find(skill_name.to_sym)
          
          return nil unless definition
          
          # Create SkillPackage from definition
          SkillPackage.new(
            name: definition.name.to_s,
            source_path: skill_path,
            metadata: extract_metadata(definition),
            content: nil,  # Ruby skills don't have SKILL.md content
            type: :ruby_native
          )
        rescue => e
          SmartBot.logger&.error "Failed to load Ruby skill #{skill_path}: #{e.message}"
          nil
        end
      end
      
      private
      
      def extract_metadata(definition)
        SkillMetadata.new(
          name: definition.name.to_s,
          description: definition.description,
          version: definition.version,
          author: definition.author,
          triggers: infer_triggers(definition),
          type: :ruby_native
        )
      end
      
      def infer_triggers(definition)
        # Infer triggers from description and tools
        triggers = [definition.name.to_s]
        definition.description.to_s.downcase.split.each do |word|
          triggers << word if word.length > 3
        end
        triggers.uniq
      end
    end
  end
end
```

#### 3.4.2 MarkdownSkillAdapter

```ruby
module SmartBot
  module SkillSystem
    class MarkdownSkillAdapter
      def load(skill_path)
        skill_md = File.join(skill_path, "SKILL.md")
        return nil unless File.exist?(skill_md)
        
        # Check if there's also a skill.yaml
        skill_yaml = File.join(skill_path, "skill.yaml")
        
        content = File.read(skill_md)
        
        if File.exist?(skill_yaml)
          # Full spec-compliant skill
          load_full_skill(skill_path, skill_md, skill_yaml)
        else
          # Simple SKILL.md only skill
          load_simple_skill(skill_path, skill_md, content)
        end
      end
      
      private
      
      def load_full_skill(skill_path, skill_md, skill_yaml)
        yaml_content = YAML.load_file(skill_yaml)
        md_content = File.read(skill_md)
        
        metadata = SkillMetadata.from_yaml(yaml_content)
        
        SkillPackage.new(
          name: metadata.name,
          source_path: skill_path,
          metadata: metadata,
          content: md_content,
          type: detect_type(yaml_content)
        )
      end
      
      def load_simple_skill(skill_path, skill_md, content)
        # Parse frontmatter from SKILL.md
        parser = SkillMdParser.new(content, skill_path)
        
        return nil unless parser.valid?
        
        metadata = SkillMetadata.from_frontmatter(
          parser.frontmatter,
          parser.remaining_content
        )
        
        SkillPackage.new(
          name: parser.name,
          source_path: skill_path,
          metadata: metadata,
          content: parser.remaining_content,
          type: :instruction
        )
      end
      
      def detect_type(yaml)
        spec = yaml.dig("spec", "type")
        case spec
        when "script" then :script
        when "instruction" then :instruction
        else :instruction
        end
      end
    end
    
    # Parser for SKILL.md frontmatter
    class SkillMdParser
      attr_reader :name, :description, :version, :author, 
                  :frontmatter, :remaining_content, :skill_path
      
      def initialize(raw_content, skill_path = nil)
        @raw_content = raw_content
        @skill_path = skill_path
        @frontmatter = {}
        @remaining_content = raw_content
        parse!
      end
      
      def valid?
        !@name.nil? && !@name.empty?
      end
      
      private
      
      def parse!
        # Parse YAML frontmatter
        if @raw_content =~ /\A---\s*\n(.+?)\n---\s*\n(.*)\z/m
          yaml_content = $1
          @remaining_content = $2
          
          begin
            @frontmatter = YAML.safe_load(yaml_content, 
              permitted_classes: [Date, Time], 
              aliases: true
            ) || {}
            
            @name = normalize_name(@frontmatter["name"])
            @description = @frontmatter["description"]
            @version = @frontmatter["version"] || "0.1.0"
            @author = @frontmatter["author"] || "Unknown"
          rescue Psych::SyntaxError
            @remaining_content = @raw_content
          end
        end
      end
      
      def normalize_name(name)
        return nil if name.nil?
        
        name.downcase
            .gsub(/[^a-z0-9]+/, "_")
            .gsub(/^_+|_+$/, "")
            .gsub(/_+/, "_")
      end
    end
  end
end
```

---

## 4. 数据流

### 4.1 路由到执行的数据流

```
User Query
    │
    ▼
┌─────────────────┐
│     Router      │
│  ┌───────────┐  │
│  │  Recall   │  │───▶ Hard Triggers ───▶ SkillRegistry.find()
│  │  Phase 1  │  │───▶ Rule Matching ───▶ SkillRegistry.find_by_trigger()
│  └───────────┘  │───▶ Semantic Search ─▶ SkillRegistry.semantic_search()
│  ┌───────────┐  │
│  │   Score   │  │───▶ SkillScorer.score() ───▶ Score breakdown
│  │  Phase 2  │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │   Build   │  │───▶ ActivationPlan.new()
│  │    Plan   │  │        ├── skills[]
│  └───────────┘  │        ├── parameters{}
│                 │        ├── fallback_chain[]
│                 │        └── parallel_groups[]
└─────────────────┘
    │
    ▼ ActivationPlan
┌─────────────────┐
│   Executor      │
│  ┌───────────┐  │
│  │  Sandbox  │  │───▶ Permission checks
│  │  Checks   │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │  Execute  │  │───▶ Invoke skill via adapter
│  │   Skill   │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │  Repair   │  │───▶ RepairLoop (if failure)
│  │   Loop    │  │
│  └───────────┘  │
│  ┌───────────┐  │
│  │  Fallback │  │───▶ FallbackStateMachine (if needed)
│  │   State   │  │
│  └───────────┘  │
└─────────────────┘
    │
    ▼ ExecutionResult
```

### 4.2 渐进加载数据流

```
Skill Discovery
    │
    ▼
┌─────────────────┐     ┌─────────────────┐
│  Discovery      │────▶│  Load Metadata  │
│  (file system)  │     │  (skill.yaml +   │
│                 │     │   frontmatter)   │
└─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  Register in    │
                        │  SkillRegistry  │
                        │  (lightweight)  │
                        └─────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  Routing      │   │  Context      │   │  Execution    │
    │  (triggers    │   │  Building     │   │  (full load)  │
    │   only)       │   │  (description │   │               │
    │                 │   │   + summary)  │   │               │
    └───────────────┘   └───────────────┘   └───────────────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │  Load Full    │
                                            │  Content      │
                                            │  (SKILL.md)   │
                                            └───────────────┘
                                                    │
                                                    ▼
                                            ┌───────────────┐
                                            │  Load Scripts │
                                            │  & References │
                                            │  (on demand)  │
                                            └───────────────┘
```

---

## 5. 集成点

### 5.1 与现有代码集成

```ruby
# lib/smart_bot/skill_system.rb
module SmartBot
  module SkillSystem
    class << self
      def configure
        @config ||= Configuration.new
        yield @config if block_given?
        @config
      end
      
      def registry
        @registry ||= SkillRegistry.new
      end
      
      def router
        @router ||= Router.new(registry: registry)
      end
      
      def executor
        @executor ||= SkillExecutor.new
      end
      
      # Main entry point for skill execution
      def execute(query, context: {})
        # Route query to skills
        plan = router.route(query: query, context: context)
        
        if plan.empty?
          return { success: false, error: "No matching skills found" }
        end
        
        # Execute plan
        executor.execute(plan, context: context)
      end
      
      # Load all skills
      def load_skills(workspace: nil)
        workspace ||= File.expand_path("~/.smart_bot/workspace")
        
        loader = UnifiedLoader.new(workspace: workspace)
        skills = loader.load_all
        
        skills.each { |skill| registry.register(skill) }
        
        registry.stats
      end
    end
  end
end
```

### 5.2 CLI 集成

```ruby
# lib/smart_bot/cli/skill_commands.rb
module SmartBot
  module CLI
    class SkillCommands
      desc "install SOURCE", "Install a skill from git/registry/path"
      def install(source)
        installer = SkillSystem::SkillInstaller.new
        result = installer.install(source)
        
        if result.success?
          say "Skill installed: #{result.skill_name}"
        else
          say_error "Failed to install: #{result.error}"
        end
      end
      
      desc "list", "List installed skills"
      def list
        registry = SkillSystem.registry
        
        say "Installed Skills:"
        registry.each do |skill|
          status = skill.available? ? "✓" : "✗"
          say "  #{status} #{skill.name} (#{skill.metadata.version})"
          say "     #{skill.metadata.description}"
        end
      end
      
      desc "run SKILL_NAME TASK", "Run a specific skill"
      def run(skill_name, task)
        result = SkillSystem.execute(
          "$#{skill_name} #{task}",
          context: { workspace: Dir.pwd }
        )
        
        say result.value if result.success?
        say_error result.error if result.failure?
      end
    end
  end
end
```

### 5.3 Agent Loop 集成

```ruby
# lib/smart_bot/agent/loop.rb
class AgentLoop
  def initialize
    @skill_system = SmartBot::SkillSystem
    @skill_system.load_skills
  end
  
  def process_message(message, context: {})
    # Check if this should be routed to skills
    plan = @skill_system.router.route(
      query: message,
      context: context,
      history: @conversation.history
    )
    
    if plan.primary_skill && plan.skills.first.score > 0.8
      # High confidence - execute skill directly
      result = @skill_system.executor.execute(plan, context: context)
      format_skill_result(result)
    else
      # Low confidence or no match - use regular agent processing
      process_with_tools(message, context)
    end
  end
  
  private
  
  def format_skill_result(result)
    if result.success?
      result.primary_result.value
    else
      "I encountered an issue: #{result.primary_result.error}"
    end
  end
end
```

---

## 6. 迁移路径

### 6.1 向后兼容策略

```ruby
# Legacy adapter for existing skills
class LegacySkillAdapter
  def self.wrap_legacy_skills
    # Convert existing SmartBot::Skill definitions to SkillPackages
    SmartBot::Skill.registry.each do |name, definition|
      package = SkillPackage.new(
        name: name.to_s,
        source_path: definition.config[:skill_path] || "builtin",
        metadata: extract_metadata(definition),
        content: nil,
        type: :ruby_native
      )
      
      SmartBot::SkillSystem.registry.register(package)
    end
  end
end
```

### 6.2 渐进迁移计划

1. **Phase 1**: 添加新的 SkillSystem 模块，保持现有代码不变
2. **Phase 2**: 迁移 UnifiedSkillLoader 使用新的 Loader
3. **Phase 3**: 添加 Router 和 Executor，作为可选路径
4. **Phase 4**: 更新 Agent Loop 使用新的路由系统
5. **Phase 5**: 弃用旧的 Skill 注册方式

---

## 7. 配置

```yaml
# config/smart_bot.yml
skill_system:
  # Routing
  semantic_top_k: 3
  selection_threshold: 0.65
  max_parallel_skills: 2
  
  # Execution
  max_retry_per_path: 1
  max_skill_delegate_depth: 2
  default_timeout: 120
  
  # Repair
  max_repair_attempts_per_run: 2
  max_patched_files_per_attempt: 3
  
  # Discovery
  discovery_paths:
    - "%{workspace}/skills"
    - "%{home}/.agents/skills"
  
  # Safety
  require_signed_skills: false
  default_sandbox: process
```

---

## 8. 总结

这个架构设计：

1. **统一了两种 spec**: 通过 `SkillPackage` 和 `SkillMetadata` 统一表示不同格式的 Skill
2. **实现了智能路由**: 两阶段路由（召回 + 打分）自动匹配最合适的 Skill
3. **提供了安全执行**: 沙箱和权限控制确保执行安全
4. **支持自我修复**: RepairLoop 和 FallbackStateMachine 处理失败情况
5. **保持向后兼容**: 通过 Adapter 层支持现有 Ruby Skills 和 Markdown Skills
6. **渐进加载**: 减少上下文开销，提高性能

核心优势：
- **清晰的关注点分离**: 路由、执行、适配各司其职
- **可扩展的设计**: 新的 Skill 格式可以通过添加 Adapter 支持
- **可观测性**: 全链路事件追踪便于调试和优化
- **类型安全**: 使用 Value Objects 和 Result Types 减少错误
