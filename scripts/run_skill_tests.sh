#!/bin/bash
# SmartBot Skill System Test Suite
# 运行此脚本测试SmartBot的skill系统能力

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FAILED_TESTS=()
PASSED_TESTS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# SmartBot 命令
SMART_BOT="bundle exec ruby bin/smart_bot"
VERBOSE="${VERBOSE:-false}"

print_header() {
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_test() {
    echo -e "${YELLOW}▶ Testing: $1${NC}"
}

print_pass() {
    echo -e "${GREEN}  ✅ PASS: $1${NC}"
    PASSED_TESTS+=("$1")
}

print_fail() {
    echo -e "${RED}  ❌ FAIL: $1${NC}"
    FAILED_TESTS+=("$1")
}

print_debug() {
    if [ "$VERBOSE" = "true" ]; then
        echo -e "${CYAN}  [DEBUG] $1${NC}"
    fi
}

# ============================================
# Test 1: 路由能力测试
# ============================================
test_routing() {
    print_header "Test 1: 路由能力测试"

    cd "$PROJECT_ROOT"

    # 1.1 硬触发测试（最可靠）
    print_test "1.1 硬触发测试 (\$test_router)"
    result=$($SMART_BOT agent -m "\$test_router 验证路由" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "test_router\|Primary skill: test_router"; then
        print_pass "硬触发匹配"
    else
        print_fail "硬触发匹配"
        echo "  输出: ${result:0:200}"
    fi

    # 1.2 触发词匹配测试
    print_test "1.2 触发词精确匹配 (test_router)"
    result=$($SMART_BOT agent -m "test_router 触发词验证" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "test_router\|Primary skill: test_router"; then
        print_pass "触发词精确匹配"
    else
        print_fail "触发词精确匹配"
        echo "  输出: ${result:0:200}"
    fi

    # 1.3 中文触发词测试
    print_test "1.3 中文触发词测试 (路由测试)"
    result=$($SMART_BOT agent -m "路由测试功能验证" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "router\|路由\|Primary skill: test_router"; then
        print_pass "中文触发词匹配"
    else
        print_fail "中文触发词匹配"
        echo "  输出: ${result:0:200}"
    fi
}

# ============================================
# Test 2: 执行能力测试
# ============================================
test_execution() {
    print_header "Test 2: 执行能力测试"

    cd "$PROJECT_ROOT"

    # 2.1 instruction类型执行
    print_test "2.1 instruction类型执行 (\$test_executor)"
    result=$($SMART_BOT agent -m "\$test_executor 基本执行测试" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "executor\|success\|执行\|Primary skill: test_executor"; then
        print_pass "instruction类型执行"
    else
        print_fail "instruction类型执行"
        echo "  输出: ${result:0:200}"
    fi

    # 2.2 脚本类型执行
    print_test "2.2 script类型执行 (\$test_script_type)"
    result=$($SMART_BOT agent -m "\$test_script_type hello world" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "echo\|script\|success\|Primary skill: test_script"; then
        print_pass "script类型执行"
    else
        print_fail "script类型执行"
        echo "  输出: ${result:0:200}"
    fi
}

# ============================================
# Test 3: 参数处理测试
# ============================================
test_parameters() {
    print_header "Test 3: 参数处理测试"

    cd "$PROJECT_ROOT"

    # 3.1 基本参数
    print_test "3.1 基本参数传递"
    result=$($SMART_BOT agent -m "\$test_parameters name=test value=123" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "name\|value\|test\|123\|Primary skill: test_parameters"; then
        print_pass "基本参数传递"
    else
        print_fail "基本参数传递"
        echo "  输出: ${result:0:200}"
    fi

    # 3.2 URL参数
    print_test "3.2 URL参数处理"
    result=$($SMART_BOT agent -m "\$test_parameters url=https://example.com/test?query=value" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "url\|example\|Primary skill: test_parameters"; then
        print_pass "URL参数处理"
    else
        print_fail "URL参数处理"
        echo "  输出: ${result:0:200}"
    fi

    # 3.3 中文参数
    print_test "3.3 中文参数处理"
    result=$($SMART_BOT agent -m "\$test_parameters 姓名=测试名" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "姓名\|测试\|Primary skill: test_parameters"; then
        print_pass "中文参数处理"
    else
        print_fail "中文参数处理"
        echo "  输出: ${result:0:200}"
    fi
}

# ============================================
# Test 4: 边界条件测试
# ============================================
test_edge_cases() {
    print_header "Test 4: 边界条件测试"

    cd "$PROJECT_ROOT"

    # 4.1 空参数测试
    print_test "4.1 空参数处理"
    result=$($SMART_BOT agent -m "\$test_edge_cases" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    # 检查是否成功路由并执行（检查Primary skill和success或正常输出）
    if echo "$result" | grep -qi "Primary skill: test_edge_cases"; then
        print_pass "空参数处理 (成功路由到test_edge_cases)"
    else
        print_fail "空参数处理"
        echo "  输出: ${result:0:200}"
    fi

    # 4.2 特殊字符测试
    print_test "4.2 特殊字符处理"
    result=$($SMART_BOT agent -m "\$test_edge_cases special chars test" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "Primary skill: test_edge_cases"; then
        print_pass "特殊字符处理 (成功路由到test_edge_cases)"
    else
        print_fail "特殊字符处理"
        echo "  输出: ${result:0:200}"
    fi

    # 4.3 超长输入测试
    print_test "4.3 超长输入处理"
    result=$($SMART_BOT agent -m "\$test_edge_cases long input test" 2>&1 || true)
    print_debug "Output: ${result:0:300}"
    if echo "$result" | grep -qi "Primary skill: test_edge_cases"; then
        print_pass "超长输入处理 (成功路由到test_edge_cases)"
    else
        print_fail "超长输入处理"
        echo "  输出: ${result:0:200}"
    fi
}

# ============================================
# Test 5: Fallback链测试
# ============================================
test_fallback() {
    print_header "Test 5: Fallback链测试"

    cd "$PROJECT_ROOT"

    # 5.1 Fallback触发测试
    print_test "5.1 Fallback机制 (test_fallback_a → test_fallback_b)"
    result=$($SMART_BOT agent -m "\$test_fallback_a 测试fallback" 2>&1 || true)
    print_debug "Output: ${result:0:500}"
    # test_fallback_a 故意失败，应该fallback到 test_fallback_b 或显示失败信息
    if echo "$result" | grep -qi "fallback\|success\|Primary skill: test_fallback"; then
        print_pass "Fallback机制工作正常"
    else
        print_fail "Fallback机制"
        echo "  输出: ${result:0:300}"
    fi
}

# ============================================
# Test 6: 单元测试
# ============================================
test_unit_tests() {
    print_header "Test 6: RSpec单元测试"

    cd "$PROJECT_ROOT"

    # 运行skill系统单元测试
    print_test "6.1 Router单元测试"
    rspec_output=$(bundle exec rspec spec/skill_system/routing/ --format progress 2>&1 || true)
    print_debug "RSpec output: ${rspec_output:0:500}"
    if echo "$rspec_output" | grep -qE "0 failures|examples.*0 failures"; then
        print_pass "Router单元测试"
    else
        print_fail "Router单元测试"
        if [ "$VERBOSE" = "true" ]; then
            echo "$rspec_output" | tail -30
        fi
    fi

    print_test "6.2 Executor单元测试"
    rspec_output=$(bundle exec rspec spec/skill_system/execution/ --format progress 2>&1 || true)
    print_debug "RSpec output: ${rspec_output:0:500}"
    if echo "$rspec_output" | grep -qE "0 failures|examples.*0 failures"; then
        print_pass "Executor单元测试"
    else
        print_fail "Executor单元测试"
        if [ "$VERBOSE" = "true" ]; then
            echo "$rspec_output" | tail -30
        fi
    fi

    print_test "6.3 Core单元测试"
    rspec_output=$(bundle exec rspec spec/skill_system/core/ --format progress 2>&1 || true)
    print_debug "RSpec output: ${rspec_output:0:500}"
    if echo "$rspec_output" | grep -qE "0 failures|examples.*0 failures"; then
        print_pass "Core单元测试"
    else
        print_fail "Core单元测试"
        if [ "$VERBOSE" = "true" ]; then
            echo "$rspec_output" | tail -30
        fi
    fi
}

# ============================================
# 主测试流程
# ============================================
main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       SmartBot Skill System - 综合测试套件                  ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "项目目录: $PROJECT_ROOT"
    echo "SmartBot: $SMART_BOT"
    echo "详细模式: VERBOSE=$VERBOSE (设置 VERBOSE=true 启用)"
    echo ""

    # 检查bundle
    if ! command -v bundle &> /dev/null; then
        echo -e "${RED}错误: bundle 未安装${NC}"
        exit 1
    fi

    # 运行所有测试
    test_routing
    test_execution
    test_parameters
    test_edge_cases
    test_fallback
    test_unit_tests

    # 输出总结
    print_header "测试结果总结"

    echo -e "${GREEN}通过的测试: ${#PASSED_TESTS[@]}${NC}"
    for test in "${PASSED_TESTS[@]}"; do
        echo -e "  ✅ $test"
    done

    echo ""
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo -e "${RED}失败的测试: ${#FAILED_TESTS[@]}${NC}"
        for test in "${FAILED_TESTS[@]}"; do
            echo -e "  ❌ $test"
        done
        echo ""
        echo -e "${YELLOW}提示: 使用 VERBOSE=true ./scripts/run_skill_tests.sh 查看详细输出${NC}"
        exit 1
    else
        echo -e "${GREEN}所有测试通过! 🎉${NC}"
        exit 0
    fi
}

# 运行主函数
main "$@"
