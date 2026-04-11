#!/usr/bin/env bash
# bench/setup_perf.sh — V26 Phase C2.0.4 deliverable
#
# Mechanizes the CPU + memory configuration steps from
# bench/METHODOLOGY.md §3.1-§3.5 + §4.1-§4.2 so paper-grade benchmark
# runs are one command away from a clean default state.
#
# Usage:
#   bash bench/setup_perf.sh                # apply (sudo prompts internally)
#   bash bench/setup_perf.sh apply          # same as above
#   bash bench/setup_perf.sh --check        # report current state, no changes (sudo-free)
#   bash bench/setup_perf.sh --restore      # undo: governor→powersave, turbo→on, HT→on, swap→on
#   bash bench/setup_perf.sh --help         # print this header
#
# What apply does (per METHODOLOGY.md):
#   §3.1 Governor → performance        (per-CPU scaling_governor)
#   §3.3 Intel Turbo boost → OFF       (intel_pstate/no_turbo = 1)
#   §3.5 HT siblings → OFFLINE         (echo 0 > sibling cpu's online)
#   §4.1 Drop page caches              (sync + drop_caches = 3)
#   §4.2 Swap → OFF                    (swapoff -a)
#
# Verification (per V26 plan §C2.0.4 task table):
#   cd ~/Documents/fajarquant && bash bench/setup_perf.sh && \
#       cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
#   → expected output: performance
#
# Safety: this script refuses to run if the CPU model doesn't match the
# expected hardware (Intel i9-14900HX from bench/hardware_snapshot.txt).
# Override with --force if you really know what you're doing.
#
# Idempotency: running apply twice is a no-op. running --check after apply
# reports the post-apply state.
#
# All privileged operations use `sudo` internally; you do NOT need to run
# this script as root. The script will prompt for sudo password ONCE
# (cached for the lifetime of the run) on the first privileged operation.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────
EXPECTED_CPU="Intel(R) Core(TM) i9-14900HX"
SCRIPT_NAME="bench/setup_perf.sh"
SCRIPT_VERSION="1.0 (V26 Phase C2.0.4)"

# ANSI colors (degrade gracefully if not a tty)
if [ -t 1 ]; then
    RED=$'\033[31m'
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    BLUE=$'\033[34m'
    BOLD=$'\033[1m'
    RESET=$'\033[0m'
else
    RED="" GREEN="" YELLOW="" BLUE="" BOLD="" RESET=""
fi

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
log()    { printf "%s[setup_perf]%s %s\n" "$BLUE" "$RESET" "$*"; }
warn()   { printf "%s[setup_perf]%s %s%s%s\n" "$BLUE" "$RESET" "$YELLOW" "$*" "$RESET"; }
ok()     { printf "%s[setup_perf]%s %s%s%s\n" "$BLUE" "$RESET" "$GREEN" "$*" "$RESET"; }
fail()   { printf "%s[setup_perf]%s %s%s%s\n" "$BLUE" "$RESET" "$RED" "$*" "$RESET" >&2; exit 1; }

require_sudo() {
    if ! sudo -n true 2>/dev/null; then
        log "Privileged operations require sudo. You'll be prompted now."
        sudo -v || fail "sudo authentication failed"
    fi
}

check_cpu_model() {
    local actual
    actual=$(grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //' | xargs)
    if [ "$actual" != "$EXPECTED_CPU" ]; then
        warn "CPU mismatch:"
        warn "  expected: $EXPECTED_CPU"
        warn "  actual:   $actual"
        warn "This script is tuned for the Lenovo Legion Pro test machine"
        warn "(see bench/hardware_snapshot.txt). Override with --force."
        if [ "${FORCE:-0}" != "1" ]; then
            fail "CPU mismatch (use --force to override)"
        fi
        warn "Continuing anyway because --force was set"
    fi
}

# ─────────────────────────────────────────────────────────────────
# §3.1 Governor pinning
# ─────────────────────────────────────────────────────────────────
set_governor() {
    local target=$1
    log "[1/5] Setting CPU governor to: ${BOLD}$target${RESET}"
    for cpu_gov in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
        if [ -w "$cpu_gov" ] || [ -e "$cpu_gov" ]; then
            echo "$target" | sudo tee "$cpu_gov" > /dev/null
        fi
    done
}

# ─────────────────────────────────────────────────────────────────
# §3.3 Intel Turbo boost
# ─────────────────────────────────────────────────────────────────
set_turbo() {
    local val=$1  # 0 = off (no_turbo=1), 1 = on (no_turbo=0)
    local label
    [ "$val" = "0" ] && label="OFF" || label="ON"
    log "[2/5] Setting Intel Turbo boost: ${BOLD}$label${RESET}"

    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        local no_turbo
        [ "$val" = "0" ] && no_turbo=1 || no_turbo=0
        echo "$no_turbo" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
    elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
        # AMD path (not used on this machine, but defensive)
        echo "$val" | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null
    else
        warn "  No turbo control file found, skipping"
    fi
}

# ─────────────────────────────────────────────────────────────────
# §3.5 HT siblings: take HT sibling logical CPUs offline
# ─────────────────────────────────────────────────────────────────
set_ht_siblings_offline() {
    log "[3/5] Taking HT sibling logical CPUs ${BOLD}OFFLINE${RESET}"
    local count=0
    for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
        local cpunum
        cpunum=$(basename "$cpu_dir" | sed 's/cpu//')
        # cpu0 cannot be taken offline on x86
        [ "$cpunum" = "0" ] && continue

        local siblings_file="$cpu_dir/topology/thread_siblings_list"
        [ -f "$siblings_file" ] || continue

        # thread_siblings_list looks like "12,16" — primary,sibling
        # We want to take the SECOND number offline (the sibling), not the primary
        local primary sibling
        primary=$(cat "$siblings_file" | cut -d, -f1)
        sibling=$(cat "$siblings_file" | cut -d, -f2)

        if [ -n "$sibling" ] && [ "$cpunum" = "$sibling" ] && [ "$primary" != "$sibling" ]; then
            echo 0 | sudo tee "$cpu_dir/online" > /dev/null
            count=$((count + 1))
        fi
    done
    log "       → $count HT siblings taken offline"
}

set_ht_siblings_online() {
    log "[3/5] Bringing HT sibling logical CPUs ${BOLD}ONLINE${RESET}"
    local count=0
    for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
        local cpunum
        cpunum=$(basename "$cpu_dir" | sed 's/cpu//')
        [ "$cpunum" = "0" ] && continue
        if [ -f "$cpu_dir/online" ]; then
            local current
            current=$(cat "$cpu_dir/online")
            if [ "$current" = "0" ]; then
                echo 1 | sudo tee "$cpu_dir/online" > /dev/null
                count=$((count + 1))
            fi
        fi
    done
    log "       → $count CPUs brought back online"
}

# ─────────────────────────────────────────────────────────────────
# §4.1 Drop page caches
# ─────────────────────────────────────────────────────────────────
drop_caches() {
    log "[4/5] Dropping page caches (sync + drop_caches=3)"
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
}

# ─────────────────────────────────────────────────────────────────
# §4.2 Swap
# ─────────────────────────────────────────────────────────────────
disable_swap() {
    log "[5/5] Disabling swap"
    sudo swapoff -a
}

enable_swap() {
    log "[5/5] Re-enabling swap"
    sudo swapon -a
}

# ─────────────────────────────────────────────────────────────────
# State reporter (sudo-free)
# ─────────────────────────────────────────────────────────────────
check_state() {
    printf "\n%s=== Current state ===%s\n" "$BOLD" "$RESET"
    printf "  CPU model:        %s\n" "$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //' | xargs)"
    printf "  Governor (cpu0):  %s%s%s\n" "$BOLD" "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" "$RESET"
    if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
        local nt
        nt=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
        local turbo_label
        [ "$nt" = "1" ] && turbo_label="OFF" || turbo_label="ON"
        printf "  Turbo boost:      %s%s%s (no_turbo=%s)\n" "$BOLD" "$turbo_label" "$RESET" "$nt"
    fi
    printf "  Online CPUs:      %s\n" "$(cat /sys/devices/system/cpu/online)"
    local online_count
    online_count=$(grep -c processor /proc/cpuinfo)
    printf "  Online count:     %s%s%s\n" "$BOLD" "$online_count" "$RESET"
    if command -v swapon >/dev/null; then
        local swap_count
        swap_count=$(swapon --show=NAME --noheadings 2>/dev/null | wc -l)
        printf "  Swap devices:     %s\n" "$swap_count"
    fi
    printf "  CPU max freq:     %s Hz\n" "$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)"
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq ]; then
        printf "  CPU cur freq:     %s Hz\n" "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
    fi
    printf "  Free RAM:         %s\n" "$(free -h | awk '/^Mem:/ {print $7}')"
    printf "\n"
}

# ─────────────────────────────────────────────────────────────────
# Top-level commands
# ─────────────────────────────────────────────────────────────────
apply_perf() {
    log "Applying paper-grade benchmark configuration..."
    log "(per bench/METHODOLOGY.md §3.1-§3.5 + §4.1-§4.2)"
    check_cpu_model
    require_sudo
    set_governor performance
    set_turbo 0
    set_ht_siblings_offline
    drop_caches
    disable_swap
    ok "Configuration applied. Run benchmarks now."
    log "Restore default with: bash $SCRIPT_NAME --restore"
    check_state
}

restore_default() {
    log "Restoring default configuration..."
    check_cpu_model
    require_sudo
    set_ht_siblings_online
    set_turbo 1
    set_governor powersave
    enable_swap
    ok "Restored. Re-apply before next benchmark run."
    check_state
}

print_help() {
    sed -n '2,25p' "$0" | sed 's/^# \?//'
}

main() {
    local cmd="${1:-apply}"
    case "$cmd" in
        apply|--apply)
            apply_perf
            ;;
        --check|check|status)
            check_state
            ;;
        --restore|restore|undo)
            restore_default
            ;;
        --help|-h|help)
            print_help
            ;;
        --version|-V)
            printf "%s %s\n" "$SCRIPT_NAME" "$SCRIPT_VERSION"
            ;;
        *)
            fail "Unknown option: $cmd. Use --help for usage."
            ;;
    esac
}

main "$@"
