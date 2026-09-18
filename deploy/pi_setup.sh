#!/usr/bin/env bash
# Raspberry Pi 5 setup for the ADTS onboard tracker, from a clean Raspberry Pi OS
# Bookworm (64-bit) install to a service that's installed but not yet enabled.
# Safe to re-run: every step checks what's already done and skips it.
#
# Usage (from the repo root, on the Pi):
#     sudo deploy/pi_setup.sh [options]
#
# Options:
#   -y, --yes                assume "yes" to every confirmation (for unattended runs)
#   -n, --dry-run             print what would be done; change nothing
#       --user <name>         install for this user instead of auto-detecting one
#       --skip-reboot-prompt  don't ask to reboot at the end
#       --no-color            disable coloured output
#       --verify               skip setup; just check the hardware (run this after reboot)
#   -h, --help                show this help
#
# See docs/DEPLOY_PI.md for the full walkthrough, including what happens after this
# script (compiling the model, first run by hand, enabling the service).
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
ASSUME_YES=0
DRY_RUN=0
SKIP_REBOOT_PROMPT=0
FORCE_NO_COLOR=0
VERIFY_MODE=0
SHOW_HELP=0
USER_OVERRIDE=""
HAILO_INSTALLED=0
STEP_NUM=0
STEP_TOTAL=8
LOG_FILE="/var/log/adts-setup.log"
declare -a SUMMARY_NAMES=()
declare -a SUMMARY_STATUS=()
declare -a TMP_FILES=()

cleanup() {
    local f
    for f in "${TMP_FILES[@]}"; do
        if [[ -n "$f" && -e "$f" ]]; then rm -f "$f"; fi
    done
    return 0
}
trap cleanup EXIT

on_error() {
    local line=$1
    printf '\n%s%s%s Setup failed at line %s.\n' "${C_RED:-}" "${ICON_ERR:-x}" "${C_RESET:-}" "$line"
    if [[ -f "$LOG_FILE" ]]; then echo "See $LOG_FILE for the full log."; fi
    echo "Re-run the script — completed steps are skipped, so it picks up where it left off."
    exit 1
}
trap 'on_error $LINENO' ERR

# ---------------------------------------------------------------------------
# Terminal UI helpers
# ---------------------------------------------------------------------------
setup_ui() {
    if [[ -t 1 ]]; then IS_TTY=1; else IS_TTY=0; fi

    if (( IS_TTY )) && (( ! FORCE_NO_COLOR )) && [[ -z "${NO_COLOR:-}" ]] \
        && command -v tput >/dev/null 2>&1 && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
        C_RED=$(tput setaf 1); C_GREEN=$(tput setaf 2); C_YELLOW=$(tput setaf 3)
        C_BLUE=$(tput setaf 4); C_BOLD=$(tput bold); C_DIM=$(tput dim); C_RESET=$(tput sgr0)
    else
        C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""; C_DIM=""; C_RESET=""
    fi

    if [[ "${LANG:-}${LC_ALL:-}" == *UTF-8* || "${LANG:-}${LC_ALL:-}" == *utf8* ]]; then
        ICON_OK="✓"; ICON_WARN="⚠"; ICON_ERR="✗"
        SPIN_FRAMES=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    else
        ICON_OK="OK"; ICON_WARN="!!"; ICON_ERR="XX"
        SPIN_FRAMES=('|' '/' '-' '\')
    fi
}

section() {
    STEP_NUM=$((STEP_NUM + 1))
    printf '\n%s[%d/%d] %s%s\n' "${C_BOLD}${C_BLUE}" "$STEP_NUM" "$STEP_TOTAL" "$1" "$C_RESET"
}
ok()   { printf '  %s%s%s %s\n' "$C_GREEN" "$ICON_OK" "$C_RESET" "$1"; }
warn() { printf '  %s%s%s %s\n' "$C_YELLOW" "$ICON_WARN" "$C_RESET" "$1"; }
err()  { printf '  %s%s%s %s\n' "$C_RED" "$ICON_ERR" "$C_RESET" "$1"; }
info() { printf '  %s%s%s\n' "$C_DIM" "$1" "$C_RESET"; }
die()  { err "$1"; exit 1; }
record() { SUMMARY_NAMES+=("$1"); SUMMARY_STATUS+=("$2"); }

# y/n prompt. Always "yes" under --yes. Reads from the controlling terminal
# directly so it still works with stdin redirected.
confirm() {
    local prompt=$1 default=${2:-Y} suffix ans
    (( ASSUME_YES )) && return 0
    if [[ "$default" == "Y" ]]; then suffix="[Y/n]"; else suffix="[y/N]"; fi
    if [[ -r /dev/tty ]]; then
        read -r -p "  ${prompt} ${suffix} " ans < /dev/tty || ans=""
    else
        warn "No terminal to ask '$prompt' — assuming default ($default)."
        ans=""
    fi
    ans=${ans:-$default}
    [[ "$ans" =~ ^[Yy] ]]
}

# Gate for anything that mutates the system: no-ops and prints under --dry-run.
run() {
    if (( DRY_RUN )); then
        printf '  %s[dry-run] would run:%s %s\n' "$C_DIM" "$C_RESET" "$*"
        return 0
    fi
    "$@"
}

# Runs a (possibly slow) command with a spinner, logging its output and
# showing just the tail on failure instead of drowning the terminal.
run_spin() {
    local desc=$1; shift
    if (( DRY_RUN )); then
        printf '  %s[dry-run] would run:%s %s (%s)\n' "$C_DIM" "$C_RESET" "$desc" "$*"
        return 0
    fi
    local logf; logf=$(mktemp)
    TMP_FILES+=("$logf")
    "$@" >"$logf" 2>&1 &
    local pid=$!
    if (( IS_TTY )); then
        local i=0
        while kill -0 "$pid" 2>/dev/null; do
            printf '\r\033[K  %s %s' "${SPIN_FRAMES[$((i % ${#SPIN_FRAMES[@]}))]}" "$desc" > /dev/tty 2>/dev/null || true
            i=$((i + 1))
            sleep 0.1
        done
        printf '\r\033[K' > /dev/tty 2>/dev/null || true
    fi
    local status
    if wait "$pid"; then status=0; else status=$?; fi
    if (( status == 0 )); then
        ok "$desc"
        cat "$logf" >> "$LOG_FILE" 2>/dev/null || true
        return 0
    else
        err "$desc failed (exit $status)"
        echo "  last output:"
        tail -n 20 "$logf" | sed 's/^/    /'
        return "$status"
    fi
}

# Retries a flaky command (apt/network) with a growing delay between tries.
retry() {
    local attempts=$1 delay=$2; shift 2
    local n=1
    until "$@"; do
        if (( n >= attempts )); then return 1; fi
        n=$((n + 1))
        sleep "$delay"
    done
}

show_help() {
    sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -y|--yes) ASSUME_YES=1 ;;
            -n|--dry-run) DRY_RUN=1 ;;
            --skip-reboot-prompt) SKIP_REBOOT_PROMPT=1 ;;
            --no-color) FORCE_NO_COLOR=1 ;;
            --user) shift; USER_OVERRIDE=${1:-}; [[ -n "$USER_OVERRIDE" ]] || { echo "--user needs a value" >&2; exit 1; } ;;
            --verify) VERIFY_MODE=1 ;;
            -h|--help) SHOW_HELP=1 ;;
            *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
        esac
        shift
    done
}

require_root_or_reexec() {
    if [[ $EUID -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            die "This needs root and 'sudo' isn't installed. Run it as root instead."
        fi
        echo "Root is needed for package installs and system config — re-running with sudo."
        exec sudo -E -- "${BASH_SOURCE[0]}" "${ORIG_ARGS[@]}"
    fi
}

resolve_repo_and_user() {
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    for f in requirements-pi.txt deploy/adts.service deploy/adts.env; do
        [[ -f "$REPO/$f" ]] || die "Expected $f under $REPO — is this the adts repo root? Clone it and re-run from there."
    done

    RUN_USER="${USER_OVERRIDE:-${SUDO_USER:-}}"
    if [[ -z "$RUN_USER" || "$RUN_USER" == "root" ]]; then
        RUN_USER=$(stat -c '%U' "$REPO" 2>/dev/null || true)
    fi
    if [[ -z "$RUN_USER" || "$RUN_USER" == "root" ]]; then
        die "Can't tell which non-root user should own the venv and run the service. Re-run with: sudo deploy/pi_setup.sh --user <name>"
    fi
    id -u "$RUN_USER" >/dev/null 2>&1 || die "User '$RUN_USER' doesn't exist. Pass the right one with --user <name>."

    if [[ -f /boot/firmware/config.txt ]]; then
        CONFIG=/boot/firmware/config.txt
    elif [[ -f /boot/config.txt ]]; then
        CONFIG=/boot/config.txt
    else
        die "Can't find config.txt under /boot/firmware or /boot — is this really Raspberry Pi OS?"
    fi
}

banner() {
    printf '\n%s== ADTS — Raspberry Pi setup ==%s\n' "${C_BOLD}${C_BLUE}" "$C_RESET"
    printf '  repo: %s\n' "$REPO"
    printf '  user: %s\n' "$RUN_USER"
    printf '  log:  %s\n' "$LOG_FILE"
    if (( DRY_RUN )); then printf '  %s[dry-run: no changes will be made]%s\n' "$C_YELLOW" "$C_RESET"; fi
    return 0
}

setup_logging() {
    mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    if (( ! DRY_RUN )); then
        : > "$LOG_FILE" 2>/dev/null || LOG_FILE=$(mktemp)
        exec > >(tee -a "$LOG_FILE") 2>&1
    fi
}

step_preflight() {
    section "Preflight checks"

    local arch; arch=$(uname -m)
    if [[ "$arch" != "aarch64" ]]; then
        die "Detected $arch. ADTS needs 64-bit Raspberry Pi OS (aarch64) — Hailo and the camera stack aren't available on 32-bit. Reflash with Raspberry Pi Imager: 'Raspberry Pi OS Lite (64-bit)'."
    fi
    ok "64-bit OS ($arch)"

    local model=""
    [[ -r /proc/device-tree/model ]] && model=$(tr -d '\0' < /proc/device-tree/model)
    if [[ "$model" == *"Raspberry Pi 5"* ]]; then
        ok "Board: $model"
    else
        warn "Board reports '${model:-unknown}', not a Raspberry Pi 5."
        warn "The PCIe/UART tuning below targets the Pi 5; the Hailo-8L AI HAT+/M.2 kit is Pi 5-only."
        confirm "Continue anyway?" N || die "Aborted."
        record "Board check" "WARN"
    fi

    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        if [[ "${VERSION_CODENAME:-}" == "bookworm" ]]; then
            ok "OS: ${PRETTY_NAME:-Raspberry Pi OS Bookworm}"
        else
            warn "Detected ${PRETTY_NAME:-an unrecognised OS} — this script targets Raspberry Pi OS Bookworm. Package names or paths may differ."
            confirm "Continue anyway?" N || die "Aborted."
            record "OS check" "WARN"
        fi
    fi
    ok "Boot config: $CONFIG"

    info "Checking network..."
    local net_ok=0
    if command -v curl >/dev/null 2>&1; then
        retry 3 5 curl -fsS --max-time 5 -o /dev/null https://archive.raspberrypi.com && net_ok=1
    elif command -v wget >/dev/null 2>&1; then
        retry 3 5 wget -q --timeout=5 -O /dev/null https://archive.raspberrypi.com && net_ok=1
    else
        retry 3 5 ping -c1 -W3 1.1.1.1 >/dev/null 2>&1 && net_ok=1
    fi
    if (( net_ok )); then
        ok "Network reachable"
    else
        warn "Couldn't reach the network — apt-get will likely fail."
        confirm "Continue anyway?" N || die "Connect via 'sudo raspi-config' -> System Options -> Wireless LAN (or plug in Ethernet), then re-run."
        record "Network check" "WARN"
    fi

    local avail_mb; avail_mb=$(( $(df --output=avail "$REPO" | tail -1) / 1024 ))
    if (( avail_mb < 2048 )); then
        warn "Only ${avail_mb} MiB free — apt packages plus the venv need roughly 2 GiB."
        confirm "Continue anyway?" N || die "Free up space and re-run."
        record "Disk space" "WARN"
    else
        ok "Disk space: ${avail_mb} MiB free"
    fi

    record "Preflight checks" "OK"
}

step_packages() {
    section "Packages"
    run_spin "apt-get update" retry 3 10 apt-get update -qq || die "apt-get update failed repeatedly — check the network and try again."

    local core_pkgs=(python3-picamera2 python3-opencv python3-scipy python3-numpy python3-venv
        gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly)

    if apt-cache show hailo-all >/dev/null 2>&1; then
        if run_spin "Installing packages incl. hailo-all (downloads firmware, can take a while)" \
            retry 2 10 apt-get install -y hailo-all "${core_pkgs[@]}"; then
            HAILO_INSTALLED=1
            record "Packages (incl. Hailo)" "OK"
        else
            warn "hailo-all failed to install. Retrying without it so the rest of setup can finish."
            run_spin "Installing packages (without hailo-all)" retry 2 10 apt-get install -y "${core_pkgs[@]}" \
                || die "Package install failed — see the log above."
            record "Packages" "OK"
            record "Hailo packages" "FAIL"
        fi
    else
        warn "hailo-all isn't available from apt. Either the Raspberry Pi apt repo isn't configured, or this OS image predates it."
        warn "Try: sudo apt-get update --allow-releaseinfo-change, or reflash with the latest Raspberry Pi Imager, then re-run this script."
        confirm "Continue installing the rest (camera/gstreamer/venv) without Hailo support?" Y || die "Aborted."
        run_spin "Installing packages (without hailo-all)" retry 2 10 apt-get install -y "${core_pkgs[@]}" \
            || die "Package install failed — see the log above."
        record "Packages" "OK"
        record "Hailo packages" "SKIP"
    fi
}

config_add_line() {
    if grep -qxF "$1" "$CONFIG" 2>/dev/null; then
        info "already set: $1"
    else
        run bash -c "printf '%s\n' \"\$1\" >> \"\$2\"" _ "$1" "$CONFIG"
        ok "added: $1"
    fi
}

step_config_txt() {
    section "config.txt ($CONFIG)"
    if ! grep -q '# --- ADTS ---' "$CONFIG" 2>/dev/null; then
        local backup="${CONFIG}.adts-bak-$(date +%Y%m%d%H%M%S)"
        run cp "$CONFIG" "$backup"
        info "Backed up to $(basename "$backup")"
    fi
    config_add_line "# --- ADTS ---"
    config_add_line "dtparam=pciex1_gen=3"   # PCIe Gen3 to the Hailo (default is Gen2)
    config_add_line "dtparam=uart0=on"       # GPIO14/15 UART -> /dev/ttyAMA0, to the flight controller
    config_add_line "enable_uart=1"
    record "config.txt" "OK"
}

step_serial() {
    section "Serial port (UART to flight controller)"
    if ! command -v raspi-config >/dev/null 2>&1; then
        warn "raspi-config not found — skipping automatic serial configuration."
        warn "By hand: Interface Options -> Serial Port -> login shell over serial: No, serial port hardware: Yes."
        record "Serial config" "SKIP"
        return
    fi
    run raspi-config nonint do_serial_cons 1   # login console off — it would fight MAVLink for the port
    run raspi-config nonint do_serial_hw 0     # hardware UART on
    ok "Console disabled on UART, hardware UART enabled"
    record "Serial config" "OK"
}

step_boot_target() {
    section "Boot target"
    local current; current=$(systemctl get-default 2>/dev/null || echo "unknown")
    if [[ "$current" == "multi-user.target" ]]; then
        ok "Already boots to console (no desktop)"
        record "Boot target" "SKIP"
        return
    fi
    warn "Currently boots to '$current'. ADTS needs the HDMI output to itself (kmssink), so no desktop can be running."
    if confirm "Set boot target to console-only (multi-user.target)?" Y; then
        run systemctl set-default multi-user.target
        ok "Boot target set to multi-user.target (takes effect next boot)"
        record "Boot target" "OK"
    else
        warn "Left as '$current' — the service may fail to get the display. Change later with: sudo systemctl set-default multi-user.target"
        record "Boot target" "SKIP"
    fi
}

step_groups() {
    section "User groups ($RUN_USER)"
    local needed=(video render dialout) to_add=() g
    for g in "${needed[@]}"; do
        if id -nG "$RUN_USER" 2>/dev/null | tr ' ' '\n' | grep -qx "$g"; then
            info "already in group: $g"
        else
            to_add+=("$g")
        fi
    done
    if (( ${#to_add[@]} )); then
        run usermod -aG "$(IFS=,; echo "${to_add[*]}")" "$RUN_USER"
        ok "Added $RUN_USER to: ${to_add[*]}"
    else
        ok "Already in all required groups"
    fi
    info "Group membership applies after $RUN_USER logs in again (the reboot at the end covers it)."
    record "User groups" "OK"
}

step_venv() {
    section "Python virtualenv"
    local venv="$REPO/.venv"
    if [[ -x "$venv/bin/python" ]]; then
        info "venv already exists at $venv"
        if confirm "Recreate it from scratch?" N; then
            run rm -rf "$venv"
        fi
    fi
    if [[ ! -x "$venv/bin/python" ]]; then
        run sudo -u "$RUN_USER" python3 -m venv --system-site-packages "$venv"
        ok "Created venv"
    fi
    run_spin "Installing pip requirements" retry 3 5 sudo -u "$RUN_USER" "$venv/bin/pip" install -q -r "$REPO/requirements-pi.txt" \
        || die "pip install failed — see the log above. Check the network and re-run."
    record "Python venv" "OK"
}

step_service() {
    section "systemd service"
    local rendered; rendered=$(mktemp)
    TMP_FILES+=("$rendered")
    sed -e "s#@REPO@#$REPO#g" -e "s#@USER@#$RUN_USER#g" "$REPO/deploy/adts.service" > "$rendered"
    local target=/etc/systemd/system/adts.service

    if [[ -f "$target" ]] && cmp -s "$rendered" "$target"; then
        info "Service file already up to date"
        record "systemd service" "SKIP"
    else
        if [[ -f "$target" ]]; then
            warn "adts.service already exists and differs from the one this repo would install."
            if ! confirm "Overwrite it?" Y; then
                warn "Left the existing service file in place."
                record "systemd service" "SKIP"
                return
            fi
            run cp "$target" "${target}.adts-bak-$(date +%Y%m%d%H%M%S)"
        fi
        run cp "$rendered" "$target"
        ok "Installed $target"
        record "systemd service" "OK"
    fi

    if [[ -f /etc/default/adts ]]; then
        info "/etc/default/adts already exists — leaving your edits in place"
    else
        run cp "$REPO/deploy/adts.env" /etc/default/adts
        ok "Installed default /etc/default/adts (edit ADTS_ARGS before enabling)"
    fi
    if ! run systemctl daemon-reload; then
        warn "systemctl daemon-reload failed — is systemd running as PID 1? (this step doesn't apply inside most containers)"
    fi
    return 0
}

print_summary_table() {
    local i name status
    for i in "${!SUMMARY_NAMES[@]}"; do
        name="${SUMMARY_NAMES[$i]}"; status="${SUMMARY_STATUS[$i]}"
        case "$status" in
            OK)   printf '  %s%s%s %s\n' "$C_GREEN" "$ICON_OK" "$C_RESET" "$name" ;;
            SKIP) printf '  %s--%s %s (skipped)\n' "$C_DIM" "$C_RESET" "$name" ;;
            WARN) printf '  %s%s%s %s (warning)\n' "$C_YELLOW" "$ICON_WARN" "$C_RESET" "$name" ;;
            FAIL) printf '  %s%s%s %s (failed)\n' "$C_RED" "$ICON_ERR" "$C_RESET" "$name" ;;
        esac
    done
}

final_summary() {
    printf '\n%s== Summary ==%s\n' "${C_BOLD}${C_BLUE}" "$C_RESET"
    print_summary_table

    local hef_count; hef_count=$(find "$REPO/models" -maxdepth 1 -name '*.hef' 2>/dev/null | wc -l)
    echo
    if (( hef_count == 0 )); then
        warn "No .hef model found in $REPO/models/ — the service will fail to start until you compile one and copy it there (docs/DEPLOY_PI.md, sections 3-4)."
    else
        ok "$hef_count .hef model(s) present in $REPO/models/"
    fi
    if (( ! HAILO_INSTALLED )); then
        warn "hailo-all wasn't installed — detection won't run until you install it and re-run this script."
    fi

    echo
    echo "Next steps:"
    echo "  1. Reboot to apply config.txt, serial and group changes."
    echo "  2. Verify hardware:   deploy/pi_setup.sh --verify"
    echo "  3. Test by hand:      docs/DEPLOY_PI.md, section 5"
    echo "  4. Then enable:       sudo systemctl enable --now adts"

    if (( DRY_RUN )); then
        echo
        warn "This was a dry run — nothing was changed."
        return
    fi
    if (( SKIP_REBOOT_PROMPT )); then
        warn "Remember to reboot before testing: sudo reboot"
        return
    fi
    if confirm "Reboot now?" Y; then
        ok "Rebooting..."
        run reboot || warn "reboot failed — reboot manually: sudo reboot"
    else
        warn "Remember to reboot before testing: sudo reboot"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# --verify: post-reboot hardware checks (no root required)
# ---------------------------------------------------------------------------
run_verify() {
    printf '\n%s== ADTS — hardware verification ==%s\n\n' "${C_BOLD}${C_BLUE}" "$C_RESET"
    local fail=0 out

    if command -v hailortcli >/dev/null 2>&1; then
        if out=$(hailortcli fw-control identify 2>&1) && [[ "$out" == *"HAILO8L"* ]]; then
            ok "Hailo-8L detected"
        else
            err "Hailo not detected as HAILO8L"
            echo "$out" | sed 's/^/    /'
            fail=1
        fi
    else
        err "hailortcli not found — install hailo-all: sudo apt-get install hailo-all"
        fail=1
    fi

    if command -v rpicam-hello >/dev/null 2>&1; then
        if out=$(rpicam-hello --list-cameras 2>&1) && [[ "$out" != *"No cameras"* ]]; then
            ok "Camera detected"
        else
            err "No camera detected — check the CSI ribbon cable (contacts toward the USB ports) and that it's seated in CAM0"
            echo "$out" | sed 's/^/    /'
            fail=1
        fi
    else
        err "rpicam-hello not found — install python3-picamera2"
        fail=1
    fi

    if [[ -c /dev/ttyAMA0 ]]; then
        ok "/dev/ttyAMA0 present"
    else
        err "/dev/ttyAMA0 missing — check dtparam=uart0=on in config.txt and that you rebooted after setup"
        fail=1
    fi

    if command -v vcgencmd >/dev/null 2>&1; then
        local t; t=$(vcgencmd get_throttled 2>&1)
        if [[ "$t" == "throttled=0x0" ]]; then
            ok "Power/thermal OK ($t)"
        else
            warn "Power/thermal flag set: $t — check the BEC and cooling (docs/DEPLOY_PI.md, section 1)"
        fi
    fi

    echo
    if (( fail )); then
        err "Some checks failed — see above."
    else
        ok "All checks passed."
    fi
    return "$fail"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    ORIG_ARGS=("$@")
    parse_args "$@"
    setup_ui

    if (( SHOW_HELP )); then
        show_help
        exit 0
    fi
    if (( VERIFY_MODE )); then
        if run_verify; then exit 0; else exit 1; fi
    fi

    (( DRY_RUN )) || require_root_or_reexec
    resolve_repo_and_user
    setup_logging
    banner

    step_preflight
    step_packages
    step_config_txt
    step_serial
    step_boot_target
    step_groups
    step_venv
    step_service
    final_summary
}

main "$@"
