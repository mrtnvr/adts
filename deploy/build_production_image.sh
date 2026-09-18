#!/usr/bin/env bash
# Turns an already-working ADTS install (deploy/pi_setup.sh has been run, the service has
# been tested by hand per docs/DEPLOY_PI.md) into a sealed "golden master" for a shipped
# unit: no SSH, no console, no login shell for anyone, only compiled bytecode on disk, and
# guided (not blind) EEPROM signed-boot + full-disk-encryption steps. See
# docs/PRODUCTION_HARDENING.md for the full walkthrough, the threat model this defends
# against, and what it deliberately does NOT defend against (see step 8's notes).
#
# Run this ONCE, on a bench unit, before it's cloned to every shipped device — not on your
# only dev board, and not on a unit you can't afford to re-flash if a step goes wrong.
# Several steps here are irreversible (EEPROM OTP writes, disk repartitioning) or remove
# your own way back in (SSH, console, login shells) — that's the point, but it also means
# there is no "undo": if this is your only bench unit, keep a second SD card image of it
# from BEFORE this script ran.
#
# Usage (from the repo root, on the Pi, after pi_setup.sh and a tested service):
#     sudo deploy/build_production_image.sh [options]
#
# Options:
#   -y, --yes                assume "yes" to every confirmation EXCEPT the initial typed
#                             "PROCEED" safety gate (that one always requires typing it out)
#   -n, --dry-run             print what would be done; change nothing
#       --service-user <name> dedicated system account the production service runs as
#                             (default: adtssvc; created if it doesn't exist)
#       --no-color            disable coloured output
#   -h, --help                show this help
#
# What this does NOT do (see docs/PRODUCTION_HARDENING.md for why, and the upgrade path):
#   - It does not add a hardware secure element. The LUKS key ends up embedded in the
#     signed initramfs, which stops casual/curious access but NOT someone who pulls the SD
#     card and reads the boot partition directly with a card reader - only a discrete
#     secure element closes that gap. Deferred by choice, not by accident.
#   - It does not set up remote/OTA updates. Updates are factory reflashes only.
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
ASSUME_YES=0
DRY_RUN=0
FORCE_NO_COLOR=0
SHOW_HELP=0
SERVICE_USER="adtssvc"
PROD_DIR="/opt/adts"
DATA_DIR="/var/lib/adts"
STEP_NUM=0
STEP_TOTAL=9
LOG_FILE="/var/log/adts-production-image.log"
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
    printf '\n%s%s%s Failed at line %s.\n' "${C_RED:-}" "${ICON_ERR:-x}" "${C_RESET:-}" "$line"
    if [[ -f "$LOG_FILE" ]]; then echo "See $LOG_FILE for the full log."; fi
    echo "Re-run the script — completed steps are skipped, so it picks up where it left off."
    echo "The two guided hardware steps (EEPROM signing, LUKS) are never re-attempted"
    echo "automatically; re-check their state by hand before re-running."
    exit 1
}
trap 'on_error $LINENO' ERR

# ---------------------------------------------------------------------------
# Terminal UI helpers (same as pi_setup.sh, kept in sync on purpose)
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

confirm() {
    local prompt=$1 default=${2:-Y} suffix ans
    (( ASSUME_YES )) && return 0
    if [[ "$default" == "Y" ]]; then suffix="[Y/n]"; else suffix="[y/N]"; fi
    # 2>/dev/null has to wrap the probe, not follow it on the same line as the real read,
    # or bash prints its own diagnostic for the failed open before that can suppress it.
    if { : < /dev/tty; } 2>/dev/null; then
        read -r -p "  ${prompt} ${suffix} " ans < /dev/tty || ans=""
    else
        warn "No terminal to ask '$prompt' — assuming default ($default)."
        ans=""
    fi
    ans=${ans:-$default}
    [[ "$ans" =~ ^[Yy] ]]
}

run() {
    if (( DRY_RUN )); then
        printf '  %s[dry-run] would run:%s %s\n' "$C_DIM" "$C_RESET" "$*"
        return 0
    fi
    "$@"
}

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

show_help() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# Argument parsing / preflight
# ---------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -y|--yes) ASSUME_YES=1 ;;
            -n|--dry-run) DRY_RUN=1 ;;
            --service-user) shift; SERVICE_USER=${1:-}; [[ -n "$SERVICE_USER" ]] || { echo "--service-user needs a value" >&2; exit 1; } ;;
            --no-color) FORCE_NO_COLOR=1 ;;
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
        echo "Root is needed for account/package/boot changes — re-running with sudo."
        exec sudo -E -- "${BASH_SOURCE[0]}" "${ORIG_ARGS[@]}"
    fi
}

resolve_repo_and_config() {
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    if [[ -f /boot/firmware/config.txt ]]; then
        CONFIG=/boot/firmware/config.txt
        CMDLINE=/boot/firmware/cmdline.txt
    elif [[ -f /boot/config.txt ]]; then
        CONFIG=/boot/config.txt
        CMDLINE=/boot/cmdline.txt
    else
        die "Can't find config.txt/cmdline.txt under /boot/firmware or /boot — is this really Raspberry Pi OS?"
    fi
    [[ -f "$CMDLINE" ]] || die "Expected $CMDLINE to exist alongside $CONFIG."
}

banner() {
    printf '\n%s== ADTS — production image hardening ==%s\n' "${C_BOLD}${C_BLUE}" "$C_RESET"
    printf '  repo:          %s\n' "$REPO"
    printf '  service user:  %s (new, dedicated - not your dev account)\n' "$SERVICE_USER"
    printf '  app tree:      %s (bytecode only, read-only)\n' "$PROD_DIR"
    printf '  data dir:      %s (recordings, the only writable app path)\n' "$DATA_DIR"
    printf '  log:           %s\n' "$LOG_FILE"
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
    section "Preflight — this must be run on a bench unit you can afford to lose"

    local model=""
    [[ -r /proc/device-tree/model ]] && model=$(tr -d '\0' < /proc/device-tree/model)
    if [[ "$model" == *"Raspberry Pi 5"* ]]; then
        ok "Board: $model"
    else
        warn "Board reports '${model:-unknown}', not a Raspberry Pi 5 — the EEPROM signed-boot step targets Pi 5/CM5 firmware."
        confirm "Continue anyway?" N || die "Aborted."
    fi

    [[ -f /etc/systemd/system/adts.service || -f "$REPO/deploy/adts.service" ]] \
        || die "No adts.service found. Run deploy/pi_setup.sh and test the tracker by hand (docs/DEPLOY_PI.md) before hardening it."
    if systemctl is-enabled adts >/dev/null 2>&1 || [[ -f /etc/systemd/system/adts.service ]]; then
        ok "adts.service is installed"
    fi
    confirm "Have you already run pi_setup.sh AND verified the tracker (camera, Hailo, MAVLink) works by hand?" N \
        || die "Do that first (docs/DEPLOY_PI.md sections 4-5) — this script removes your only way to debug it afterwards."

    warn "This is IRREVERSIBLE on real hardware: no SSH, no console, no login shell, no way"
    warn "back in except re-flashing the SD card at a bench. EEPROM OTP writes (step 8) can"
    warn "brick the board if done wrong. Do this on a spare unit first, not your only one."
    # Probe /dev/tty separately from reading it: a real terminal that got EOF (Ctrl-D)
    # must still die() below, not fall through to the --yes bypass meant for "no terminal
    # at all". The 2>/dev/null has to wrap the probe, not follow it, or bash prints its
    # own diagnostic for the failed open before a same-command redirect can suppress it.
    if { : < /dev/tty; } 2>/dev/null; then
        local phrase
        read -r -p "  Type PROCEED to continue: " phrase < /dev/tty || phrase=""
        [[ "$phrase" == "PROCEED" ]] || die "Aborted (typed '$phrase', expected PROCEED)."
    elif (( ASSUME_YES )); then
        warn "--yes given with no terminal: proceeding without the typed confirmation."
    else
        die "No terminal to confirm on, and --yes doesn't cover this gate on purpose. Run interactively."
    fi
    record "Preflight" "OK"
}

# ---------------------------------------------------------------------------
# Step 2: dedicated, non-login service account
# ---------------------------------------------------------------------------
step_service_account() {
    section "Dedicated service account ($SERVICE_USER)"
    if id -u "$SERVICE_USER" >/dev/null 2>&1; then
        info "$SERVICE_USER already exists"
    else
        run useradd --system --no-create-home --shell /usr/sbin/nologin \
            --groups video,render,dialout "$SERVICE_USER"
        ok "Created system user $SERVICE_USER (nologin, in video/render/dialout groups)"
    fi
    record "Service account" "OK"
}

# ---------------------------------------------------------------------------
# Step 3: bytecode-only production tree + its own systemd unit
# ---------------------------------------------------------------------------
step_production_tree() {
    section "Bytecode-only app tree ($PROD_DIR)"
    [[ -x "$REPO/.venv/bin/python" ]] || die "$REPO/.venv is missing — run pi_setup.sh first."
    find "$REPO/models" -maxdepth 1 -name '*.hef' -print -quit 2>/dev/null | grep -q . \
        || warn "No .hef model in $REPO/models — copying anyway, but the service won't start until one exists."

    run mkdir -p "$PROD_DIR" "$DATA_DIR/rec"
    run rsync -a --delete "$REPO/.venv/" "$PROD_DIR/.venv/"
    run rsync -a --delete "$REPO/adts/" "$PROD_DIR/adts/"
    run mkdir -p "$PROD_DIR/models"
    run rsync -a --delete "$REPO/models/" "$PROD_DIR/models/"
    info "Copied .venv, adts/, models/ — not weights/, videos/, tests/, tools/, docs/, .git"

    if (( ! DRY_RUN )); then
        "$PROD_DIR/.venv/bin/python" -m compileall -b -q "$PROD_DIR/adts"
        find "$PROD_DIR/adts" -name '*.py' -delete
        find "$PROD_DIR/adts" -name '__pycache__' -type d -exec rm -rf {} +
    else
        info "[dry-run] would compile adts/*.py to *.pyc (legacy layout) and delete the .py sources"
    fi
    [[ -f "$PROD_DIR/adts/__main__.pyc" || $DRY_RUN -eq 1 ]] \
        || die "compileall didn't produce __main__.pyc — check the log above."
    ok "adts/ is bytecode-only in $PROD_DIR"

    run chown -R "root:$SERVICE_USER" "$PROD_DIR"
    run chmod -R g+rX,o-rwx "$PROD_DIR"
    run chown "$SERVICE_USER:$SERVICE_USER" "$DATA_DIR" "$DATA_DIR/rec"
    run chmod 750 "$DATA_DIR" "$DATA_DIR/rec"

    local rendered; rendered=$(mktemp)
    TMP_FILES+=("$rendered")
    sed -e "s#@REPO@#$PROD_DIR#g" -e "s#@USER@#$SERVICE_USER#g" "$REPO/deploy/adts.service" > "$rendered"
    run cp "$rendered" /etc/systemd/system/adts.service
    run mkdir -p /etc/systemd/system/adts.service.d
    cat > "$rendered" <<-EOF
	# Production-only hardening on top of deploy/adts.service's shared baseline (see the
	# comment block in that file for why some directives are deliberately NOT set there).
	# Safe here specifically because this image has no other account and no other home
	# directory left once step 4 (account lockdown) has run.
	[Service]
	ProtectHome=yes
	# Reset the base unit's ReadWritePaths=$PROD_DIR (the app tree is frozen bytecode in
	# production, nothing should write there) and grant only the data directory instead.
	ReadWritePaths=
	ReadWritePaths=$DATA_DIR
	EOF
    run cp "$rendered" /etc/systemd/system/adts.service.d/10-production.conf

    if [[ -f /etc/default/adts ]]; then
        run sed -i "s#--record [^ \"]*#--record $DATA_DIR/rec#" /etc/default/adts
    else
        run cp "$REPO/deploy/adts.env" /etc/default/adts
        run sed -i "s#--record [^ \"]*#--record $DATA_DIR/rec#" /etc/default/adts
    fi
    ok "Recording path set to $DATA_DIR/rec"
    run systemctl daemon-reload
    record "Production app tree" "OK"
}

# ---------------------------------------------------------------------------
# Step 4: remove/lock every human, interactive account
# ---------------------------------------------------------------------------
step_lock_accounts() {
    section "Account lockdown"
    if id -u pi >/dev/null 2>&1; then
        if confirm "Delete the default 'pi' account entirely (recommended)?" Y; then
            run pkill -u pi 2>/dev/null || true
            run userdel -r pi 2>/dev/null || run userdel pi
            ok "Removed 'pi'"
        else
            run usermod -s /usr/sbin/nologin -L pi
            warn "Kept 'pi' but locked its password and shell"
        fi
    else
        info "No 'pi' account present"
    fi

    local u uid shell locked=()
    while IFS=: read -r u _ uid _ _ _ shell; do
        (( uid >= 1000 && uid < 65534 )) || continue
        [[ "$u" == "$SERVICE_USER" ]] && continue
        case "$shell" in */nologin|*/false) continue ;; esac
        run usermod -s /usr/sbin/nologin -L "$u"
        run gpasswd -d "$u" sudo 2>/dev/null || true
        locked+=("$u")
    done < /etc/passwd
    if (( ${#locked[@]} )); then
        ok "Locked login + sudo for: ${locked[*]}"
    else
        ok "No other interactive accounts found"
    fi

    run passwd -l root
    ok "Locked root's password (no console/SSH exists to use it anyway, after step 5-6)"
    record "Account lockdown" "OK"
}

# ---------------------------------------------------------------------------
# Step 5: no SSH at all
# ---------------------------------------------------------------------------
step_purge_ssh() {
    section "Remove SSH"
    if dpkg -l openssh-server 2>/dev/null | grep -q '^ii'; then
        run systemctl stop ssh ssh.socket 2>/dev/null || true
        run_spin "Purging openssh-server" apt-get purge -y openssh-server openssh-sftp-server
    else
        info "openssh-server not installed"
    fi
    run rm -rf /etc/ssh
    run systemctl mask ssh.service ssh.socket 2>/dev/null || true
    ok "SSH removed and masked"
    record "SSH removal" "OK"
}

# ---------------------------------------------------------------------------
# Step 6: no console, no getty, no SysRq
# ---------------------------------------------------------------------------
cmdline_remove_token() {
    local pattern=$1
    if (( DRY_RUN )); then
        printf '  %s[dry-run] would remove cmdline.txt tokens matching:%s %s\n' "$C_DIM" "$C_RESET" "$pattern"
        return 0
    fi
    local cur; cur=$(cat "$CMDLINE")
    local new; new=$(echo "$cur" | tr ' ' '\n' | grep -vE "$pattern" | tr '\n' ' ' | sed 's/ *$//')
    if [[ "$new" != "$cur" ]]; then
        printf '%s\n' "$new" > "$CMDLINE"
        ok "Removed cmdline.txt tokens matching: $pattern"
    else
        info "No matching cmdline.txt tokens: $pattern"
    fi
}

cmdline_add_token() {
    local token=$1
    if (( DRY_RUN )); then
        printf '  %s[dry-run] would ensure cmdline.txt has token:%s %s\n' "$C_DIM" "$C_RESET" "$token"
        return 0
    fi
    if grep -qw -- "$token" "$CMDLINE" 2>/dev/null; then
        info "already set: $token"
    else
        sed -i "s/\$/ $token/" "$CMDLINE"
        ok "added: $token"
    fi
}

step_disable_consoles() {
    section "Disable console/getty and SysRq"
    local backup; backup="${CMDLINE}.adts-bak-$(date +%Y%m%d%H%M%S)"
    run cp "$CMDLINE" "$backup"
    info "Backed up cmdline.txt to $(basename "$backup")"

    cmdline_remove_token 'console=(serial0|ttyAMA0|tty1)(,[0-9]+)?'
    cmdline_add_token "quiet"
    cmdline_add_token "loglevel=0"
    cmdline_add_token "vt.global_cursor_default=0"
    cmdline_add_token "logo.nologo"

    run systemctl mask serial-getty@.service getty@tty1.service getty@.service console-getty.service 2>/dev/null || true
    ok "Masked getty/serial-getty units"

    run bash -c 'printf "kernel.sysrq = 0\n" > /etc/sysctl.d/99-adts-hardening.conf'
    run sysctl -w kernel.sysrq=0 >/dev/null 2>&1 || true
    ok "Magic SysRq disabled"

    local current; current=$(systemctl get-default 2>/dev/null || echo "unknown")
    if [[ "$current" != "multi-user.target" ]]; then
        warn "Boot target is '$current', not multi-user.target — run pi_setup.sh's boot-target step first."
    fi
    record "Console/getty/SysRq" "OK"
}

# ---------------------------------------------------------------------------
# Step 7: read-only root
# ---------------------------------------------------------------------------
step_readonly_root() {
    section "Read-only root filesystem"
    if ! command -v raspi-config >/dev/null 2>&1; then
        warn "raspi-config not found — enable Overlay FS by hand: Performance Options -> Overlay FS."
        record "Read-only root" "SKIP"
        return
    fi
    if run raspi-config nonint enable_overlayfs; then
        ok "Overlay FS enabled (takes effect next boot)"
        record "Read-only root" "OK"
    else
        warn "raspi-config couldn't enable Overlay FS automatically on this OS version."
        warn "Do it by hand: sudo raspi-config -> Performance Options -> Overlay FS -> Yes to both prompts."
        record "Read-only root" "SKIP"
    fi
}

# ---------------------------------------------------------------------------
# Step 8: EEPROM signed boot — guided, never run blind
# ---------------------------------------------------------------------------
step_signed_boot() {
    section "EEPROM signed boot (guided — see docs/PRODUCTION_HARDENING.md)"
    if ! command -v rpi-eeprom-config >/dev/null 2>&1; then
        warn "rpi-eeprom-update tools not found (apt install rpi-eeprom). Skipping."
        record "Signed boot" "SKIP"
        return
    fi
    echo "  This OTP-burns a public key hash into the board — irreversible, and a wrong key"
    echo "  or a wrong BOOT_ORDER locks you out of the board for good if you get it wrong."
    echo "  Full command sequence: docs/PRODUCTION_HARDENING.md, 'EEPROM signed boot'."
    echo "  Do it now, by hand, in another terminal — this script waits."
    if ! confirm "Have you signed the boot image and set SIGNED_BOOT=1 in the EEPROM config?" N; then
        warn "Skipped — the board will boot any image on this SD card until you do this."
        record "Signed boot" "SKIP"
        return
    fi
    local cfg; cfg=$(rpi-eeprom-config 2>/dev/null || echo "")
    if grep -q '^SIGNED_BOOT=1' <<<"$cfg"; then
        ok "SIGNED_BOOT=1 confirmed"
        if grep -qE '^BOOT_ORDER=0*1$' <<<"$cfg"; then
            ok "BOOT_ORDER restricted to SD only"
        else
            warn "BOOT_ORDER isn't SD-only ($(grep '^BOOT_ORDER=' <<<"$cfg")) — USB/network boot fallback may still work."
        fi
        record "Signed boot" "OK"
    else
        warn "rpi-eeprom-config doesn't show SIGNED_BOOT=1 yet — re-run this step once it's applied and locked."
        record "Signed boot" "WARN"
    fi
}

# ---------------------------------------------------------------------------
# Step 9: full-disk encryption — guided, never run blind
# ---------------------------------------------------------------------------
step_disk_encryption() {
    section "Full-disk encryption (guided — see docs/PRODUCTION_HARDENING.md)"
    echo "  Converting the running root filesystem to LUKS in place isn't something this"
    echo "  script attempts blind: it needs a separate boot medium, a repartition, and an"
    echo "  initramfs rebuild with the key embedded. Full steps, and the known limitation"
    echo "  (this key isn't secret against someone who reads the boot partition directly —"
    echo "  no secure element yet), are in docs/PRODUCTION_HARDENING.md, 'Disk encryption'."
    if ! confirm "Has this unit's root filesystem already been converted to LUKS?" N; then
        warn "Skipped — the SD card is plain, readable ext4 until you do this."
        record "Disk encryption" "SKIP"
        return
    fi
    local src; src=$(findmnt -no SOURCE / 2>/dev/null || echo "")
    if [[ "$src" == /dev/mapper/* ]]; then
        ok "Root is mounted from $src (LUKS mapper device)"
        record "Disk encryption" "OK"
    else
        warn "Root is mounted from '$src', not a /dev/mapper/* device — doesn't look encrypted yet."
        record "Disk encryption" "WARN"
    fi
}

print_summary_table() {
    local i name status
    for i in "${!SUMMARY_NAMES[@]}"; do
        name="${SUMMARY_NAMES[$i]}"; status="${SUMMARY_STATUS[$i]}"
        case "$status" in
            OK)   printf '  %s%s%s %s\n' "$C_GREEN" "$ICON_OK" "$C_RESET" "$name" ;;
            SKIP) printf '  %s--%s %s (skipped)\n' "$C_DIM" "$C_RESET" "$name" ;;
            WARN) printf '  %s%s%s %s (needs attention)\n' "$C_YELLOW" "$ICON_WARN" "$C_RESET" "$name" ;;
            FAIL) printf '  %s%s%s %s (failed)\n' "$C_RED" "$ICON_ERR" "$C_RESET" "$name" ;;
        esac
    done
}

final_summary() {
    printf '\n%s== Summary ==%s\n' "${C_BOLD}${C_BLUE}" "$C_RESET"
    print_summary_table
    echo
    echo "Before cloning this SD card to other units:"
    echo "  1. Reboot, then verify per docs/PRODUCTION_HARDENING.md's checklist (no SSH, no"
    echo "     getty, no console prompt on a USB-UART adapter, service starts clean)."
    echo "  2. Pull the SD card and image it on a bench PC to confirm what a determined"
    echo "     attacker would actually see."
    echo "  3. Any step marked SKIP or (needs attention) above is NOT done — this is not a"
    echo "     production-ready image until every step is OK."
    if (( DRY_RUN )); then
        echo
        warn "This was a dry run — nothing was changed."
    fi
    return 0
}

main() {
    ORIG_ARGS=("$@")
    parse_args "$@"
    setup_ui

    if (( SHOW_HELP )); then
        show_help
        exit 0
    fi

    (( DRY_RUN )) || require_root_or_reexec
    resolve_repo_and_config
    setup_logging
    banner

    step_preflight
    step_service_account
    step_production_tree
    step_lock_accounts
    step_purge_ssh
    step_disable_consoles
    step_readonly_root
    step_signed_boot
    step_disk_encryption
    final_summary
}

main "$@"
