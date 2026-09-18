# Sealing a unit for shipment

`deploy/pi_setup.sh` gets one Pi from a clean OS image to a working, tested tracker. This
document is the next step: turning that Pi into a **sealed product** nobody but you should
be able to get a shell on or read the source off — for the case where the customer holds
the physical device, not just the network.

**Threat model:** a determined, technically capable person with the device in hand — they
will open the case, pull the SD card, probe UART/debug pads, and try booting a different
image on the board. Not in scope: chip-off forensics on desoldered flash. No remote/field
access exists either way (SSH is removed entirely below) and no OTA updates — a returned
or misbehaving unit is serviced by re-flashing its SD card at your bench, not by logging in.

Read this whole document before running anything. Several steps are irreversible (EEPROM
OTP writes) or destructive (disk repartitioning) — do the first pass on a spare unit you
can afford to lose, not your only bench Pi.

## Layers, and what each one actually buys you

| # | Layer | Stops | Doesn't stop |
|---|---|---|---|
| 1 | Case/physical | Casual opening; makes tampering visible | A determined person who accepts the visible tamper evidence |
| 2 | Signed boot (EEPROM) | Booting any OS image on this board except the one you signed | Reading the SD card's contents directly (no boot involved) |
| 3 | LUKS (software-only, no secure element) | Casual/curious reads of a pulled SD card | Someone who reads the **boot partition** itself and extracts the embedded key — see below |
| 4 | OS hardening | Getting a shell at all: no SSH, no console, no getty, no login accounts | An exploit against the running service itself (defence in depth, not this document's focus) |
| 5 | Bytecode-only shipment | Casual "just open the .py in an editor" curiosity | Deliberate decompilation — layer 3 is the real backstop, this just raises the floor |

Layer 3's gap is the one to understand before you rely on this: **signed boot guarantees
which code runs; it does not hide what's readable from a card reader with no code
execution at all.** The boot partition has to stay in the clear for the ROM bootloader to
read it, so any key embedded in the initramfs sitting on that partition can be extracted
by unpacking the initramfs image on a completely different computer — no signing, no
running code, no exploit needed, just `binwalk`/`cpio` and reading a file. This setup
still stops the vast majority of people (anyone not specifically going after the key), and
layer 1 (the sealed case, tamper-evident) carries more of the real load here than the
encryption does. **The only fix that closes this gap is a small hardware secure element**
(e.g. an I2C ATECC608B, wired to the Pi's header) holding the LUKS key and releasing it
only to the verified boot chain — deferred for now by choice. If you add one later, only
the "where does the initramfs get the key from" step below changes; everything else in
this document stays as-is.

## 0. Case (do this on every unit, no tooling needed)

- Security screws (torx/pentalobe) on the enclosure.
- A tamper-evident seal or void-sticker across the seam.
- Epoxy/hot-glue over the UART debug pads and any unused GPIO pins inside the case, once
  the MAVLink UART wiring is confirmed working. Leave the SD card reachable only by
  opening the case.

## Production flow: golden master, then per-unit personalisation

The account/SSH/console/bytecode/EEPROM-signing steps below don't contain any per-device
secret, so do them **once** on a golden master and clone its SD card to every unit. The
LUKS key in step 2 **is** a per-device secret — generate it **fresh on every individual
card** after cloning, so one leaked unit's key doesn't expose the rest of the fleet.

```
golden master  →  clone to each card  →  per-card: fresh LUKS key (step 2 below)  →  ship
(steps 1, 3-6)                            (step 2)
```

## 1. Run the automated steps

```bash
sudo deploy/pi_setup.sh                 # if not already done, and verify by hand (DEPLOY_PI.md)
sudo deploy/build_production_image.sh   # accounts, SSH, console, bytecode-only tree, overlay FS
```

`build_production_image.sh` asks you to type `PROCEED` (not just y/n — that gate ignores
`--yes` on purpose) before doing anything, then runs through: a dedicated `adtssvc` system
account with no login shell, a bytecode-only copy of `adts/` at `/opt/adts`, removal of
every other interactive account, a full SSH purge, masking every getty/serial-console
unit, disabling Magic SysRq, quiet kiosk boot flags, and enabling the read-only overlay
filesystem. It stops and asks before the two guided steps below (2 and the EEPROM part of
3) rather than attempting either blind — full disk repartitioning and OTP writes are not
things to get wrong unattended.

## 2. Disk encryption (do this fresh, per unit)

Boot from a **separate** rescue medium (a second SD card or USB stick with Raspberry Pi OS)
with the target card in a USB adapter — never try to encrypt the root filesystem you're
currently running from.

```bash
# On the rescue system, target card as /dev/sdX (double-check with lsblk — wrong device = data loss):
sudo cryptsetup luksFormat /dev/sdX2                     # temporary passphrase, prompted
sudo dd if=/dev/urandom of=adts.key bs=512 count=4        # the real, per-unit key
sudo cryptsetup luksAddKey /dev/sdX2 adts.key
sudo cryptsetup luksRemoveKey /dev/sdX2                   # drop the passphrase slot - keyfile only
sudo cryptsetup open /dev/sdX2 adts_root
sudo mkfs.ext4 /dev/mapper/adts_root
sudo mount /dev/mapper/adts_root /mnt
sudo rsync -aHAX /mnt/old-root/ /mnt/                     # the golden master's existing rootfs
```

Then, with the new root mounted and chrooted (or edited directly by UUID/path from the
rescue system):

```bash
# /etc/crypttab (one line):
adts_root  UUID=<luks-uuid-from: cryptsetup luksUUID /dev/sdX2>  /etc/adts-luks.key  luks

sudo install -m 000 -o root -g root adts.key /mnt/etc/adts-luks.key
# /etc/cryptsetup-initramfs/conf-hook:
KEYFILE_PATTERN="/etc/adts-luks.key"
UMASK=0077

# /boot/firmware/cmdline.txt: change root= to the mapper device
root=/dev/mapper/adts_root

sudo chroot /mnt apt-get install -y cryptsetup-initramfs
sudo chroot /mnt update-initramfs -u -k all
```

Reboot the target unit on its own (remove it from the rescue adapter, boot normally).
Confirm: `findmnt / -o SOURCE` shows `/dev/mapper/adts_root`, and it came up with **no**
passphrase prompt (the embedded keyfile unlocked it automatically).

## 3. EEPROM signed boot (once, on the golden master, before cloning)

This burns a public-key hash into the board's OTP — irreversible, and specific to Pi
5/CM5 firmware with signed-boot support. Cross-check every flag against `rpi-eeprom-config
--help` and the version actually installed (`dpkg -l rpi-eeprom`) before running anything
here for real; firmware tooling flags have changed across releases and this is not a step
to guess on.

```bash
sudo apt-get install -y rpi-eeprom rpi-eeprom-images
openssl genrsa -out adts-boot-private.pem 2048
openssl rsa -in adts-boot-private.pem -pubout -out adts-boot-public.pem

# Build + sign a boot image (bootloader + your config), producing a .sig alongside it:
sudo rpi-eeprom-digest -i /path/to/bootconf.bin -k adts-boot-private.pem -o /path/to/bootconf.sig

# Add SIGNED_BOOT=1 and a locked BOOT_ORDER (SD only, no USB/network fallback) to the
# EEPROM config, then flash + apply it. rpi-eeprom-config's exact invocation depends on
# your installed version - see its --help and Raspberry Pi's own secure-boot documentation
# for the current, correct sequence.
sudo rpi-eeprom-config --edit     # set SIGNED_BOOT=1, BOOT_ORDER=0xf1 (SD only)
sudo rpi-eeprom-config --apply --lock   # burns the OTP key hash and locks the config
```

Verify before shipping: `sudo rpi-eeprom-config` output shows `SIGNED_BOOT=1` and a
locked, SD-only `BOOT_ORDER`. `build_production_image.sh`'s step 8 checks both when you
tell it this is done.

## Verification checklist (every golden master, before cloning)

1. `systemctl status ssh` → not found / masked. No getty on `tty1` or the serial port.
   `cat /proc/cmdline` has no `console=`. `Ctrl+Alt+F2` does nothing. A USB-UART adapter
   on the debug pads shows no login prompt.
2. Pull the SD card, image it (`dd`) on a bench PC **from a spare/test unit, not the
   golden master**: `cryptsetup luksDump` confirms the LUKS header; `strings`/`file` on
   the raw partition shows only ciphertext, no readable Python source.
3. On that same spare unit, try booting a stock, unsigned Raspberry Pi OS image → the
   board must refuse. Try `rpi-eeprom-config` to move `BOOT_ORDER` back to allow USB boot
   → must be rejected (locked).
4. `sudo systemctl status adts` comes up clean as `adtssvc`, not your dev account.
   `--profile` in the logs shows the usual FPS — the sandboxing in `deploy/adts.service`
   costs nothing at runtime, but confirm on real hardware anyway.

## Servicing a returned unit

There is no shell, no SSH, and root's password is locked — by design. The only path back
in is physical: pull the SD card at your bench and re-flash it from a current golden
master (repeating the per-unit LUKS step with a fresh key). Don't keep a "master key" or
backdoor account to make this more convenient — that would undo layers 3 and 4 for every
unit in the field, not just the one you're servicing.
