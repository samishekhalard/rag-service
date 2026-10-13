# Vertex-7 — Maintenance

## Scheduled intervals

The controller tracks running hours per pump and raises a maintenance
reminder at 4000 hours. The reminder is informational and never stops the
station.

Mechanical seal inspection is recommended every 8000 hours or annually,
whichever comes first. Bearing lubrication follows the pump manufacturer's
schedule, which the controller does not track.

## Firmware

Firmware is updated over the service port using a USB stick formatted FAT32.
The update file must be placed in the root of the stick and named `vx7.fw`.
The controller verifies a checksum before writing; a failed checksum aborts the
update and leaves the running firmware untouched.

An update takes roughly four minutes. The station must be stopped first — the
controller refuses to start an update while any pump is running.

## Configuration backup

Settings are exported to the same USB stick as a single file, `vx7-config.json`.
A backup taken on one Vertex-7 can be restored to another unit of the same
hardware revision. Restoring across hardware revisions is rejected, because
sensor scaling tables differ between revisions.

## Battery

The real-time clock is backed by a coin cell with a typical life of five years.
When it fails, the controller keeps running and keeps its configuration, but
alarm timestamps restart from the firmware build date until the clock is set.
