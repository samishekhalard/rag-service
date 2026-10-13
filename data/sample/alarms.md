# Vertex-7 — Alarms and Fault Codes

Alarms are latched. A latched alarm stays active after the condition clears and
must be acknowledged from the panel or over the fieldbus before the station
will resume automatic operation.

## Fault codes

**F-101 Dry run.** Suction pressure stayed below the dry-run threshold, default
0.4 bar, for longer than the dry-run delay, default 5 seconds. The controller
stops all pumps immediately. Dry running destroys mechanical seals within
seconds, which is why this fault bypasses the normal ramp-down.

**F-204 Sensor fault.** A 4-20 mA transmitter reported a current outside the
valid band, below 3.6 mA or above 20.4 mA. The usual cause is a broken wire or
a transmitter that has lost power. The station falls back to manual mode at the
last known good speed.

**F-310 Overpressure.** Discharge pressure exceeded the overpressure limit for
two consecutive control cycles. Default limit is 16 bar.

**F-412 Drive communication lost.** The controller received no response from a
variable frequency drive for three consecutive polls. The affected pump is
removed from the staging pool; the station continues on the remaining pumps.

**F-530 Hunting detected.** More than six stage transitions occurred within ten
minutes. This is a diagnostic alarm — it does not stop the station. It usually
means the stage-up and stage-down delays are too close together for the
station's demand profile.

## Acknowledgement

Acknowledging an alarm clears the latch but does not clear the condition. If
the underlying condition is still present, the alarm re-latches at the next
control cycle.
