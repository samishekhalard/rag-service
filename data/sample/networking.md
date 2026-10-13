# Vertex-7 — Networking and Integration

## Fieldbus

The controller speaks Modbus TCP on port 502 and, with the expansion module,
Modbus RTU over RS-485. Holding registers expose setpoints and mode; input
registers expose process values and alarm state. Register addresses are fixed
and do not change between firmware versions within the same major release.

Writing to the mode register has the same effect as changing mode on the panel,
including the one-cycle delay.

## OPC UA

An OPC UA server is available on port 4840 when the integration licence is
installed. It exposes the same data as the Modbus map, organised as an object
per pump, plus historical access to the last 30 days of process values sampled
at one minute.

Anonymous access is disabled by default. The server supports username and
password, and certificate-based authentication.

## MQTT

The controller can publish telemetry to an MQTT broker. Topics are built from a
configurable station prefix, for example `plant/station-3/pump/2/speed`.
Publication interval defaults to 10 seconds and the quality of service level is
configurable per topic. MQTT is publish-only: commands cannot be sent to the
controller over MQTT, which is intentional, because the transport gives no
delivery guarantee suitable for control.

## Time synchronisation

The controller synchronises its clock over NTP when a server is configured.
Without NTP, clock drift is typically under two minutes per month.
