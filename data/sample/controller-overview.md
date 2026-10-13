# Vertex-7 Pump Controller — Overview

The Vertex-7 is a programmable controller for centrifugal pump stations with
two to eight pumps. It regulates discharge pressure by varying drive speed and
by staging pumps in and out as demand changes.

## Operating modes

The controller runs in one of three modes. In **pressure mode** the unit holds
a discharge setpoint, measured at the station header, by adjusting the lead
pump's drive speed. In **flow mode** it holds a volumetric setpoint instead,
using the magnetic flow meter on the common discharge line. In **manual mode**
the operator fixes drive speed directly and all staging logic is suspended.

Mode is selected from the front panel or over the fieldbus. A mode change takes
effect at the next control cycle, which is 100 ms by default and configurable
down to 20 ms.

## Staging

Pumps are staged in when the lead pump has held above 92 percent of rated speed
for longer than the stage-up delay, which defaults to 30 seconds. They are
staged out when combined speed falls below 60 percent for the stage-down delay,
which defaults to 90 seconds. The asymmetry is deliberate: staging out too
eagerly causes hunting, where the station cycles a pump on and off repeatedly.

Duty rotation distributes running hours across the available pumps. The
controller selects the pump with the fewest accumulated hours as the next lead
whenever the station restarts from idle.

## Sensors

Each Vertex-7 accepts two pressure transmitters, one on suction and one on
discharge, plus an optional flow meter. Transmitters are 4-20 mA. A reading
below 3.6 mA or above 20.4 mA is treated as a sensor fault rather than a
process value.
