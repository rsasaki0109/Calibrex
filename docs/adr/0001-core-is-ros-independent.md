# ADR 0001: Core Is ROS-Independent

## Status

Accepted.

## Decision

`src/slac/core` must not import ROS message packages or Autoware runtime
packages. ROS1, ROS2, MCAP, and Autoware integrations are adapters.

## Consequence

The internal schema can remain stable across robotics stacks, and autonomous
driving users can integrate through export and adapter layers without forcing
all users into the same runtime.
