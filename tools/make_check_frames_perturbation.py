"""Write a ``slac.check_frames`` YAML that turns one frame of a bag's /tf_static by a yaw.

A known-bad candidate for ``calibrex check``: the child frame's rotation is
left-multiplied by a rotation of ``--yaw-deg`` about the *parent* frame's z axis
(``R' = Rz(yaw) R``), the translation is unchanged. Pass the file with
``--tf``; it overrides the bag's /tf_static for that frame::

    python tools/make_check_frames_perturbation.py BAG velo_link 1.0 --output velo_yaw1.yaml
    calibrex check BAG --tf velo_yaw1.yaml ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from scipy.spatial.transform import Rotation

from calibrex.check.tf_sources import load_bag_tf_static


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path)
    parser.add_argument("frame", help="child frame whose rotation is perturbed")
    parser.add_argument("yaw_deg", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = load_bag_tf_static(args.bag)
    if source is None:
        parser.error("the bag has no /tf_static")
    edge = next((item for item in source.edges if item.child == args.frame), None)
    if edge is None:
        parser.error(f"/tf_static has no child frame {args.frame!r}")
    rotation = Rotation.from_euler("z", args.yaw_deg, degrees=True) * Rotation.from_quat(
        edge.transform.rotation_quat_xyzw
    )
    payload = {
        "schema_version": "slac.check_frames/v0.1",
        "frames": [
            {
                "name": edge.child,
                "parent": edge.parent,
                "translation_m": [float(v) for v in edge.transform.translation_m],
                "rotation_quat_xyzw": [float(v) for v in rotation.as_quat()],
            }
        ],
    }
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.output} ({args.frame} yaw {args.yaw_deg:+g} deg about {edge.parent} z)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
