#!/usr/bin/env python3
"""
Roblox -> CS2 (Hammer Source1 import) converter.

This writes Source1 VMF files intended to be imported by Source 2 Hammer
(File -> Open -> Source 1 Map Files). Source 2 VMAP is binary DMX, so this
script targets the documented VMF import path for CS2 workflows.

Outputs in one run:
- <name>_cs2_textured.vmf
- <name>_cs2_notexture.vmf

No geometry compression/merging is applied.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import roblox_to_vmf as s1

Vector3 = Tuple[float, float, float]


def parse_offset(raw: str) -> Vector3:
    return s1.parse_offset(raw)


def load_parts_and_spawns(input_path: Path, forced_format: str) -> Tuple[List[s1.Part], List[Vector3]]:
    auto_rbxm = forced_format == "auto" and input_path.suffix.lower() == ".rbxm"

    if forced_format == "rbxm" or auto_rbxm:
        return s1.load_map_from_rbxm_binary(input_path)

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    fmt = s1.detect_input_format(raw, forced_format)
    parts = s1.load_parts(raw, fmt)
    if fmt == "snapshot":
        spawns = s1.extract_spawnpoints_from_snapshot(raw)
    elif fmt == "parts":
        spawns = s1.extract_spawnpoints_from_parts_json(raw)
    else:
        spawns = []
    return parts, s1.dedupe_points(spawns)


def make_no_texture_vmf(vmf_text: str, fallback_material: str) -> str:
    out_lines: List[str] = []
    rx = re.compile(r'^"material"\s+"([^"]+)"$')

    for line in vmf_text.splitlines():
        stripped = line.strip()
        m = rx.match(stripped)
        if not m:
            out_lines.append(line)
            continue

        mat = m.group(1)
        if mat.upper() == "TOOLS/TOOLSSKYBOX":
            out_lines.append(line)
            continue

        indent = line[: len(line) - len(line.lstrip(" \t"))]
        out_lines.append(f'{indent}"material" "{fallback_material}"')

    return "\n".join(out_lines) + "\n"


def convert(args: argparse.Namespace) -> Dict[str, str]:
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts, spawn_points = load_parts_and_spawns(input_path, args.input_format)

    textured_stats = s1.ConversionStats(total_input_parts=len(parts), total_input_spawns=len(spawn_points))
    textured_vmf, variants = s1.build_world_vmf(
        parts=parts,
        spawn_points=spawn_points,
        scale=float(args.scale),
        offset=parse_offset(args.offset),
        cylinder_sides=max(6, int(args.cylinder_sides)),
        material_base_prefix=args.material_base_prefix.strip("/"),
        reverse_winding=bool(args.reverse_winding),
        plane_snap=max(0.0, float(args.plane_snap)),
        non_sky_world_as_func_detail=bool(args.non_sky_world_as_func_detail),
        add_skybox=bool(args.add_skybox),
        skybox_margin=max(0.0, float(args.skybox_margin)),
        skybox_thickness=max(1.0, float(args.skybox_thickness)),
        spawn_height_source=float(args.spawn_height_source),
        stats=textured_stats,
    )

    map_name = args.map_name.strip() or input_path.stem

    textured_path = out_dir / f"{map_name}_cs2_textured.vmf"
    textured_path.write_text(textured_vmf, encoding="utf-8")

    no_texture_text = make_no_texture_vmf(textured_vmf, args.no_texture_material)
    no_texture_path = out_dir / f"{map_name}_cs2_notexture.vmf"
    no_texture_path.write_text(no_texture_text, encoding="utf-8")

    vmt_count = 0
    vmt_dir = out_dir / args.output_vmt_subdir
    if not args.skip_vmt:
        vmt_count = s1.write_vmt_files(vmt_dir, variants)

    return {
        "textured_vmf": str(textured_path),
        "notexture_vmf": str(no_texture_path),
        "vmt_dir": str(vmt_dir),
        "input_parts": str(len(parts)),
        "input_spawns": str(len(spawn_points)),
        "vmt_count": str(vmt_count),
        "exported_parts": str(textured_stats.exported_parts),
        "exported_solids": str(textured_stats.exported_solids),
        "skybox_solids": str(textured_stats.skybox_solids_added),
        "func_detail_entities": str(textured_stats.func_detail_entities_created),
        "player_starts": str(textured_stats.player_starts_created),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert Roblox map data to CS2-targeted Source1 VMF files (textured + no-texture)."
    )
    p.add_argument("--input", required=True, help="Input .rbxm or JSON file")
    p.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "parts", "snapshot", "rbxm"],
        help="Input schema. auto detects snapshot/parts and uses RBXM when extension is .rbxm.",
    )
    p.add_argument("--output-dir", default="out", help="Output directory")
    p.add_argument("--map-name", default="", help="Base output map name (default: input stem)")

    p.add_argument("--scale", type=float, default=s1.DEFAULT_SCALE, help="Stud -> Source unit scale")
    p.add_argument("--offset", default="0,0,0", help="World offset in Source units as x,y,z")
    p.add_argument(
        "--cylinder-sides",
        type=int,
        default=s1.DEFAULT_CYLINDER_SIDES,
        help="Cylinder approximation sides (>=6)",
    )
    p.add_argument(
        "--material-base-prefix",
        default="roblox_base",
        help="$basetexture prefix for generated VMTs in textured output",
    )
    p.add_argument(
        "--output-vmt-subdir",
        default="materials/roblox_generated",
        help="Relative subfolder under output-dir for generated VMT files",
    )
    p.add_argument("--skip-vmt", action="store_true", help="Skip writing VMT files")

    p.add_argument(
        "--no-texture-material",
        default="TOOLS/TOOLSNODRAW",
        help="Material used for all non-sky faces in *_cs2_notexture.vmf",
    )

    p.add_argument("--reverse-winding", action="store_true", default=True)
    p.add_argument("--no-reverse-winding", dest="reverse_winding", action="store_false")
    p.add_argument(
        "--plane-snap",
        type=float,
        default=1.0,
        help="Snap plane coords to this grid (Source units)",
    )

    p.add_argument("--add-skybox", action="store_true", default=True)
    p.add_argument("--no-skybox", dest="add_skybox", action="store_false")
    p.add_argument("--skybox-margin", type=float, default=1024.0)
    p.add_argument("--skybox-thickness", type=float, default=128.0)
    p.add_argument("--spawn-height-source", type=float, default=48.0)

    p.add_argument(
        "--non-sky-world-as-func-detail",
        action="store_true",
        default=True,
        help="Emit non-sky brushes as func_detail entities (recommended for imported VMFs)",
    )
    p.add_argument(
        "--no-non-sky-world-as-func-detail",
        dest="non_sky_world_as_func_detail",
        action="store_false",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = convert(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(
        "[ok] cs2_vmf "
        f"textured={result['textured_vmf']} "
        f"notexture={result['notexture_vmf']} "
        f"vmt_dir={result['vmt_dir']} "
        f"input_parts={result['input_parts']} "
        f"input_spawns={result['input_spawns']} "
        f"exported_parts={result['exported_parts']} "
        f"solids={result['exported_solids']} "
        f"skybox_solids={result['skybox_solids']} "
        f"func_detail_entities={result['func_detail_entities']} "
        f"player_starts={result['player_starts']} "
        f"vmt={result['vmt_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
