#!/usr/bin/env python3
"""
Roblox Part JSON -> Source VMF converter.

Supported geometry:
- Block parts
- Cylinder parts (approximated as N-sided prisms)

Input formats:
- parts: {"parts":[...]} or [...]
- snapshot: serialized tree with _class/_properties/_children like server.lua session snapshots
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Vector3 = Tuple[float, float, float]
ColorRGB = Tuple[int, int, int]
Matrix3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


DEFAULT_SCALE = 14.4
DEFAULT_CYLINDER_SIDES = 12
DEFAULT_WORLD_MATERIAL = "tools/toolsskybox"


PART_CLASSES = {"Part"}
SUPPORTED_SHAPES = {"block", "cylinder"}

PART_TYPE_NUM_MAP: Dict[int, str] = {
    0: "ball",
    1: "block",
    2: "cylinder",
}

MATERIAL_ENUM_NUM_MAP: Dict[int, str] = {
    256: "Plastic",
    272: "SmoothPlastic",
    288: "Neon",
    512: "Wood",
    528: "WoodPlanks",
    784: "Marble",
    800: "Slate",
    816: "Concrete",
    832: "Granite",
    848: "Brick",
    864: "Pebble",
    880: "Cobblestone",
    1040: "CorrodedMetal",
    1056: "DiamondPlate",
    1072: "Foil",
    1088: "Metal",
    1280: "Grass",
    1296: "Sand",
    1312: "Fabric",
    1536: "Ice",
    1568: "Glacier",
}


MATERIAL_BASE_MAP: Dict[str, str] = {
    "Plastic": "plastic",
    "SmoothPlastic": "smoothplastic",
    "Neon": "neon",
    "Metal": "metal",
    "DiamondPlate": "diamondplate",
    "CorrodedMetal": "corroded_metal",
    "Foil": "foil",
    "Wood": "wood",
    "WoodPlanks": "wood_planks",
    "Grass": "grass",
    "Slate": "slate",
    "Concrete": "concrete",
    "Brick": "brick",
    "Pebble": "pebble",
    "Cobblestone": "cobblestone",
    "Rock": "rock",
    "Sand": "sand",
    "Fabric": "fabric",
    "Granite": "granite",
    "Marble": "marble",
    "Basalt": "basalt",
    "CrackedLava": "cracked_lava",
    "Asphalt": "asphalt",
    "Mud": "mud",
    "Ground": "ground",
    "Ice": "ice",
    "Snow": "snow",
    "Glacier": "glacier",
    "Glass": "glass",
    "ForceField": "forcefield",
}


MATERIAL_SURFACEPROP_MAP: Dict[str, str] = {
    "Metal": "metal",
    "DiamondPlate": "metal",
    "CorrodedMetal": "metal",
    "Foil": "metal",
    "Wood": "wood",
    "WoodPlanks": "wood",
    "Grass": "grass",
    "Sand": "sand",
    "Glass": "glass",
    "Ice": "ice",
    "Snow": "snow",
    "Fabric": "cloth",
    "Mud": "dirt",
    "Ground": "dirt",
    "Rock": "rock",
    "Cobblestone": "rock",
    "Concrete": "concrete",
    "Brick": "brick",
}


@dataclass(frozen=True)
class CFrame:
    position: Vector3
    rotation: Matrix3


@dataclass(frozen=True)
class Part:
    name: str
    shape: str
    size: Vector3
    cframe: CFrame
    material: str
    color: ColorRGB


@dataclass(frozen=True)
class MaterialVariant:
    vmf_path: str
    output_name: str
    base_texture: str
    color: ColorRGB
    surfaceprop: str


@dataclass
class ConversionStats:
    total_input_parts: int = 0
    total_input_spawns: int = 0
    merged_part_candidates: int = 0
    merged_part_output: int = 0
    merged_part_reduction: int = 0
    merge_stages_run: int = 0
    merge_cycles_run: int = 0
    exported_parts: int = 0
    skipped_parts: int = 0
    dropped_by_limit: int = 0
    skybox_solids_added: int = 0
    player_starts_created: int = 0
    func_detail_entities_created: int = 0
    exported_solids: int = 0
    generated_vmt: int = 0


def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


def sanitize_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip().lower()).strip("_")
    return s or "unnamed"


def to_int_rgb(values: Sequence[float]) -> ColorRGB:
    if len(values) != 3:
        return (255, 255, 255)
    vals = [float(v) for v in values]
    if max(vals) <= 1.0:
        vals = [round(v * 255.0) for v in vals]
    return (
        int(clamp(vals[0], 0, 255)),
        int(clamp(vals[1], 0, 255)),
        int(clamp(vals[2], 0, 255)),
    )


def decode_vector3(value: object) -> Optional[Vector3]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    if isinstance(value, dict):
        t = value.get("_type")
        if t == "Vector3":
            return (float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0)))
        if {"x", "y", "z"}.issubset(value.keys()):
            return (float(value["x"]), float(value["y"]), float(value["z"]))
    return None


def decode_color(value: object) -> ColorRGB:
    if isinstance(value, (list, tuple)):
        try:
            return to_int_rgb(value)  # type: ignore[arg-type]
        except Exception:
            return (255, 255, 255)
    if isinstance(value, dict):
        t = value.get("_type")
        if t == "Color3":
            return to_int_rgb(
                [
                    float(value.get("r", 1.0)),
                    float(value.get("g", 1.0)),
                    float(value.get("b", 1.0)),
                ]
            )
        if {"r", "g", "b"}.issubset(value.keys()):
            return to_int_rgb([float(value["r"]), float(value["g"]), float(value["b"])])
    return (255, 255, 255)


def decode_enum_name(value: object) -> Optional[str]:
    if isinstance(value, str):
        if "." in value:
            return value.rsplit(".", 1)[-1]
        return value
    if isinstance(value, dict):
        if value.get("_type") == "Enum":
            n = value.get("name")
            return str(n) if n is not None else None
        n = value.get("name")
        if isinstance(n, str):
            return n
    return None


def decode_cframe(value: object) -> Optional[CFrame]:
    components: Optional[List[float]] = None
    if isinstance(value, dict) and value.get("_type") == "CFrame":
        raw = value.get("components")
        if isinstance(raw, list) and len(raw) >= 12:
            components = [float(v) for v in raw[:12]]
    elif isinstance(value, list) and len(value) >= 12:
        components = [float(v) for v in value[:12]]

    if not components:
        return None

    px, py, pz = components[0], components[1], components[2]
    rotation: Matrix3 = (
        (components[3], components[4], components[5]),
        (components[6], components[7], components[8]),
        (components[9], components[10], components[11]),
    )
    return CFrame(position=(px, py, pz), rotation=rotation)


def normalize_shape(shape_name: str) -> Optional[str]:
    raw = shape_name.strip().lower()
    if raw in ("block", "parttype.block", "cube"):
        return "block"
    if raw in ("cylinder", "parttype.cylinder"):
        return "cylinder"
    return None


def parse_part_row(row: dict) -> Optional[Part]:
    shape_raw = row.get("shape") or row.get("Shape") or "Block"
    if isinstance(shape_raw, (int, float)):
        shape_raw = PART_TYPE_NUM_MAP.get(int(shape_raw), "Block")
    if not isinstance(shape_raw, str):
        shape_enum = decode_enum_name(shape_raw)
        shape_raw = shape_enum or "Block"
    shape = normalize_shape(shape_raw)
    if shape is None:
        return None

    size = decode_vector3(row.get("size") or row.get("Size"))
    cframe = decode_cframe(row.get("cframe") or row.get("CFrame"))
    if not size or not cframe:
        return None

    material_raw = row.get("material") or row.get("Material") or "Plastic"
    if isinstance(material_raw, (int, float)):
        material = MATERIAL_ENUM_NUM_MAP.get(int(material_raw), f"Material_{int(material_raw)}")
    else:
        material = str(material_raw)
    color = decode_color(row.get("color") or row.get("Color") or [255, 255, 255])
    name = str(row.get("name") or row.get("Name") or "Part")

    return Part(name=name, shape=shape, size=size, cframe=cframe, material=material, color=color)


def extract_parts_from_list_json(raw: object) -> List[Part]:
    rows: List[dict] = []
    if isinstance(raw, list):
        rows = [r for r in raw if isinstance(r, dict)]
    elif isinstance(raw, dict):
        parts = raw.get("parts")
        if isinstance(parts, list):
            rows = [r for r in parts if isinstance(r, dict)]
    out: List[Part] = []
    for row in rows:
        part = parse_part_row(row)
        if part:
            out.append(part)
    return out


def extract_part_from_snapshot_node(node: dict) -> Optional[Part]:
    cls = str(node.get("_class") or "")
    if cls not in PART_CLASSES:
        return None

    props = node.get("_properties")
    if not isinstance(props, dict):
        return None

    shape_name = decode_enum_name(props.get("Shape")) or "Block"
    shape = normalize_shape(shape_name)
    if shape is None:
        return None

    size = decode_vector3(props.get("Size"))
    cframe = decode_cframe(props.get("CFrame"))
    if not size or not cframe:
        return None

    material = decode_enum_name(props.get("Material")) or "Plastic"
    color = decode_color(props.get("Color"))
    name = str(node.get("_name") or "Part")
    return Part(name=name, shape=shape, size=size, cframe=cframe, material=material, color=color)


def walk_snapshot(node: object, out: List[Part]) -> None:
    if not isinstance(node, dict):
        return

    maybe_part = extract_part_from_snapshot_node(node)
    if maybe_part:
        out.append(maybe_part)

    children = node.get("_children")
    if isinstance(children, list):
        for child in children:
            walk_snapshot(child, out)


def extract_parts_from_snapshot(raw: object) -> List[Part]:
    out: List[Part] = []
    walk_snapshot(raw, out)
    return out


def extract_spawnpoints_from_parts_json(raw: object) -> List[Vector3]:
    rows: List[dict] = []
    if isinstance(raw, list):
        rows = [r for r in raw if isinstance(r, dict)]
    elif isinstance(raw, dict):
        parts = raw.get("parts")
        if isinstance(parts, list):
            rows = [r for r in parts if isinstance(r, dict)]

    out: List[Vector3] = []
    for row in rows:
        cls = str(
            row.get("className")
            or row.get("ClassName")
            or row.get("class")
            or row.get("Class")
            or ""
        ).strip().lower()
        if cls != "spawnlocation":
            continue

        cf = decode_cframe(row.get("cframe") or row.get("CFrame"))
        if cf:
            out.append(cf.position)
            continue
        pos = decode_vector3(row.get("position") or row.get("Position"))
        if pos:
            out.append(pos)
    return out


def extract_spawnpoints_from_snapshot(raw: object) -> List[Vector3]:
    out: List[Vector3] = []

    def walk(node: object) -> None:
        if not isinstance(node, dict):
            return
        cls = str(node.get("_class") or "").strip().lower()
        if cls == "spawnlocation":
            props = node.get("_properties")
            if isinstance(props, dict):
                cf = decode_cframe(props.get("CFrame"))
                if cf:
                    out.append(cf.position)
                else:
                    pos = decode_vector3(props.get("Position"))
                    if pos:
                        out.append(pos)
        children = node.get("_children")
        if isinstance(children, list):
            for child in children:
                walk(child)

    walk(raw)
    return out


def dedupe_points(points: Sequence[Vector3], eps: float = 1e-4) -> List[Vector3]:
    out: List[Vector3] = []
    seen = set()
    inv = 1.0 / max(eps, 1e-9)
    for x, y, z in points:
        key = (round(x * inv), round(y * inv), round(z * inv))
        if key in seen:
            continue
        seen.add(key)
        out.append((x, y, z))
    return out


def source_transform(point: Vector3, scale: float, offset: Vector3) -> Vector3:
    rx, ry, rz = point
    ox, oy, oz = offset
    return (
        (rx * scale) + ox,
        ((-rz) * scale) + oy,
        (ry * scale) + oz,
    )


def apply_cframe(cframe: CFrame, local: Vector3) -> Vector3:
    lx, ly, lz = local
    px, py, pz = cframe.position
    r = cframe.rotation
    return (
        px + (r[0][0] * lx) + (r[0][1] * ly) + (r[0][2] * lz),
        py + (r[1][0] * lx) + (r[1][1] * ly) + (r[1][2] * lz),
        pz + (r[2][0] * lx) + (r[2][1] * ly) + (r[2][2] * lz),
    )


def block_local_vertices(size: Vector3) -> List[Vector3]:
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    return [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]


def block_face_indices() -> List[List[int]]:
    return [
        [1, 2, 6, 5],  # +X
        [0, 4, 7, 3],  # -X
        [3, 7, 6, 2],  # +Y
        [0, 1, 5, 4],  # -Y
        [4, 5, 6, 7],  # +Z
        [0, 3, 2, 1],  # -Z
    ]


def cylinder_local_rings(size: Vector3, sides: int) -> Tuple[List[Vector3], List[Vector3]]:
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    top: List[Vector3] = []
    bot: List[Vector3] = []
    for i in range(sides):
        a = (2.0 * math.pi * i) / sides
        x = math.cos(a) * hx
        z = math.sin(a) * hz
        top.append((x, hy, z))
        bot.append((x, -hy, z))
    return top, bot


def snap_value(v: float, snap: float) -> float:
    if snap <= 0:
        return v
    return round(v / snap) * snap


def format_num(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    text = f"{v:.6f}".rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def format_plane(a: Vector3, b: Vector3, c: Vector3, snap: float) -> str:
    a = (snap_value(a[0], snap), snap_value(a[1], snap), snap_value(a[2], snap))
    b = (snap_value(b[0], snap), snap_value(b[1], snap), snap_value(b[2], snap))
    c = (snap_value(c[0], snap), snap_value(c[1], snap), snap_value(c[2], snap))
    return (
        f"({format_num(a[0])} {format_num(a[1])} {format_num(a[2])}) "
        f"({format_num(b[0])} {format_num(b[1])} {format_num(b[2])}) "
        f"({format_num(c[0])} {format_num(c[1])} {format_num(c[2])})"
    )


def vec_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_dot(a: Vector3, b: Vector3) -> float:
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def vec_cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    )


def vec_len(v: Vector3) -> float:
    return math.sqrt(vec_dot(v, v))


def vec_norm(v: Vector3) -> Vector3:
    l = vec_len(v)
    if l <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / l, v[1] / l, v[2] / l)


def compute_face_uv_axes(a: Vector3, b: Vector3, c: Vector3) -> Tuple[str, str]:
    normal = vec_norm(vec_cross(vec_sub(b, a), vec_sub(c, a)))

    # Build tangent basis perpendicular to the plane normal.
    up = (0.0, 0.0, 1.0)
    if abs(vec_dot(normal, up)) > 0.95:
        up = (0.0, 1.0, 0.0)
    u = vec_norm(vec_cross(up, normal))
    v = vec_norm(vec_cross(normal, u))

    uaxis = f'[{format_num(u[0])} {format_num(u[1])} {format_num(u[2])} 0] 0.25'
    vaxis = f'[{format_num(v[0])} {format_num(v[1])} {format_num(v[2])} 0] 0.25'
    return uaxis, vaxis


def map_material_variant(part: Part, base_prefix: str) -> MaterialVariant:
    base_name = MATERIAL_BASE_MAP.get(part.material, sanitize_name(part.material))
    out_name = f"{sanitize_name(base_name)}_{part.color[0]:03d}_{part.color[1]:03d}_{part.color[2]:03d}"
    vmf_path = f"roblox_generated/{out_name}"
    base_texture = f"{base_prefix}/{sanitize_name(base_name)}"
    surfaceprop = MATERIAL_SURFACEPROP_MAP.get(part.material, "concrete")
    return MaterialVariant(
        vmf_path=vmf_path,
        output_name=out_name,
        base_texture=base_texture,
        color=part.color,
        surfaceprop=surfaceprop,
    )


def build_side_block(
    side_id: int,
    a: Vector3,
    b: Vector3,
    c: Vector3,
    material: str,
    plane_snap: float,
) -> str:
    plane = format_plane(a, b, c, plane_snap)
    uaxis, vaxis = compute_face_uv_axes(a, b, c)
    lines = [
        "side",
        "{",
        f'"id" "{side_id}"',
        f'"plane" "{plane}"',
        f'"material" "{material}"',
        f'"uaxis" "{uaxis}"',
        f'"vaxis" "{vaxis}"',
        '"rotation" "0"',
        '"lightmapscale" "16"',
        '"smoothing_groups" "0"',
        "}",
    ]
    return "\n".join(lines)


def build_solid_block(solid_id: int, side_blocks: Iterable[str]) -> str:
    out = ["solid", "{", f'"id" "{solid_id}"']
    out.extend(side_blocks)
    out.extend([
        "editor",
        "{",
        '"color" "0 150 131"',
        '"visgroupshown" "1"',
        '"visgroupautoshown" "1"',
        "}",
    ])
    out.append("}")
    return "\n".join(out)


def part_to_side_planes(
    part: Part,
    scale: float,
    offset: Vector3,
    cylinder_sides: int,
    reverse_winding: bool,
    plane_snap: float,
) -> List[Tuple[Vector3, Vector3, Vector3]]:
    planes: List[Tuple[Vector3, Vector3, Vector3]] = []
    def make_plane(a: Vector3, b: Vector3, c: Vector3, reverse: bool) -> Tuple[Vector3, Vector3, Vector3]:
        if reverse:
            return (a, c, b)
        return (a, b, c)
    if part.shape == "block":
        verts_world = [apply_cframe(part.cframe, v) for v in block_local_vertices(part.size)]
        verts = [source_transform(v, scale, offset) for v in verts_world]
        for idx in block_face_indices():
            a, b, c = verts[idx[0]], verts[idx[1]], verts[idx[2]]
            planes.append(make_plane(a, b, c, reverse_winding))
    elif part.shape == "cylinder":
        top_local, bot_local = cylinder_local_rings(part.size, cylinder_sides)
        top = [source_transform(apply_cframe(part.cframe, v), scale, offset) for v in top_local]
        bot = [source_transform(apply_cframe(part.cframe, v), scale, offset) for v in bot_local]
        for i in range(cylinder_sides):
            j = (i + 1) % cylinder_sides
            planes.append(make_plane(bot[i], bot[j], top[j], reverse_winding))
        planes.append(make_plane(top[0], top[1], top[2], reverse_winding))
        planes.append(make_plane(bot[2], bot[1], bot[0], reverse_winding))
    return planes


def source_box_planes(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    min_z: float,
    max_z: float,
    reverse_winding: bool,
) -> List[Tuple[Vector3, Vector3, Vector3]]:
    verts = [
        (min_x, min_y, min_z),  # 0
        (max_x, min_y, min_z),  # 1
        (max_x, max_y, min_z),  # 2
        (min_x, max_y, min_z),  # 3
        (min_x, min_y, max_z),  # 4
        (max_x, min_y, max_z),  # 5
        (max_x, max_y, max_z),  # 6
        (min_x, max_y, max_z),  # 7
    ]
    out: List[Tuple[Vector3, Vector3, Vector3]] = []
    for idx in block_face_indices():
        a, b, c = verts[idx[0]], verts[idx[1]], verts[idx[2]]
        if reverse_winding:
            out.append((a, c, b))
        else:
            out.append((a, b, c))
    return out


def build_entity_block(
    entity_id: int,
    classname: str,
    kv_pairs: Sequence[Tuple[str, str]],
    editor_color: str,
    logical_pos: str,
) -> str:
    lines = [
        "entity",
        "{",
        f'"id" "{entity_id}"',
        f'"classname" "{classname}"',
    ]
    for k, v in kv_pairs:
        lines.append(f'"{k}" "{v}"')
    lines.extend([
        "editor",
        "{",
        f'"color" "{editor_color}"',
        '"visgroupshown" "1"',
        '"visgroupautoshown" "1"',
        f'"logicalpos" "{logical_pos}"',
        "}",
        "}",
    ])
    return "\n".join(lines)


def build_func_detail_entity_block(entity_id: int, solid_block: str, logical_pos: str) -> str:
    lines = [
        "entity",
        "{",
        f'"id" "{entity_id}"',
        '"classname" "func_detail"',
        solid_block,
        "editor",
        "{",
        '"color" "0 0 255"',
        '"visgroupshown" "1"',
        '"visgroupautoshown" "1"',
        f'"logicalpos" "{logical_pos}"',
        "}",
        "}",
    ]
    return "\n".join(lines)


def build_world_vmf(
    parts: Sequence[Part],
    spawn_points: Sequence[Vector3],
    scale: float,
    offset: Vector3,
    cylinder_sides: int,
    material_base_prefix: str,
    reverse_winding: bool,
    plane_snap: float,
    non_sky_world_as_func_detail: bool,
    add_skybox: bool,
    skybox_margin: float,
    skybox_thickness: float,
    spawn_height_source: float,
    stats: ConversionStats,
) -> Tuple[str, Dict[str, MaterialVariant]]:
    next_id = 2
    world_solids: List[str] = []
    entities: List[str] = []
    variants: Dict[str, MaterialVariant] = {}
    min_x, min_y, min_z = math.inf, math.inf, math.inf
    max_x, max_y, max_z = -math.inf, -math.inf, -math.inf

    def include_point(p: Vector3) -> None:
        nonlocal min_x, min_y, min_z, max_x, max_y, max_z
        min_x = min(min_x, p[0])
        min_y = min(min_y, p[1])
        min_z = min(min_z, p[2])
        max_x = max(max_x, p[0])
        max_y = max(max_y, p[1])
        max_z = max(max_z, p[2])

    for spawn in spawn_points:
        include_point(source_transform(spawn, scale, offset))

    for part in parts:
        planes = part_to_side_planes(part, scale, offset, cylinder_sides, reverse_winding, plane_snap)
        if not planes:
            stats.skipped_parts += 1
            continue
        stats.exported_parts += 1

        mat = map_material_variant(part, material_base_prefix)
        variants[mat.output_name] = mat

        side_blocks: List[str] = []
        part_plane_snap = plane_snap
        min_source_dim = min(abs(part.size[0] * scale), abs(part.size[1] * scale), abs(part.size[2] * scale))
        if part_plane_snap > 0 and not is_axis_aligned_rotation(part.cframe, 1e-3):
            # Avoid distorting rotated brushes.
            part_plane_snap = 0.0
        if part_plane_snap > 0 and min_source_dim < (part_plane_snap * 2.0):
            # Prevent tiny geometry from collapsing to degenerate brushes when snap is coarse.
            part_plane_snap = 0.0
        for a, b, c in planes:
            side_blocks.append(build_side_block(next_id, a, b, c, mat.vmf_path, part_plane_snap))
            include_point(a)
            include_point(b)
            include_point(c)
            next_id += 1
        solid_block = build_solid_block(next_id, side_blocks)
        next_id += 1
        if non_sky_world_as_func_detail:
            logical_y = 2000 + (stats.func_detail_entities_created * 32)
            entities.append(build_func_detail_entity_block(next_id, solid_block, f"[500 {logical_y}]"))
            next_id += 1
            stats.func_detail_entities_created += 1
        else:
            world_solids.append(solid_block)
        stats.exported_solids += 1

    if min_x == math.inf:
        min_x, min_y, min_z = -1024.0, -1024.0, -1024.0
        max_x, max_y, max_z = 1024.0, 1024.0, 1024.0

    if add_skybox:
        margin = max(0.0, float(skybox_margin))
        thickness = max(1.0, float(skybox_thickness))

        inner_min_x, inner_min_y, inner_min_z = min_x - margin, min_y - margin, min_z - margin
        inner_max_x, inner_max_y, inner_max_z = max_x + margin, max_y + margin, max_z + margin

        outer_min_x, outer_min_y, outer_min_z = inner_min_x - thickness, inner_min_y - thickness, inner_min_z - thickness
        outer_max_x, outer_max_y, outer_max_z = inner_max_x + thickness, inner_max_y + thickness, inner_max_z + thickness

        sky_boxes = [
            (outer_min_x, inner_min_x, outer_min_y, outer_max_y, outer_min_z, outer_max_z),  # -X wall
            (inner_max_x, outer_max_x, outer_min_y, outer_max_y, outer_min_z, outer_max_z),  # +X wall
            (inner_min_x, inner_max_x, outer_min_y, inner_min_y, outer_min_z, outer_max_z),  # -Y wall
            (inner_min_x, inner_max_x, inner_max_y, outer_max_y, outer_min_z, outer_max_z),  # +Y wall
            (inner_min_x, inner_max_x, inner_min_y, inner_max_y, outer_min_z, inner_min_z),  # floor
            (inner_min_x, inner_max_x, inner_min_y, inner_max_y, inner_max_z, outer_max_z),  # ceiling
        ]

        for bmin_x, bmax_x, bmin_y, bmax_y, bmin_z, bmax_z in sky_boxes:
            side_blocks = []
            for a, b, c in source_box_planes(
                bmin_x,
                bmax_x,
                bmin_y,
                bmax_y,
                bmin_z,
                bmax_z,
                reverse_winding=reverse_winding,
            ):
                side_blocks.append(build_side_block(next_id, a, b, c, "TOOLS/TOOLSSKYBOX", plane_snap))
                next_id += 1
            world_solids.append(build_solid_block(next_id, side_blocks))
            next_id += 1
            stats.skybox_solids_added += 1
            stats.exported_solids += 1

    for i, spawn in enumerate(dedupe_points(spawn_points)):
        sx, sy, sz = source_transform(spawn, scale, offset)
        sz = sz + float(spawn_height_source)
        origin = f"{format_num(sx)} {format_num(sy)} {format_num(sz)}"
        logical_y = 1000 + (i * 64)
        entities.append(
            build_entity_block(
                entity_id=next_id,
                classname="info_player_start",
                kv_pairs=[("angles", "0 0 0"), ("origin", origin)],
                editor_color="0 255 0",
                logical_pos=f"[0 {logical_y}]",
            )
        )
        next_id += 1
        stats.player_starts_created += 1

    header = [
        "versioninfo",
        "{",
        '"editorversion" "400"',
        '"editorbuild" "0"',
        '"mapversion" "1"',
        '"formatversion" "100"',
        '"prefab" "0"',
        "}",
        "visgroups",
        "{",
        "}",
        "viewsettings",
        "{",
        '"bSnapToGrid" "1"',
        '"bShowGrid" "1"',
        '"nGridSpacing" "64"',
        '"bShow3DGrid" "0"',
        "}",
        "world",
        "{",
        '"id" "1"',
        '"mapversion" "1"',
        '"classname" "worldspawn"',
        '"detailmaterial" "detail/detailsprites"',
        '"detailvbsp" "detail.vbsp"',
        '"maxpropscreenwidth" "-1"',
        '"newunit" "1"',
        f'"material" "{DEFAULT_WORLD_MATERIAL}"',
        '"skyname" "sky_day01_01"',
    ]
    footer = [
        "}",
    ]
    footer.extend(entities)
    footer.extend([
        "cameras",
        "{",
        '"activecamera" "-1"',
        "}",
        "cordon",
        "{",
        '"mins" "(-1024 -1024 -1024)"',
        '"maxs" "(1024 1024 1024)"',
        '"active" "0"',
        "}",
    ])

    vmf = "\n".join(header + world_solids + footer) + "\n"
    return vmf, variants


def write_vmt_files(out_dir: Path, variants: Dict[str, MaterialVariant]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for key in sorted(variants.keys()):
        v = variants[key]
        r, g, b = v.color
        r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
        content = (
            '"LightmappedGeneric"\n'
            "{\n"
            f'\t"$basetexture" "{v.base_texture}"\n'
            f'\t"$color2" "[{r_f:.4f} {g_f:.4f} {b_f:.4f}]"\n'
            f'\t"$surfaceprop" "{v.surfaceprop}"\n'
            "}\n"
        )
        path = out_dir / f"{v.output_name}.vmt"
        path.write_text(content, encoding="utf-8")
        count += 1
    return count


def detect_input_format(raw: object, forced: str) -> str:
    if forced != "auto":
        return forced
    if isinstance(raw, dict) and "_class" in raw:
        return "snapshot"
    return "parts"


def load_parts(raw: object, input_format: str) -> List[Part]:
    if input_format == "parts":
        return extract_parts_from_list_json(raw)
    if input_format == "snapshot":
        return extract_parts_from_snapshot(raw)
    raise ValueError(f"Unsupported input format: {input_format}")


def load_map_from_rbxm_binary(input_path: Path) -> Tuple[List[Part], List[Vector3]]:
    script_dir = Path(__file__).resolve().parent
    local_reader = script_dir / "node_modules" / "rbx-reader"
    reader_target = str(local_reader) if local_reader.exists() else "rbx-reader"

    js = r"""
const fs = require("node:fs");
const reader = require(process.argv[1]);
const inputPath = process.argv[2];
const parsed = reader.parseBuffer(fs.readFileSync(inputPath));
const out = [];
const spawns = [];
for (const inst of (parsed.instances || [])) {
  if (!inst) continue;
  if (inst.ClassName === "Part") {
    out.push({
      name: inst.Name || "Part",
      shape: inst.shape ?? inst.Shape ?? "Block",
      size: inst.size ?? inst.Size ?? null,
      cframe: inst.CFrame ?? inst.cframe ?? null,
      material: inst.Material ?? "Plastic",
      color: inst.Color3uint8 ?? inst.Color ?? [1, 1, 1],
    });
  }
  if (inst.ClassName === "SpawnLocation") {
    const cf = inst.CFrame ?? inst.cframe ?? null;
    if (Array.isArray(cf) && cf.length >= 3) {
      spawns.push([Number(cf[0]) || 0, Number(cf[1]) || 0, Number(cf[2]) || 0]);
    }
  }
}
process.stdout.write(JSON.stringify({ parts: out, spawns }));
"""

    proc = subprocess.run(
        ["node", "-e", js, reader_target, str(input_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"Failed to parse RBXM via node/rbx-reader: {err[:500]}")
    if not proc.stdout.strip():
        raise RuntimeError("RBXM parser returned empty output")

    payload = json.loads(proc.stdout)
    parts = extract_parts_from_list_json(payload)

    spawns: List[Vector3] = []
    raw_spawns = payload.get("spawns") if isinstance(payload, dict) else None
    if isinstance(raw_spawns, list):
        for row in raw_spawns:
            pos = decode_vector3(row)
            if pos:
                spawns.append(pos)
    return parts, dedupe_points(spawns)


def load_parts_from_rbxm_binary(input_path: Path) -> List[Part]:
    parts, _ = load_map_from_rbxm_binary(input_path)
    return parts


def parse_offset(raw: str) -> Vector3:
    pieces = [p.strip() for p in raw.split(",")]
    if len(pieces) != 3:
        raise ValueError("offset must have 3 comma-separated values (x,y,z)")
    return (float(pieces[0]), float(pieces[1]), float(pieces[2]))


def part_volume(part: Part) -> float:
    return float(part.size[0] * part.size[1] * part.size[2])


def quantize_coord(v: float, grid: float, tolerance: float, phase: float = 0.0) -> Optional[int]:
    if grid <= 0:
        return None
    q = (v - phase) / grid
    n = round(q)
    snapped = phase + (n * grid)
    if abs(v - snapped) <= tolerance:
        return int(n)
    return None


def is_axis_aligned_rotation(cframe: CFrame, tolerance: float) -> bool:
    r = cframe.rotation
    # Each row and column must contain exactly one +/-1 and the rest near 0.
    for i in range(3):
        near_unit = 0
        for j in range(3):
            a = abs(r[i][j])
            if abs(a - 1.0) <= tolerance:
                near_unit += 1
            elif a > tolerance:
                return False
        if near_unit != 1:
            return False

    for j in range(3):
        near_unit = 0
        for i in range(3):
            a = abs(r[i][j])
            if abs(a - 1.0) <= tolerance:
                near_unit += 1
        if near_unit != 1:
            return False
    return True


def part_to_axis_aligned_bounds(
    part: Part,
    axis_tolerance: float,
    rotation_tolerance: float,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    if part.shape != "block":
        return None
    if not is_axis_aligned_rotation(part.cframe, rotation_tolerance):
        return None

    verts = [apply_cframe(part.cframe, v) for v in block_local_vertices(part.size)]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)

    # Only merge cuboids that are axis-aligned in Roblox world space.
    for vx, vy, vz in verts:
        if abs(vx - min_x) > axis_tolerance and abs(vx - max_x) > axis_tolerance:
            return None
        if abs(vy - min_y) > axis_tolerance and abs(vy - max_y) > axis_tolerance:
            return None
        if abs(vz - min_z) > axis_tolerance and abs(vz - max_z) > axis_tolerance:
            return None

    return (min_x, max_x, min_y, max_y, min_z, max_z)


def normalize_mod(value: float, grid: float) -> float:
    if grid <= 0:
        return 0.0
    rem = math.fmod(value, grid)
    if rem < 0:
        rem += grid
    if abs(rem - grid) <= 1e-12:
        rem = 0.0
    return rem


def estimate_axis_phase(values: Sequence[float], grid: float, tolerance: float) -> float:
    if grid <= 0 or not values:
        return 0.0

    rems = [normalize_mod(v, grid) for v in values]
    if not rems:
        return 0.0

    bucket = max(1e-6, tolerance, grid / 4096.0)
    counts: Dict[int, int] = {}
    sums: Dict[int, float] = {}
    for rem in rems:
        if rem >= grid - (bucket * 0.5):
            rem = 0.0
        key = int(round(rem / bucket))
        counts[key] = counts.get(key, 0) + 1
        sums[key] = sums.get(key, 0.0) + rem

    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    phase = sums[best_key] / max(1, counts[best_key])
    return normalize_mod(phase, grid)


def cluster_phase_candidates(
    values: Sequence[float],
    grid: float,
    tolerance: float,
    max_candidates: int = 4,
) -> List[float]:
    if grid <= 0 or not values:
        return [0.0]

    rems = [normalize_mod(v, grid) for v in values]
    bucket = max(1e-6, tolerance, grid / 8192.0)
    counts: Dict[int, int] = {}
    sums: Dict[int, float] = {}
    for rem in rems:
        if rem >= grid - (bucket * 0.5):
            rem = 0.0
        key = int(round(rem / bucket))
        counts[key] = counts.get(key, 0) + 1
        sums[key] = sums.get(key, 0.0) + rem

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    phases: List[float] = []
    seen = set()

    dominant = estimate_axis_phase(values, grid, tolerance)
    phases.append(dominant)
    seen.add(round(dominant, 9))

    for key, _count in ranked:
        if len(phases) >= max(1, int(max_candidates)):
            break
        phase = normalize_mod(sums[key] / max(1, counts[key]), grid)
        k = round(phase, 9)
        if k in seen:
            continue
        seen.add(k)
        phases.append(phase)

    # Keep 0-phase as a fallback candidate when not already included.
    if len(phases) < max(1, int(max_candidates)):
        k0 = round(0.0, 9)
        if k0 not in seen:
            phases.append(0.0)

    return phases


def quantize_bounds_to_cuboid(
    bounds: Tuple[float, float, float, float, float, float],
    grid: float,
    snap_tolerance: float,
    phases: Tuple[float, float, float],
) -> Optional[Tuple[int, int, int, int, int, int]]:
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    phase_x, phase_y, phase_z = phases

    q_min_x = quantize_coord(min_x, grid, snap_tolerance, phase_x)
    q_max_x = quantize_coord(max_x, grid, snap_tolerance, phase_x)
    q_min_y = quantize_coord(min_y, grid, snap_tolerance, phase_y)
    q_max_y = quantize_coord(max_y, grid, snap_tolerance, phase_y)
    q_min_z = quantize_coord(min_z, grid, snap_tolerance, phase_z)
    q_max_z = quantize_coord(max_z, grid, snap_tolerance, phase_z)
    if None in (q_min_x, q_max_x, q_min_y, q_max_y, q_min_z, q_max_z):
        return None

    x0, x1 = int(q_min_x), int(q_max_x)
    y0, y1 = int(q_min_y), int(q_max_y)
    z0, z1 = int(q_min_z), int(q_max_z)
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return None
    return (x0, x1, y0, y1, z0, z1)


def cuboid_quantization_error(
    bounds: Tuple[float, float, float, float, float, float],
    cuboid: Tuple[int, int, int, int, int, int],
    grid: float,
    phases: Tuple[float, float, float],
) -> float:
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    x0, x1, y0, y1, z0, z1 = cuboid
    phase_x, phase_y, phase_z = phases
    snapped = (
        phase_x + (x0 * grid),
        phase_x + (x1 * grid),
        phase_y + (y0 * grid),
        phase_y + (y1 * grid),
        phase_z + (z0 * grid),
        phase_z + (z1 * grid),
    )
    return (
        abs(min_x - snapped[0])
        + abs(max_x - snapped[1])
        + abs(min_y - snapped[2])
        + abs(max_y - snapped[3])
        + abs(min_z - snapped[4])
        + abs(max_z - snapped[5])
    )


def merge_cuboids_along_axis(
    cuboids: Sequence[Tuple[int, int, int, int, int, int]],
    axis: str,
) -> List[Tuple[int, int, int, int, int, int]]:
    buckets: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = {}

    if axis == "x":
        for x0, x1, y0, y1, z0, z1 in cuboids:
            buckets.setdefault((y0, y1, z0, z1), []).append((x0, x1))
        merged: List[Tuple[int, int, int, int, int, int]] = []
        for (y0, y1, z0, z1), spans in buckets.items():
            spans.sort(key=lambda it: (it[0], it[1]))
            cur0, cur1 = spans[0]
            for s0, s1 in spans[1:]:
                if s0 <= cur1:
                    cur1 = max(cur1, s1)
                else:
                    merged.append((cur0, cur1, y0, y1, z0, z1))
                    cur0, cur1 = s0, s1
            merged.append((cur0, cur1, y0, y1, z0, z1))
        return merged

    if axis == "y":
        for x0, x1, y0, y1, z0, z1 in cuboids:
            buckets.setdefault((x0, x1, z0, z1), []).append((y0, y1))
        merged = []
        for (x0, x1, z0, z1), spans in buckets.items():
            spans.sort(key=lambda it: (it[0], it[1]))
            cur0, cur1 = spans[0]
            for s0, s1 in spans[1:]:
                if s0 <= cur1:
                    cur1 = max(cur1, s1)
                else:
                    merged.append((x0, x1, cur0, cur1, z0, z1))
                    cur0, cur1 = s0, s1
            merged.append((x0, x1, cur0, cur1, z0, z1))
        return merged

    # axis == "z"
    for x0, x1, y0, y1, z0, z1 in cuboids:
        buckets.setdefault((x0, x1, y0, y1), []).append((z0, z1))
    merged = []
    for (x0, x1, y0, y1), spans in buckets.items():
        spans.sort(key=lambda it: (it[0], it[1]))
        cur0, cur1 = spans[0]
        for s0, s1 in spans[1:]:
            if s0 <= cur1:
                cur1 = max(cur1, s1)
            else:
                merged.append((x0, x1, y0, y1, cur0, cur1))
                cur0, cur1 = s0, s1
        merged.append((x0, x1, y0, y1, cur0, cur1))
    return merged


def merge_cuboids(
    cuboids: Sequence[Tuple[int, int, int, int, int, int]],
    max_passes: int,
) -> List[Tuple[int, int, int, int, int, int]]:
    if not cuboids:
        return []
    current = list(set(cuboids))
    passes = max(1, int(max_passes))
    for _ in range(passes):
        before = len(current)
        current = merge_cuboids_along_axis(current, "x")
        current = merge_cuboids_along_axis(current, "y")
        current = merge_cuboids_along_axis(current, "z")
        current = list(set(current))
        if len(current) == before:
            break
    return current


def cuboid_to_part(
    cuboid: Tuple[int, int, int, int, int, int],
    grid: float,
    phases: Tuple[float, float, float],
    material: str,
    color: ColorRGB,
) -> Part:
    x0, x1, y0, y1, z0, z1 = cuboid
    phase_x, phase_y, phase_z = phases
    min_x, max_x = phase_x + (x0 * grid), phase_x + (x1 * grid)
    min_y, max_y = phase_y + (y0 * grid), phase_y + (y1 * grid)
    min_z, max_z = phase_z + (z0 * grid), phase_z + (z1 * grid)
    size = (max_x - min_x, max_y - min_y, max_z - min_z)
    center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5)
    return Part(
        name="MergedPart",
        shape="block",
        size=size,
        cframe=CFrame(
            position=center,
            rotation=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        ),
        material=material,
        color=color,
    )


def merge_nearby_block_parts(
    parts: Sequence[Part],
    grid: float,
    snap_tolerance: float,
    axis_tolerance: float,
    rotation_tolerance: float,
    max_passes: int,
) -> Tuple[List[Part], Dict[str, int]]:
    mergeable_groups: Dict[Tuple[str, ColorRGB], List[Tuple[Part, Tuple[float, float, float, float, float, float]]]] = {}
    passthrough: List[Part] = []
    mergeable_count = 0

    for part in parts:
        bounds = part_to_axis_aligned_bounds(part, axis_tolerance, rotation_tolerance)
        if bounds is None:
            passthrough.append(part)
            continue
        mergeable_groups.setdefault((part.material, part.color), []).append((part, bounds))

    merged_parts: List[Part] = []
    merged_output_count = 0
    for (material, color), group_rows in mergeable_groups.items():
        x_values: List[float] = []
        y_values: List[float] = []
        z_values: List[float] = []
        for _, bounds in group_rows:
            x_values.extend((bounds[0], bounds[1]))
            y_values.extend((bounds[2], bounds[3]))
            z_values.extend((bounds[4], bounds[5]))

        phases_x = cluster_phase_candidates(x_values, grid, snap_tolerance, max_candidates=4)
        phases_y = cluster_phase_candidates(y_values, grid, snap_tolerance, max_candidates=4)
        phases_z = cluster_phase_candidates(z_values, grid, snap_tolerance, max_candidates=4)

        phase_groups: Dict[Tuple[float, float, float], List[Tuple[int, int, int, int, int, int]]] = {}
        for original_part, bounds in group_rows:
            best_cuboid = None
            best_phase_key = None
            best_error = math.inf

            min_x, max_x, min_y, max_y, min_z, max_z = bounds
            valid_x: List[Tuple[float, int, int]] = []
            valid_y: List[Tuple[float, int, int]] = []
            valid_z: List[Tuple[float, int, int]] = []

            for phase_x in phases_x:
                qx0 = quantize_coord(min_x, grid, snap_tolerance, phase_x)
                qx1 = quantize_coord(max_x, grid, snap_tolerance, phase_x)
                if qx0 is not None and qx1 is not None and qx1 > qx0:
                    valid_x.append((round(phase_x, 9), int(qx0), int(qx1)))
            for phase_y in phases_y:
                qy0 = quantize_coord(min_y, grid, snap_tolerance, phase_y)
                qy1 = quantize_coord(max_y, grid, snap_tolerance, phase_y)
                if qy0 is not None and qy1 is not None and qy1 > qy0:
                    valid_y.append((round(phase_y, 9), int(qy0), int(qy1)))
            for phase_z in phases_z:
                qz0 = quantize_coord(min_z, grid, snap_tolerance, phase_z)
                qz1 = quantize_coord(max_z, grid, snap_tolerance, phase_z)
                if qz0 is not None and qz1 is not None and qz1 > qz0:
                    valid_z.append((round(phase_z, 9), int(qz0), int(qz1)))

            for phase_x, qx0, qx1 in valid_x:
                for phase_y, qy0, qy1 in valid_y:
                    for phase_z, qz0, qz1 in valid_z:
                        phase_key = (
                            phase_x,
                            phase_y,
                            phase_z,
                        )
                        cuboid = (qx0, qx1, qy0, qy1, qz0, qz1)
                        err = cuboid_quantization_error(
                            bounds,
                            cuboid,
                            grid,
                            (phase_key[0], phase_key[1], phase_key[2]),
                        )
                        if err < best_error:
                            best_error = err
                            best_cuboid = cuboid
                            best_phase_key = phase_key

            if best_cuboid is None or best_phase_key is None:
                passthrough.append(original_part)
                continue
            mergeable_count += 1
            phase_groups.setdefault(best_phase_key, []).append(best_cuboid)

        for phase_key, cuboids in phase_groups.items():
            if not cuboids:
                continue
            merged_cuboids = merge_cuboids(cuboids, max_passes=max_passes)
            merged_output_count += len(merged_cuboids)
            for cuboid in merged_cuboids:
                merged_parts.append(
                    cuboid_to_part(
                        cuboid,
                        grid=grid,
                        phases=(phase_key[0], phase_key[1], phase_key[2]),
                        material=material,
                        color=color,
                    )
                )

    out = passthrough + merged_parts
    return out, {
        "mergeable_count": mergeable_count,
        "merged_output_count": merged_output_count,
        "reduction": max(0, mergeable_count - merged_output_count),
        "total_after_merge": len(out),
    }


def build_merge_grid_schedule(grid_min: float, grid_max: float) -> List[float]:
    gmin = max(1e-6, float(grid_min))
    gmax = max(gmin, float(grid_max))
    grids: List[float] = [gmin, gmax]

    # Decimal-friendly progression catches common Roblox build increments.
    exp_min = int(math.floor(math.log10(gmin))) - 1
    exp_max = int(math.ceil(math.log10(gmax))) + 1
    for exp in range(exp_min, exp_max + 1):
        scale = 10.0 ** exp
        for m in (1.0, 2.0, 2.5, 5.0):
            g = m * scale
            if gmin <= g <= gmax:
                grids.append(g)

    # Keep a binary progression too for powers-of-two aligned maps.
    g = gmin
    while g < gmax:
        g *= 2.0
        if g <= gmax:
            grids.append(g)
    # Deduplicate float noise
    out: List[float] = []
    seen = set()
    for g in grids:
        key = round(g, 9)
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    return out


def merge_nearby_block_parts_multistage(
    parts: Sequence[Part],
    grid_min: float,
    grid_max: float,
    snap_tolerance: float,
    axis_tolerance: float,
    rotation_tolerance: float,
    max_passes: int,
    force_snap_each_stage: bool,
    max_cycles: int,
) -> Tuple[List[Part], Dict[str, int]]:
    current = list(parts)
    total_reduction = 0
    stages_run = 0
    cycles_run = 0
    total_mergeable_peak = 0
    total_merged_peak = 0

    schedule = build_merge_grid_schedule(grid_min, grid_max)
    for _ in range(max(1, int(max_cycles))):
        cycle_changed = False
        cycles_run += 1
        for grid in schedule:
            stage_tol = snap_tolerance
            if force_snap_each_stage:
                # Keep this conservative; aggressive snapping causes geometry drift/holes.
                stage_tol = max(stage_tol, (grid * 0.05))

            merged, meta = merge_nearby_block_parts(
                current,
                grid=grid,
                snap_tolerance=stage_tol,
                axis_tolerance=axis_tolerance,
                rotation_tolerance=rotation_tolerance,
                max_passes=max_passes,
            )
            stages_run += 1
            stage_before = len(current)
            stage_after = len(merged)
            if stage_after < stage_before:
                cycle_changed = True
                total_reduction += (stage_before - stage_after)
            current = merged
            total_mergeable_peak = max(total_mergeable_peak, int(meta.get("mergeable_count", 0)))
            total_merged_peak = max(total_merged_peak, int(meta.get("merged_output_count", 0)))
        if not cycle_changed:
            break

    return current, {
        "mergeable_count": total_mergeable_peak,
        "merged_output_count": total_merged_peak,
        "reduction": total_reduction,
        "total_after_merge": len(current),
        "stages_run": stages_run,
        "cycles_run": cycles_run,
    }


def convert(args: argparse.Namespace) -> ConversionStats:
    stats = ConversionStats()
    input_path = Path(args.input)
    output_vmf = Path(args.output_vmf)
    spawn_points: List[Vector3] = []

    forced_format = args.input_format
    auto_rbxm = forced_format == "auto" and input_path.suffix.lower() == ".rbxm"

    if forced_format == "rbxm" or auto_rbxm:
        parts, spawn_points = load_map_from_rbxm_binary(input_path)
    else:
        raw = json.loads(input_path.read_text(encoding="utf-8"))
        format_name = detect_input_format(raw, forced_format)
        parts = load_parts(raw, format_name)
        if format_name == "snapshot":
            spawn_points = extract_spawnpoints_from_snapshot(raw)
        elif format_name == "parts":
            spawn_points = extract_spawnpoints_from_parts_json(raw)
    stats.total_input_parts = len(parts)
    spawn_points = dedupe_points(spawn_points)
    stats.total_input_spawns = len(spawn_points)

    if bool(args.merge_bricks):
        merged_parts, merge_meta = merge_nearby_block_parts_multistage(
            parts,
            grid_min=max(1e-6, float(args.merge_grid_studs)),
            grid_max=max(1e-6, float(args.merge_grid_max_studs)),
            snap_tolerance=max(0.0, float(args.merge_snap_tolerance_studs)),
            axis_tolerance=max(0.0, float(args.merge_axis_tolerance_studs)),
            rotation_tolerance=max(0.0, float(args.merge_rotation_tolerance)),
            max_passes=max(1, int(args.merge_max_passes)),
            force_snap_each_stage=bool(args.merge_force_snap_each_stage),
            max_cycles=max(1, int(args.merge_max_cycles)),
        )
        parts = merged_parts
        stats.merged_part_candidates = int(merge_meta.get("mergeable_count", 0))
        stats.merged_part_output = int(merge_meta.get("merged_output_count", 0))
        stats.merged_part_reduction = int(merge_meta.get("reduction", 0))
        stats.merge_stages_run = int(merge_meta.get("stages_run", 0))
        stats.merge_cycles_run = int(merge_meta.get("cycles_run", 0))

    max_solids = int(args.max_solids) if args.max_solids is not None else 0
    skybox_reserve = 6 if bool(args.add_skybox) else 0
    if max_solids > 0:
        part_budget = max(0, max_solids - skybox_reserve)
    else:
        part_budget = 0
    if max_solids > 0 and len(parts) > part_budget:
        # Hammer builds can choke on very high brush counts; keep the most significant brushes first.
        dropped = max(0, len(parts) - part_budget)
        parts = sorted(parts, key=part_volume, reverse=True)[:part_budget]
        stats.dropped_by_limit = max(0, dropped)

    vmf, variants = build_world_vmf(
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
        stats=stats,
    )

    output_vmf.parent.mkdir(parents=True, exist_ok=True)
    output_vmf.write_text(vmf, encoding="utf-8")

    if not args.skip_vmt:
        out_dir = Path(args.output_vmt_dir)
        stats.generated_vmt = write_vmt_files(out_dir, variants)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert Roblox part JSON to Source VMF.")
    p.add_argument("--input", required=True, help="Input JSON file")
    p.add_argument("--output-vmf", required=True, help="Output VMF path")
    p.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "parts", "snapshot", "rbxm"],
        help="Input schema. auto detects snapshot vs parts.",
    )
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Stud -> Hammer unit scale")
    p.add_argument(
        "--offset",
        default="0,0,0",
        help="World offset in Source units after conversion, as x,y,z",
    )
    p.add_argument(
        "--cylinder-sides",
        type=int,
        default=DEFAULT_CYLINDER_SIDES,
        help="Cylinder approximation sides (>=6)",
    )
    p.add_argument(
        "--material-base-prefix",
        default="roblox_base",
        help="Prefix used for $basetexture in generated VMT files",
    )
    p.add_argument(
        "--output-vmt-dir",
        default="materials/roblox_generated",
        help="Output folder for generated VMT files",
    )
    p.add_argument(
        "--skip-vmt",
        action="store_true",
        help="Skip VMT generation",
    )
    p.add_argument(
        "--max-solids",
        type=int,
        default=0,
        help="If >0, keep only this many largest parts before VMF export (useful for Hammer brush limits).",
    )
    p.add_argument(
        "--add-skybox",
        action="store_true",
        default=True,
        help="Add sealing skybox brushes around exported geometry (default on).",
    )
    p.add_argument(
        "--no-skybox",
        dest="add_skybox",
        action="store_false",
        help="Disable automatic sealing skybox generation.",
    )
    p.add_argument(
        "--skybox-margin",
        type=float,
        default=1024.0,
        help="Padding around map bounds before placing skybox walls (Source units).",
    )
    p.add_argument(
        "--skybox-thickness",
        type=float,
        default=128.0,
        help="Skybox wall thickness (Source units).",
    )
    p.add_argument(
        "--spawn-height-source",
        type=float,
        default=48.0,
        help="Vertical offset added to each info_player_start above SpawnLocation (Source units).",
    )
    p.add_argument(
        "--merge-bricks",
        action="store_true",
        default=True,
        help="Merge nearby axis-aligned block parts (same material/color) into larger cuboids before export.",
    )
    p.add_argument(
        "--no-merge-bricks",
        dest="merge_bricks",
        action="store_false",
        help="Disable brick merging before export.",
    )
    p.add_argument(
        "--merge-grid-studs",
        type=float,
        default=0.01,
        help="Grid size in studs used for merge quantization (smaller catches non-stud-aligned maps; default 0.01).",
    )
    p.add_argument(
        "--merge-grid-max-studs",
        type=float,
        default=1.0,
        help="Maximum grid size used in multistage merging (higher values allow larger merged rectangles).",
    )
    p.add_argument(
        "--merge-snap-tolerance-studs",
        type=float,
        default=0.001,
        help="Allowed snap tolerance in studs when quantizing bounds for merging.",
    )
    p.add_argument(
        "--merge-axis-tolerance-studs",
        type=float,
        default=0.001,
        help="Allowed axis-alignment tolerance in studs for deciding whether a part is mergeable.",
    )
    p.add_argument(
        "--merge-rotation-tolerance",
        type=float,
        default=0.001,
        help="Rotation matrix tolerance when deciding if a part is axis-aligned merge-safe.",
    )
    p.add_argument(
        "--merge-max-passes",
        type=int,
        default=12,
        help="Maximum x/y/z merge passes per stage per material-color group.",
    )
    p.add_argument(
        "--merge-max-cycles",
        type=int,
        default=4,
        help="How many times to rerun the full grid schedule until no further merges.",
    )
    p.add_argument(
        "--merge-force-snap-each-stage",
        action="store_true",
        default=False,
        help="At each stage, snap more aggressively to that stage grid (may distort geometry).",
    )
    p.add_argument(
        "--no-merge-force-snap-each-stage",
        dest="merge_force_snap_each_stage",
        action="store_false",
        help="Disable aggressive per-stage snapping.",
    )
    p.add_argument(
        "--reverse-winding",
        action="store_true",
        default=True,
        help="Reverse plane point winding for all faces (default on for Hammer compatibility).",
    )
    p.add_argument(
        "--no-reverse-winding",
        dest="reverse_winding",
        action="store_false",
        help="Disable reverse winding.",
    )
    p.add_argument(
        "--plane-snap",
        type=float,
        default=1.0,
        help="Snap plane coordinates to this Source-unit grid (default 1.0 to reduce MAX_MAP_PLANES errors).",
    )
    p.add_argument(
        "--non-sky-world-as-func-detail",
        action="store_true",
        default=True,
        help="Emit all non-skybox brushes as func_detail entities (default on).",
    )
    p.add_argument(
        "--no-non-sky-world-as-func-detail",
        dest="non_sky_world_as_func_detail",
        action="store_false",
        help="Keep non-sky brushes directly in worldspawn.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        stats = convert(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(
        "[ok] converted "
        f"input_parts={stats.total_input_parts} "
        f"input_spawns={stats.total_input_spawns} "
        f"mergeable_parts={stats.merged_part_candidates} "
        f"merged_parts={stats.merged_part_output} "
        f"merge_reduction={stats.merged_part_reduction} "
        f"merge_stages={stats.merge_stages_run} "
        f"merge_cycles={stats.merge_cycles_run} "
        f"exported_parts={stats.exported_parts} "
        f"skipped_parts={stats.skipped_parts} "
        f"dropped_by_limit={stats.dropped_by_limit} "
        f"skybox_solids={stats.skybox_solids_added} "
        f"func_detail_entities={stats.func_detail_entities_created} "
        f"player_starts={stats.player_starts_created} "
        f"solids={stats.exported_solids} "
        f"vmt={stats.generated_vmt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
