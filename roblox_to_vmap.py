#!/usr/bin/env python3
"""Roblox (.rbxm/.json) -> Source 2 .vmap (DMX keyvalues2) converter.

This emits native Source 2 VMAP text (keyvalues2 encoding), not Source1 VMF.
It writes two outputs in one run:
- *_cs2_textured.vmap   (CS2 stock material approximation + tint)
- *_cs2_notexture.vmap  (all non-sky geometry set to toolsnodraw)

No geometry compression/merging is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from srctools.dmx import Attribute, Color, Element, ValueType, Vec2, Vec4
    from srctools.math import Angle, Vec
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: srctools. Install with: python3 -m pip install srctools"
    ) from exc

try:
    from PIL import Image
    from PIL import ImageFile
    from PIL import ImageOps
    from PIL import ImageChops
    from PIL import ImageEnhance
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    Image = None

import roblox_to_vmf as s1

Vector3 = Tuple[float, float, float]
Face = List[int]
Box6 = Tuple[float, float, float, float, float, float]


# CS2-relative character scaling:
# ~72 source units player hull height / ~5.625 Roblox avatar studs ~= 12.8 units per stud.
DEFAULT_SCALE = 12.8
DEFAULT_CYLINDER_SIDES = s1.DEFAULT_CYLINDER_SIDES
DEFAULT_UV_STUDS_PER_TILE = 10.0
DEFAULT_MATERIAL_TEXTURED = "materials/dev/dev_measuregeneric01.vmat"
DEFAULT_MATERIAL_NOTEXTURE = "materials/tools/toolsnodraw.vmat"
SKYBOX_MATERIAL = "materials/tools/toolsskybox.vmat"

DEFAULT_ENV_SKY_PROPS: Dict[str, str] = {
    "targetname": "sky",
    "vscripts": "",
    "parentname": "",
    "parentAttachmentName": "",
    "local.origin": "",
    "local.angles": "",
    "local.scales": "",
    "useLocalOffset": "0",
    "StartDisabled": "0",
    "skyname": "materials/editor/toolscene_lighting_sky_dust.vmat",
    "tint_color": "255 255 255",
    "brightnessscale": "1.0",
}

DEFAULT_LIGHT_ENVIRONMENT_PROPS: Dict[str, str] = {
    "targetname": "",
    "vscripts": "",
    "parentname": "",
    "parentAttachmentName": "",
    "local.origin": "",
    "local.angles": "",
    "local.scales": "",
    "useLocalOffset": "0",
    "enabled": "1",
    "color": "248 249 232",
    "brightness": "2.25",
    "range": "512",
    "castshadows": "1",
    "nearclipplane": "1",
    "shadowpriority": "-1",
    "style": "0",
    "pattern": "",
    "fademindist": "-250",
    "fademaxdist": "1250",
    "rendertocubemaps": "1",
    "priority": "0",
    "lightgroup": "",
    "bouncescale": "1.0",
    "renderdiffuse": "1",
    "renderspecular": "1",
    "rendertransmissive": "1",
    "directlight": "1",
    "indirectlight": "1",
    "angulardiameter": "0.5",
    "cascadecrossfade": ".1",
    "cascadedistancefade": ".05",
    "skycolor": "138 200 255",
    "skyintensity": "0.875",
    "skytexture": "sky",
    "skytexturescale": "1.0",
    "skyambientbounce": "0 0 0",
    "sunlightminbrightness": "32",
    "skydirectlight": "1",
    "skybouncescale": "1.0",
    "brightnessscale": "1.0",
    "ambient_occlusion": "0",
    "max_occlusion_distance": "16.0",
    "fully_occluded_fraction": "1.0",
    "occlusion_exponent": "1.0",
    "baked_light_indexing": "1",
    "minroughness": "0",
    "clientSideEntity": "1",
    "allow_sst_generation": "0",
    "ambient_occlusion_proxy_override": "0",
    "ambient_occlusion_proxy_position_0": "0 0 0",
    "ambient_occlusion_proxy_position_1": "0 0 0",
    "ambient_occlusion_proxy_position_2": "0 0 0",
    "ambient_occlusion_proxy_position_3": "0 0 0",
    "ambient_occlusion_proxy_cone_angle_0": "0.3",
    "ambient_occlusion_proxy_cone_angle_1": "0.3",
    "ambient_occlusion_proxy_cone_angle_2": "0.3",
    "ambient_occlusion_proxy_cone_angle_3": "0.3",
    "ambient_occlusion_proxy_strength_0": "0.5",
    "ambient_occlusion_proxy_strength_1": "0.5",
    "ambient_occlusion_proxy_strength_2": "0.5",
    "ambient_occlusion_proxy_strength_3": "0.5",
    "ambient_occlusion_proxy_ambient_strength": "1.0",
}


CS2_MATERIAL_MAP: Dict[str, str] = {
    # Use only material paths verified in the user's working CS2 sample map.
    # Most solids use dev_measuregeneric01, reflective surfaces use reflectivity_30.
    "Plastic": "materials/dev/dev_measuregeneric01.vmat",
    "SmoothPlastic": "materials/dev/dev_measuregeneric01.vmat",
    "Neon": "materials/dev/reflectivity_30.vmat",
    "Metal": "materials/dev/reflectivity_30.vmat",
    "DiamondPlate": "materials/dev/reflectivity_30.vmat",
    "CorrodedMetal": "materials/dev/reflectivity_30.vmat",
    "Foil": "materials/dev/reflectivity_30.vmat",
    "Wood": "materials/dev/dev_measuregeneric01.vmat",
    "WoodPlanks": "materials/dev/dev_measuregeneric01.vmat",
    "Grass": "materials/dev/dev_measuregeneric01.vmat",
    "Slate": "materials/dev/dev_measuregeneric01.vmat",
    "Concrete": "materials/dev/dev_measuregeneric01.vmat",
    "Brick": "materials/dev/dev_measuregeneric01.vmat",
    "Pebble": "materials/dev/dev_measuregeneric01.vmat",
    "Cobblestone": "materials/dev/dev_measuregeneric01.vmat",
    "Rock": "materials/dev/dev_measuregeneric01.vmat",
    "Sand": "materials/dev/dev_measuregeneric01.vmat",
    "Fabric": "materials/dev/dev_measuregeneric01.vmat",
    "Granite": "materials/dev/dev_measuregeneric01.vmat",
    "Marble": "materials/dev/dev_measuregeneric01.vmat",
    "Basalt": "materials/dev/dev_measuregeneric01.vmat",
    "CrackedLava": "materials/dev/dev_measuregeneric01.vmat",
    "Asphalt": "materials/dev/dev_measuregeneric01.vmat",
    "Mud": "materials/dev/dev_measuregeneric01.vmat",
    "Ground": "materials/dev/dev_measuregeneric01.vmat",
    "Ice": "materials/dev/reflectivity_30.vmat",
    "Snow": "materials/dev/dev_measuregeneric01.vmat",
    "Glacier": "materials/dev/reflectivity_30.vmat",
    "Glass": "materials/dev/reflectivity_30.vmat",
    "ForceField": "materials/dev/reflectivity_30.vmat",
}

CUSTOM_MATERIAL_MODE_STOCK = "stock"
CUSTOM_MATERIAL_MODE_ROBLOX = "roblox_custom"
CUSTOM_MATERIAL_MODE_ROBLOX_LIBRARY = "roblox_library"

ROBLOX_MATERIAL_ALIASES: Dict[str, str] = {
    "SmoothPlastic": "Plastic",
    "Neon": "Plastic",
    "ForceField": "Glass",
}

ROBLOX_LIBRARY_EXTRA_NAMES = ["SmoothPlastic", "Neon", "ForceField"]

VTEX_TEMPLATE = """<!-- dmx encoding keyvalues2_noids 1 format vtex 1 -->
"CDmeVtex"
{
    "m_inputTextureArray" "element_array"
    [
        "CDmeInputTexture"
        {
            "m_name" "string" "InputTexture0"
            "m_fileName" "string" "__FILE_NAME__"
            "m_colorSpace" "string" "srgb"
            "m_typeString" "string" "2D"
            "m_imageProcessorArray" "element_array"
            [
                "CDmeImageProcessor"
                {
                    "m_algorithm" "string" "None"
                    "m_stringArg" "string" ""
                    "m_vFloat4Arg" "vector4" "0 0 0 0"
                }
            ]
        }
    ]
    "m_outputTypeString" "string" "2D"
    "m_outputFormat" "string" "DXT5"
    "m_outputClearColor" "vector4" "0 0 0 0"
    "m_nOutputMinDimension" "int" "0"
    "m_nOutputMaxDimension" "int" "0"
    "m_textureOutputChannelArray" "element_array"
    [
        "CDmeTextureOutputChannel"
        {
            "m_inputTextureArray" "string_array" [ "InputTexture0" ]
            "m_srcChannels" "string" "rgba"
            "m_dstChannels" "string" "rgba"
            "m_mipAlgorithm" "CDmeImageProcessor"
            {
                "m_algorithm" "string" "Box"
                "m_stringArg" "string" ""
                "m_vFloat4Arg" "vector4" "0 0 0 0"
            }
            "m_outputColorSpace" "string" "srgb"
        }
    ]
    "m_vClamp" "vector3" "0 0 0"
    "m_bNoLod" "bool" "0"
}
"""

VMAT_TEMPLATE = """<!-- kv3 encoding:text:version{e21c7f3c-8a33-41c5-9977-a76d3a32aa0d} format:generic:version{7412167c-06e9-4698-aff2-e63eb59037e7} -->
{
    shader = "csgo_unlitgeneric.vfx"
    F_PAINT_VERTEX_COLORS = 1
    F_TRANSLUCENT = 0
    F_BLEND_MODE = 0

    g_vColorTint = [1, 1, 1, 1]
    TextureColor = resource:"__COLOR_VTEX_RESOURCE__"
    g_tColor = resource:"__COLOR_VTEX_RESOURCE__"
}
"""


@dataclass
class NodeIds:
    next_node_id: int = 1

    def alloc(self) -> int:
        out = self.next_node_id
        self.next_node_id += 1
        return out


def empty_array(name: str, value_type: ValueType) -> Attribute:
    return Attribute.array(name, value_type)


def vec_sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        (a[1] * b[2]) - (a[2] * b[1]),
        (a[2] * b[0]) - (a[0] * b[2]),
        (a[0] * b[1]) - (a[1] * b[0]),
    )


def vec_dot(a: Vector3, b: Vector3) -> float:
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def vec_len(v: Vector3) -> float:
    return math.sqrt(vec_dot(v, v))


def vec_norm(v: Vector3) -> Vector3:
    l = vec_len(v)
    if l <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / l, v[1] / l, v[2] / l)


def tangent_from_normal(normal: Vector3) -> Tuple[float, float, float, float]:
    up = (0.0, 0.0, 1.0)
    if abs(vec_dot(normal, up)) > 0.95:
        up = (0.0, 1.0, 0.0)
    t = vec_norm(vec_cross(up, normal))
    return (t[0], t[1], t[2], 1.0)


def triplanar_uv(pos: Vector3, normal: Vector3, scale: float = 0.03125) -> Tuple[float, float]:
    wx, wy, wz = abs(normal[0]), abs(normal[1]), abs(normal[2])
    top = (pos[0] * wz, -pos[1] * wz)
    front = (pos[0] * wy, -pos[2] * wy)
    side = (pos[1] * wx, -pos[2] * wx)
    return ((top[0] + front[0] + side[0]) * scale, (top[1] + front[1] + side[1]) * scale)


def custom_material_slug(material_name: str) -> str:
    return s1.sanitize_name(material_name or "material").lower()


def normalize_material_lookup_key(material_name: str) -> str:
    out = []
    for ch in str(material_name or "").lower():
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def canonical_roblox_material_name(material_name: str) -> str:
    raw = str(material_name or "").strip()
    if not raw:
        return "Plastic"
    for alias_name, source_name in ROBLOX_MATERIAL_ALIASES.items():
        if alias_name.lower() == raw.lower():
            return source_name
    return raw


def map_material(material_name: str, textured: bool, material_mode: str) -> str:
    if not textured:
        return DEFAULT_MATERIAL_NOTEXTURE
    if material_mode in (CUSTOM_MATERIAL_MODE_ROBLOX, CUSTOM_MATERIAL_MODE_ROBLOX_LIBRARY):
        return f"materials/roblox_generated/{custom_material_slug(material_name)}.vmat"
    return CS2_MATERIAL_MAP.get(material_name, DEFAULT_MATERIAL_TEXTURED)


def find_default_roblox_texture_source() -> Optional[Path]:
    candidates = [
        os.environ.get("ROBLOX_MATERIALS_MODERN_DIR", ""),
        "/tmp/Roblox-Materials/Modern",
        "/tmp/Roblox-Materials-main/Modern",
    ]
    for raw in candidates:
        path = Path(raw).expanduser() if raw else None
        if path and path.is_dir():
            return path
    return None


def build_modern_material_index(texture_source_dir: Optional[Path]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not texture_source_dir or not texture_source_dir.is_dir():
        return index
    for child in texture_source_dir.iterdir():
        if child.is_dir():
            key = normalize_material_lookup_key(child.name)
            if key:
                index[key] = child.name
    return index


def resolve_roblox_texture_file(
    material_name: str,
    texture_source_dir: Optional[Path],
    modern_index: Dict[str, str],
) -> Optional[Path]:
    if not texture_source_dir or not texture_source_dir.is_dir():
        return None

    candidates = [str(material_name or "").strip(), canonical_roblox_material_name(material_name)]
    seen = set()
    for candidate_name in candidates:
        key = normalize_material_lookup_key(candidate_name)
        if not key or key in seen:
            continue
        seen.add(key)
        real_dir_name = modern_index.get(key)
        if not real_dir_name:
            continue
        color = texture_source_dir / real_dir_name / "color.png"
        if color.is_file():
            return color
    return None


def normalize_texture_png(src_png: Path, dst_png: Path) -> None:
    # Preserve Roblox texture bytes exactly.
    shutil.copyfile(src_png, dst_png)


def create_placeholder_texture_png(dst_png: Path, size: int = 64) -> None:
    if Image is None:
        # Minimal hard fail-safe: copy an existing generated texture if available.
        for sibling in dst_png.parent.glob("*.png"):
            if sibling.is_file():
                shutil.copyfile(sibling, dst_png)
                return
        raise RuntimeError("Pillow is required to generate placeholder textures")

    image = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    image.save(dst_png, format="PNG")


def create_baked_plastic_texture_png(
    dst_png: Path,
    *,
    height_png: Optional[Path] = None,
    normal_png: Optional[Path] = None,
) -> bool:
    if Image is None:
        return False

    detail_src: Optional[Path] = None
    if height_png and height_png.is_file():
        detail_src = height_png
    elif normal_png and normal_png.is_file():
        detail_src = normal_png
    if detail_src is None:
        return False

    with Image.open(detail_src) as im:
        # Keep source resolution so the resulting texture density matches other materials.
        gray = ImageOps.grayscale(im)
        gray = ImageOps.autocontrast(gray)
        # Push mid-frequency details so it reads as subtle plastic roughness.
        gray = ImageEnhance.Contrast(gray).enhance(1.4)

        # Start from Roblox-like neutral plastic gray and modulate with detail.
        base = Image.new("RGB", gray.size, (178, 178, 178))
        # Convert grayscale detail into a signed modulation around 128.
        mod = ImageOps.colorize(gray, black=(84, 84, 84), white=(236, 236, 236))
        baked = ImageChops.multiply(base, mod)
        baked = ImageEnhance.Brightness(baked).enhance(1.03)
        baked.convert("RGBA").save(dst_png, format="PNG")
    return True


def get_all_roblox_library_material_names(texture_source_dir: Optional[Path]) -> List[str]:
    names = set(CS2_MATERIAL_MAP.keys())
    for extra in ROBLOX_LIBRARY_EXTRA_NAMES:
        names.add(extra)
    if texture_source_dir and texture_source_dir.is_dir():
        for child in texture_source_dir.iterdir():
            if child.is_dir():
                names.add(child.name)
    names.discard("Air")
    return sorted(names)


def emit_roblox_custom_material_pack(
    output_dir: Path,
    material_names: Sequence[str],
    texture_source_override: str,
    generate_all: bool = False,
) -> Optional[Path]:
    if not material_names and not generate_all:
        return None

    texture_source_dir: Optional[Path] = None
    if texture_source_override.strip():
        candidate = Path(texture_source_override).expanduser()
        if candidate.is_dir():
            texture_source_dir = candidate
    if texture_source_dir is None:
        texture_source_dir = find_default_roblox_texture_source()
    modern_index = build_modern_material_index(texture_source_dir)

    pack_root = output_dir / "cs2_material_pack"
    materials_root = pack_root / "materials" / "roblox_generated"
    textures_root = materials_root / "textures"
    if materials_root.exists():
        shutil.rmtree(materials_root)
    textures_root.mkdir(parents=True, exist_ok=True)

    if generate_all:
        used = get_all_roblox_library_material_names(texture_source_dir)
    else:
        used = sorted({str(m).strip() for m in material_names if str(m).strip() and str(m).strip() != "Air"})
    for material_name in used:
        slug = custom_material_slug(material_name)
        png_out = textures_root / f"{slug}.png"
        vtex_out = textures_root / f"{slug}.vtex"
        vmat_out = materials_root / f"{slug}.vmat"

        canonical_name = canonical_roblox_material_name(material_name)
        canonical_key = normalize_material_lookup_key(canonical_name)
        canonical_dir = modern_index.get(canonical_key)
        height_src = None
        normal_src = None
        if canonical_dir and texture_source_dir:
            height_src = texture_source_dir / canonical_dir / "height.png"
            normal_src = texture_source_dir / canonical_dir / "normal.png"

        src_png = resolve_roblox_texture_file(material_name, texture_source_dir, modern_index)
        if src_png and src_png.is_file():
            normalize_texture_png(src_png, png_out)
        else:
            # Plastic in Roblox Modern has no color map; prebake height/normal detail
            # into a neutral plastic colormap for better visual fidelity.
            baked = False
            if canonical_key == "plastic":
                baked = create_baked_plastic_texture_png(
                    png_out,
                    height_png=height_src if height_src and height_src.is_file() else None,
                    normal_png=normal_src if normal_src and normal_src.is_file() else None,
                )
            if not baked:
                create_placeholder_texture_png(png_out, size=64)

        texture_file_name = f"materials/roblox_generated/textures/{slug}.png"
        vtex_out.write_text(VTEX_TEMPLATE.replace("__FILE_NAME__", texture_file_name), encoding="utf-8")

        color_vtex_resource = f"materials/roblox_generated/textures/{slug}.vtex"
        vmat_out.write_text(VMAT_TEMPLATE.replace("__COLOR_VTEX_RESOURCE__", color_vtex_resource), encoding="utf-8")

    readme = pack_root / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "CS2 custom material pack generated by roblox_to_vmap.py",
                "",
                "1) Copy the 'materials' folder from this pack into your CS2 addon content path:",
                "   Counter-Strike Global Offensive/content/csgo_addons/<your_addon>/",
                "",
                "2) Open your addon in Workshop Tools so Source2 can compile the .vtex/.vmat resources.",
                "",
                "3) Ensure your .vmap references materials/roblox_generated/*.vmat (this export does).",
                "   For one-time library setup, pass --generate-roblox-material-library.",
                "",
                f"Texture source used: {str(texture_source_dir) if texture_source_dir else '(none found; .vtex/.vmat emitted without source PNG copies)'}",
            ]
        ),
        encoding="utf-8",
    )

    # Compile-stability aliases for known-problematic materials in some CS2 toolchains.
    # Keep VMAP-facing names intact, but redirect to known-good color textures.
    for alias_slug, target_slug in (("smoothplastic", "plastic"), ("woodplanks", "wood")):
        alias_vmat = materials_root / f"{alias_slug}.vmat"
        target_vtex_resource = f"materials/roblox_generated/textures/{target_slug}.vtex"
        alias_vmat.write_text(
            VMAT_TEMPLATE.replace("__COLOR_VTEX_RESOURCE__", target_vtex_resource),
            encoding="utf-8",
        )

    return pack_root


def make_mapnode_base(el: Element, node_ids: NodeIds) -> None:
    el["origin"] = Vec(0.0, 0.0, 0.0)
    el["angles"] = Angle(0.0, 0.0, 0.0)
    el["scales"] = Vec(1.0, 1.0, 1.0)
    el["nodeID"] = node_ids.alloc()
    # Keep this field for save-path compatibility in some Workshop Tools builds.
    el["referenceID"] = 0
    el["children"] = empty_array("children", ValueType.ELEMENT)
    el["editorOnly"] = False
    el["force_hidden"] = False
    el["transformLocked"] = False
    el["variableTargetKeys"] = empty_array("variableTargetKeys", ValueType.STRING)
    el["variableNames"] = empty_array("variableNames", ValueType.STRING)


def make_plug_data(name: str = "relayPlugData") -> Element:
    plug = Element(name, "DmePlugList")
    plug["names"] = empty_array("names", ValueType.STRING)
    plug["dataTypes"] = empty_array("dataTypes", ValueType.INTEGER)
    plug["plugTypes"] = empty_array("plugTypes", ValueType.INTEGER)
    plug["descriptions"] = empty_array("descriptions", ValueType.STRING)
    return plug


def make_entity_props(name: str, kv: Dict[str, str]) -> Element:
    props = Element(name, "EditGameClassProps")
    for key, value in kv.items():
        props[key] = str(value)
    return props


def component_from_part(part: s1.Part, scale: float, offset: Vector3, cylinder_sides: int, rectangles_only: bool) -> Optional[Tuple[List[Vector3], List[Face]]]:
    if part.shape == "block":
        local = s1.block_local_vertices(part.size)
        verts = [s1.source_transform(s1.apply_cframe(part.cframe, v), scale, offset) for v in local]
        faces = [list(face) for face in s1.block_face_indices()]
        return verts, faces

    if rectangles_only:
        # Force brush-like rectangular solids for every non-block part.
        local = s1.block_local_vertices(part.size)
        verts = [s1.source_transform(s1.apply_cframe(part.cframe, v), scale, offset) for v in local]
        faces = [list(face) for face in s1.block_face_indices()]
        return verts, faces

    if part.shape == "cylinder":
        top_local, bot_local = s1.cylinder_local_rings(part.size, cylinder_sides)
        top = [s1.source_transform(s1.apply_cframe(part.cframe, v), scale, offset) for v in top_local]
        bot = [s1.source_transform(s1.apply_cframe(part.cframe, v), scale, offset) for v in bot_local]
        n = len(top)
        verts = top + bot
        faces: List[Face] = []

        for i in range(n):
            j = (i + 1) % n
            top_i, top_j = i, j
            bot_i, bot_j = n + i, n + j
            faces.append([bot_i, bot_j, top_j, top_i])

        faces.append(list(range(0, n)))
        faces.append(list(range((2 * n) - 1, n - 1, -1)))
        return verts, faces

    return None


def component_from_source_box(min_x: float, max_x: float, min_y: float, max_y: float, min_z: float, max_z: float) -> Tuple[List[Vector3], List[Face]]:
    verts = [
        (min_x, min_y, min_z),
        (max_x, min_y, min_z),
        (max_x, max_y, min_z),
        (min_x, max_y, min_z),
        (min_x, min_y, max_z),
        (max_x, min_y, max_z),
        (max_x, max_y, max_z),
        (min_x, max_y, max_z),
    ]
    faces = [list(face) for face in s1.block_face_indices()]
    return verts, faces


def verts_to_axis_aligned_box(verts: Sequence[Vector3], eps: float = 1e-5) -> Optional[Box6]:
    if len(verts) != 8:
        return None
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    if (max_x - min_x) <= eps or (max_y - min_y) <= eps or (max_z - min_z) <= eps:
        return None

    # Strictly require every vertex to sit on a box corner plane.
    def near(a: float, b: float) -> bool:
        return abs(a - b) <= eps

    for x, y, z in verts:
        if not (near(x, min_x) or near(x, max_x)):
            return None
        if not (near(y, min_y) or near(y, max_y)):
            return None
        if not (near(z, min_z) or near(z, max_z)):
            return None

    return (min_x, max_x, min_y, max_y, min_z, max_z)


def _q(value: float, precision: int = 5) -> float:
    return round(float(value), precision)


def _normalize_box(box: Box6) -> Box6:
    min_x, max_x, min_y, max_y, min_z, max_z = box
    if min_x > max_x:
        min_x, max_x = max_x, min_x
    if min_y > max_y:
        min_y, max_y = max_y, min_y
    if min_z > max_z:
        min_z, max_z = max_z, min_z
    return (_q(min_x), _q(max_x), _q(min_y), _q(max_y), _q(min_z), _q(max_z))


def merge_boxes_strict(boxes: Sequence[Box6], eps: float = 1e-5) -> List[Box6]:
    current = [_normalize_box(b) for b in boxes]

    def merge_axis(src: List[Box6], axis: str) -> List[Box6]:
        groups: Dict[Tuple[float, float, float, float], List[Tuple[float, float]]] = {}
        passthrough: List[Box6] = []

        for b in src:
            min_x, max_x, min_y, max_y, min_z, max_z = b
            if axis == "x":
                key = (min_y, max_y, min_z, max_z)
                interval = (min_x, max_x)
            elif axis == "y":
                key = (min_x, max_x, min_z, max_z)
                interval = (min_y, max_y)
            else:
                key = (min_x, max_x, min_y, max_y)
                interval = (min_z, max_z)
            groups.setdefault(key, []).append(interval)

        out: List[Box6] = []
        for key, intervals in groups.items():
            intervals.sort(key=lambda it: (it[0], it[1]))
            cur_start, cur_end = intervals[0]
            for start, end in intervals[1:]:
                # Strict visual-safe merge: only perfectly touching intervals.
                if abs(start - cur_end) <= eps:
                    cur_end = end
                else:
                    if axis == "x":
                        out.append((cur_start, cur_end, key[0], key[1], key[2], key[3]))
                    elif axis == "y":
                        out.append((key[0], key[1], cur_start, cur_end, key[2], key[3]))
                    else:
                        out.append((key[0], key[1], key[2], key[3], cur_start, cur_end))
                    cur_start, cur_end = start, end
            if axis == "x":
                out.append((cur_start, cur_end, key[0], key[1], key[2], key[3]))
            elif axis == "y":
                out.append((key[0], key[1], cur_start, cur_end, key[2], key[3]))
            else:
                out.append((key[0], key[1], key[2], key[3], cur_start, cur_end))

        out.extend(passthrough)
        return [_normalize_box(b) for b in out]

    changed = True
    while changed:
        before_count = len(current)
        for axis in ("x", "y", "z"):
            current = merge_axis(current, axis)
        # Dedupe exact boxes introduced by multiple passes.
        dedup = []
        seen = set()
        for b in current:
            if b not in seen:
                seen.add(b)
                dedup.append(b)
        current = dedup
        changed = len(current) < before_count

    return current


def build_polygon_mesh_element(
    name: str,
    components: Sequence[Tuple[List[Vector3], List[Face], Tuple[int, int, int]]],
    material_path: str,
    node_ids: NodeIds,
    uv_scale: float,
    lean_mesh: bool,
    use_vertex_tint: bool,
) -> Optional[Element]:
    positions: List[Vector3] = []
    faces: List[Face] = []
    face_tints: List[Tuple[int, int, int]] = []

    for verts, comp_faces, comp_tint in components:
        base = len(positions)
        positions.extend(verts)
        for face in comp_faces:
            if len(face) >= 3:
                faces.append([base + idx for idx in face])
                face_tints.append(comp_tint)

    if not positions or not faces:
        return None

    half_orig: List[int] = []
    half_dest: List[int] = []
    half_next: List[int] = []
    half_face: List[int] = []
    face_edge_indices: List[int] = []

    for f_idx, face in enumerate(faces):
        start = len(half_orig)
        count = len(face)
        face_edge_indices.append(start)
        for i in range(count):
            o = face[i]
            d = face[(i + 1) % count]
            half_orig.append(o)
            half_dest.append(d)
            half_next.append(start + ((i + 1) % count))
            half_face.append(f_idx)

    lookup: Dict[Tuple[int, int], int] = {}
    for i, (o, d) in enumerate(zip(half_orig, half_dest)):
        lookup[(o, d)] = i

    half_twin: List[int] = []
    for o, d in zip(half_orig, half_dest):
        half_twin.append(lookup.get((d, o), -1))

    undirected_to_edge: Dict[Tuple[int, int], int] = {}
    edge_data_indices: List[int] = []
    edge_flags: List[int] = []
    for o, d in zip(half_orig, half_dest):
        key = (o, d) if o < d else (d, o)
        idx = undirected_to_edge.get(key)
        if idx is None:
            idx = len(undirected_to_edge)
            undirected_to_edge[key] = idx
            edge_flags.append(0)
        edge_data_indices.append(idx)

    vertex_edge_indices = [-1] * len(positions)
    for he_idx, orig in enumerate(half_orig):
        if vertex_edge_indices[orig] == -1:
            vertex_edge_indices[orig] = he_idx
    for i, v in enumerate(vertex_edge_indices):
        if v == -1:
            vertex_edge_indices[i] = 0

    normals: List[Vector3] = []
    tangents: List[Tuple[float, float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    uvs1: List[Tuple[float, float]] = []
    vp_blend: List[Tuple[float, float, float, float]] = []
    vp_tint: List[Tuple[float, float, float, float]] = []

    face_normals: List[Vector3] = []
    for face in faces:
        if len(face) >= 3:
            a = positions[face[0]]
            b = positions[face[1]]
            c = positions[face[2]]
            n = vec_norm(vec_cross(vec_sub(b, a), vec_sub(c, a)))
        else:
            n = (0.0, 0.0, 1.0)
        face_normals.append(n)

    for he_idx, dest in enumerate(half_dest):
        f_idx = half_face[he_idx]
        n = face_normals[f_idx]
        t = tangent_from_normal(n)
        uv = triplanar_uv(positions[dest], n, scale=uv_scale)

        normals.append(n)
        tangents.append(t)
        uvs.append(uv)
        if not lean_mesh:
            uvs1.append(uv)
            vp_blend.append((0.0, 0.0, 0.0, 0.0))
        if use_vertex_tint:
            tint = face_tints[f_idx] if f_idx < len(face_tints) else (255, 255, 255)
            vp_tint.append((tint[0] / 255.0, tint[1] / 255.0, tint[2] / 255.0, 1.0))
        elif not lean_mesh:
            vp_tint.append((0.0, 0.0, 0.0, 0.0))

    mesh = Element(name, "CDmePolygonMesh")
    make_mapnode_base(mesh, node_ids)

    mesh["vertexEdgeIndices"] = vertex_edge_indices
    mesh["vertexDataIndices"] = list(range(len(positions)))

    mesh["edgeVertexIndices"] = half_dest
    mesh["edgeOppositeIndices"] = half_twin
    mesh["edgeNextIndices"] = half_next
    mesh["edgeFaceIndices"] = half_face
    mesh["edgeDataIndices"] = edge_data_indices
    mesh["edgeVertexDataIndices"] = list(range(len(half_dest)))

    mesh["faceEdgeIndices"] = face_edge_indices
    mesh["faceDataIndices"] = list(range(len(faces)))

    mesh["materials"] = [material_path]

    vertex_data = Element("vertexData", "CDmePolygonMeshDataArray")
    vertex_data["size"] = len(positions)
    vertex_streams = empty_array("streams", ValueType.ELEMENT)

    pos_stream = Element("position:0", "CDmePolygonMeshDataStream")
    pos_stream["standardAttributeName"] = "position"
    pos_stream["semanticName"] = "position"
    pos_stream["semanticIndex"] = 0
    pos_stream["vertexBufferLocation"] = 0
    pos_stream["dataStateFlags"] = 3
    pos_stream["data"] = [Vec(x, y, z) for x, y, z in positions]
    vertex_streams.append(pos_stream)

    vertex_data["streams"] = vertex_streams
    mesh["vertexData"] = vertex_data

    face_vertex_data = Element("faceVertexData", "CDmePolygonMeshDataArray")
    face_vertex_data["size"] = len(half_dest)
    fv_streams = empty_array("streams", ValueType.ELEMENT)

    def mk_stream(stream_name: str, std_attr: str, flags: int, values: Iterable[object]) -> Element:
        s = Element(stream_name, "CDmePolygonMeshDataStream")
        sem_name, sem_idx = stream_name.split(":", 1)
        s["standardAttributeName"] = std_attr
        s["semanticName"] = sem_name
        s["semanticIndex"] = int(sem_idx)
        s["vertexBufferLocation"] = 0
        s["dataStateFlags"] = flags
        s["data"] = list(values)
        return s

    fv_streams.append(mk_stream("texcoord:0", "texcoord", 1, [Vec2(u, v) for u, v in uvs]))
    fv_streams.append(mk_stream("normal:0", "normal", 1, [Vec(x, y, z) for x, y, z in normals]))
    if use_vertex_tint:
        fv_streams.append(mk_stream("VertexPaintTintColor:0", "VertexPaintTintColor", 1, [Vec4(*v) for v in vp_tint]))
    if not lean_mesh:
        fv_streams.append(mk_stream("texcoord:1", "texcoord1", 1, [Vec2(u, v) for u, v in uvs1]))
        fv_streams.append(mk_stream("VertexPaintBlendParams:0", "VertexPaintBlendParams", 1, [Vec4(*v) for v in vp_blend]))
        fv_streams.append(mk_stream("tangent:0", "tangent", 1, [Vec4(*t) for t in tangents]))

    face_vertex_data["streams"] = fv_streams
    mesh["faceVertexData"] = face_vertex_data

    edge_data = Element("edgeData", "CDmePolygonMeshDataArray")
    edge_data["size"] = len(edge_flags)
    edge_streams = empty_array("streams", ValueType.ELEMENT)
    edge_streams.append(mk_stream("flags:0", "flags", 3, edge_flags))
    edge_data["streams"] = edge_streams
    mesh["edgeData"] = edge_data

    face_data = Element("faceData", "CDmePolygonMeshDataArray")
    face_data["size"] = len(faces)
    face_streams = empty_array("streams", ValueType.ELEMENT)
    face_streams.append(mk_stream("materialindex:0", "materialindex", 8, [0] * len(faces)))
    face_streams.append(mk_stream("flags:0", "flags", 3, [0] * len(faces)))
    face_data["streams"] = face_streams
    mesh["faceData"] = face_data

    subdiv = Element("subdivisionData", "CDmePolygonMeshSubdivisionData")
    subdiv["subdivisionLevels"] = [0] * 8
    subdiv["streams"] = empty_array("streams", ValueType.ELEMENT)
    mesh["subdivisionData"] = subdiv

    return mesh


def make_cmap_mesh(name: str, mesh_data: Element, tint_rgb: Tuple[int, int, int], node_ids: NodeIds) -> Element:
    mesh = Element(name, "CMapMesh")
    make_mapnode_base(mesh, node_ids)

    mesh["cubeMapName"] = ""
    mesh["lightGroup"] = ""
    mesh["visexclude"] = False
    mesh["renderwithdynamic"] = False
    mesh["disableHeightDisplacement"] = False
    mesh["fademindist"] = -1.0
    mesh["fademaxdist"] = 0.0
    mesh["bakelighting"] = True
    mesh["precomputelightprobes"] = True
    mesh["renderToCubemaps"] = True
    mesh["disableShadows"] = False
    mesh["smoothingAngle"] = 40.0
    mesh["tintColor"] = Color(tint_rgb[0], tint_rgb[1], tint_rgb[2], 255)
    mesh["renderAmt"] = 255
    mesh["physicsType"] = "default"
    mesh["physicsCollisionProperty"] = ""
    mesh["physicsGroup"] = ""
    mesh["physicsInteractsAs"] = ""
    mesh["physicsInteractsWith"] = ""
    mesh["physicsInteractsExclude"] = ""
    mesh["meshData"] = mesh_data
    mesh["disablemerging"] = False
    mesh["keep_vertices"] = False
    mesh["useAsOccluder"] = False
    mesh["physicsMissingDetailLayers"] = ""
    mesh["physicsIncludedDetailLayers"] = ""
    mesh["physicsSimplificationOverride"] = False
    mesh["physicsSimplificationError"] = 0.0
    return mesh


def make_cmap_entity(name: str, classname: str, origin: Vector3, node_ids: NodeIds, extra_props: Optional[Dict[str, str]] = None) -> Element:
    ent = Element(name, "CMapEntity")
    make_mapnode_base(ent, node_ids)
    ent["origin"] = Vec(origin[0], origin[1], origin[2])

    ent["relayPlugData"] = make_plug_data()
    ent["connectionsData"] = empty_array("connectionsData", ValueType.ELEMENT)
    props = {
        "classname": classname,
        "origin": f"{origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}",
        "angles": "0 0 0",
    }
    if extra_props:
        for k, v in extra_props.items():
            props[str(k)] = str(v)
        props["classname"] = classname
        props["origin"] = f"{origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}"
    ent["entity_properties"] = make_entity_props("entity_properties", props)
    ent["hitNormal"] = Vec(0.0, 0.0, 1.0)
    ent["isProceduralEntity"] = False
    return ent


def make_world(children: List[Element], node_ids: NodeIds) -> Element:
    world = Element("world", "CMapWorld")
    make_mapnode_base(world, node_ids)
    world["children"] = children if children else empty_array("children", ValueType.ELEMENT)

    world["relayPlugData"] = make_plug_data()
    world["connectionsData"] = empty_array("connectionsData", ValueType.ELEMENT)
    world["entity_properties"] = make_entity_props("entity_properties", {"classname": "worldspawn"})

    world["nextDecalID"] = 1
    world["fixupEntityNames"] = True
    world["mapUsageType"] = "standard"
    return world


def make_visibility(node_ids: NodeIds) -> Element:
    # Valve's key/name typo is preserved in VMAPs for compatibility.
    vis = Element("visbility", "CVisibilityMgr")
    make_mapnode_base(vis, node_ids)
    vis["nodes"] = empty_array("nodes", ValueType.ELEMENT)
    vis["hiddenFlags"] = empty_array("hiddenFlags", ValueType.INTEGER)
    return vis


def make_map_variables() -> Element:
    varset = Element("mapVariables", "CMapVariableSet")
    # Hammer expects int_array here.
    varset["variableAndChoiceOrder"] = empty_array("variableAndChoiceOrder", ValueType.INTEGER)
    varset["variableNames"] = empty_array("variableNames", ValueType.STRING)
    varset["variableValues"] = empty_array("variableValues", ValueType.STRING)
    varset["variableTypeNames"] = empty_array("variableTypeNames", ValueType.STRING)
    varset["variableTypeParameters"] = empty_array("variableTypeParameters", ValueType.STRING)
    varset["m_ChoiceGroups"] = empty_array("m_ChoiceGroups", ValueType.ELEMENT)
    return varset


def make_root_selection_set() -> Element:
    sel_data = Element("objectSelectionSetData", "CObjectSelectionSetDataElement")
    sel_data["selectedObjects"] = empty_array("selectedObjects", ValueType.ELEMENT)

    root_sel = Element("root", "CMapSelectionSet")
    root_sel["children"] = empty_array("children", ValueType.ELEMENT)
    root_sel["selectionSetName"] = "root"
    root_sel["selectionSetData"] = sel_data
    return root_sel


def make_default_camera(bounds_center: Vector3) -> Element:
    cam = Element("defaultcamera", "CStoredCamera")
    cx, cy, cz = bounds_center
    cam["position"] = Vec(cx - 1024.0, cy - 1024.0, cz + 768.0)
    cam["lookat"] = Vec(cx, cy, cz)
    return cam


def make_stored_cameras() -> Element:
    cams = Element("3dcameras", "CStoredCameras")
    cams["activecamera"] = -1
    cams["cameras"] = empty_array("cameras", ValueType.ELEMENT)
    return cams


def build_vmap(parts: Sequence[s1.Part], spawn_points: Sequence[Vector3], *, textured: bool, material_mode: str, map_name: str, scale: float, uv_studs_per_tile: float, offset: Vector3, cylinder_sides: int, add_skybox: bool, skybox_margin: float, skybox_thickness: float, spawn_height_source: float, lean_mesh: bool, rectangles_only: bool, merge_colors_into_vertex_tint: bool, add_default_lighting: bool, merge_bricks: bool) -> Element:
    uv_studs_per_tile = max(0.001, float(uv_studs_per_tile))
    # Convert desired Roblox studs-per-tile into Source UV scale.
    # source_units_per_tile = uv_studs_per_tile * scale
    # uv = source_units * (1 / source_units_per_tile)
    uv_scale = 1.0 / (uv_studs_per_tile * float(scale))
    # Group by material (and optionally by tint when vertex tint merge is disabled).
    groups: Dict[Tuple[str, Tuple[int, int, int]], List[Tuple[List[Vector3], List[Face], Tuple[int, int, int]]]] = {}

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

    merge_candidates: Dict[Tuple[str, Tuple[int, int, int]], List[Box6]] = {}

    for part in parts:
        comp = component_from_part(part, scale, offset, cylinder_sides, rectangles_only)
        if comp is None:
            continue
        verts, _ = comp
        for p in verts:
            include_point(p)

        mat = map_material(part.material, textured=textured, material_mode=material_mode)
        tint = part.color if textured else (255, 255, 255)
        key_tint = (255, 255, 255) if (textured and merge_colors_into_vertex_tint and mat != SKYBOX_MATERIAL) else tint
        key = (mat, key_tint)

        # Visual-safe merge only for true block parts that are axis-aligned boxes.
        if merge_bricks and part.shape == "block":
            aabb = verts_to_axis_aligned_box(comp[0])
            if aabb is not None:
                merge_candidates.setdefault(key, []).append(aabb)
                continue

        groups.setdefault(key, []).append((comp[0], comp[1], tint))

    if merge_bricks:
        for key, boxes in merge_candidates.items():
            merged = merge_boxes_strict(boxes)
            for box in merged:
                c_verts, c_faces = component_from_source_box(*box)
                groups.setdefault(key, []).append((c_verts, c_faces, key[1]))

    transformed_spawns: List[Vector3] = []
    for sp in s1.dedupe_points(spawn_points):
        sx, sy, sz = s1.source_transform(sp, scale, offset)
        transformed_spawns.append((sx, sy, sz + float(spawn_height_source)))
        include_point((sx, sy, sz))

    if min_x == math.inf:
        min_x, min_y, min_z = -1024.0, -1024.0, -1024.0
        max_x, max_y, max_z = 1024.0, 1024.0, 1024.0

    if add_skybox:
        margin = max(0.0, skybox_margin)
        thickness = max(1.0, skybox_thickness)

        inner_min_x, inner_min_y, inner_min_z = min_x - margin, min_y - margin, min_z - margin
        inner_max_x, inner_max_y, inner_max_z = max_x + margin, max_y + margin, max_z + margin

        outer_min_x, outer_min_y, outer_min_z = inner_min_x - thickness, inner_min_y - thickness, inner_min_z - thickness
        outer_max_x, outer_max_y, outer_max_z = inner_max_x + thickness, inner_max_y + thickness, inner_max_z + thickness

        sky_boxes = [
            (outer_min_x, inner_min_x, outer_min_y, outer_max_y, outer_min_z, outer_max_z),
            (inner_max_x, outer_max_x, outer_min_y, outer_max_y, outer_min_z, outer_max_z),
            (inner_min_x, inner_max_x, outer_min_y, inner_min_y, outer_min_z, outer_max_z),
            (inner_min_x, inner_max_x, inner_max_y, outer_max_y, outer_min_z, outer_max_z),
            (inner_min_x, inner_max_x, inner_min_y, inner_max_y, outer_min_z, inner_min_z),
            (inner_min_x, inner_max_x, inner_min_y, inner_max_y, inner_max_z, outer_max_z),
        ]

        sky_components = [(c[0], c[1], (255, 255, 255)) for c in [component_from_source_box(*bounds) for bounds in sky_boxes]]
        groups[(SKYBOX_MATERIAL, (255, 255, 255))] = groups.get((SKYBOX_MATERIAL, (255, 255, 255)), []) + sky_components

    node_ids = NodeIds()
    world_children: List[Element] = []

    mesh_index = 0
    for (material_path, tint), comps in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mesh_data = build_polygon_mesh_element(
            name=f"meshData_{mesh_index}",
            components=comps,
            material_path=material_path,
            node_ids=node_ids,
            uv_scale=uv_scale,
            lean_mesh=lean_mesh,
            use_vertex_tint=(textured and merge_colors_into_vertex_tint and material_path != SKYBOX_MATERIAL),
        )
        if mesh_data is None:
            continue
        cmap_mesh = make_cmap_mesh(f"mesh_{mesh_index}", mesh_data, tint, node_ids)
        world_children.append(cmap_mesh)
        mesh_index += 1

    for i, sp in enumerate(transformed_spawns):
        world_children.append(make_cmap_entity(f"spawn_{i}", "info_player_start", sp, node_ids))

    if add_default_lighting:
        center_pos = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5)
        world_children.append(make_cmap_entity("env_sky_default", "env_sky", center_pos, node_ids, DEFAULT_ENV_SKY_PROPS))
        world_children.append(make_cmap_entity("light_environment_default", "light_environment", center_pos, node_ids, DEFAULT_LIGHT_ENVIRONMENT_PROPS))

    center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5)

    root = Element(map_name, "CMapRootElement")
    root["isprefab"] = False
    root["editorbuild"] = 400
    root["editorversion"] = 400
    root["showgrid"] = True
    root["snapRotationAngle"] = 15
    root["gridSpacing"] = 64.0
    root["show3DGrid"] = True
    root["itemFile"] = f"maps/{map_name}.vmap"
    root["defaultcamera"] = make_default_camera(center)
    root["3dcameras"] = make_stored_cameras()
    root["world"] = make_world(world_children, node_ids)
    # Valve's root key spelling in VMAP is historically "visbility".
    root["visbility"] = make_visibility(node_ids)
    root["mapVariables"] = make_map_variables()
    root["rootSelectionSet"] = make_root_selection_set()
    root["map_asset_references"] = sorted({material for (material, _tint) in groups.keys()})
    root["m_ReferencedMeshSnapshots"] = empty_array("m_ReferencedMeshSnapshots", ValueType.ELEMENT)
    root["m_bIsCordoning"] = False
    root["m_bCordonsVisible"] = False
    root["nodeInstanceData"] = empty_array("nodeInstanceData", ValueType.ELEMENT)

    return root


def write_vmap_kv2(root: Element, out_path: Path, fmt_ver: int, kv2_encoding_version: int) -> None:
    with out_path.open("wb") as f:
        root.export_kv2(f, fmt_name="vmap", fmt_ver=fmt_ver)

    data = out_path.read_bytes()
    lf = b"\n"
    first_line_end = data.find(lf)
    if first_line_end < 0:
        return
    first_line = data[:first_line_end].rstrip(b"\r")
    expected = b"<!-- dmx encoding keyvalues2 1 format vmap "
    if first_line.startswith(expected):
        tail = first_line[len(expected):]
        if b" -->" in tail:
            fmt_tail = tail.split(b" -->", 1)[0].strip()
        else:
            fmt_tail = tail.strip()
        new_line = (
            b"<!-- dmx encoding keyvalues2 "
            + str(int(kv2_encoding_version)).encode("ascii")
            + b" format vmap "
            + fmt_tail
            + b" -->"
        )
        out_path.write_bytes(new_line + lf + data[first_line_end + 1 :])


def find_dotnet_executable() -> Optional[str]:
    exe = shutil.which("dotnet")
    if exe:
        return exe
    for candidate in ("/tmp/dotnet9/dotnet", "/tmp/dotnet8/dotnet"):
        if os.path.isfile(candidate):
            return candidate
    return None


def find_bin9_converter_dll() -> Optional[str]:
    local = Path(__file__).resolve().parent / "tools" / "vmap_bin9_converter" / "bin" / "Debug" / "net9.0" / "vmap_bin9_converter.dll"
    if local.is_file():
        return str(local)
    return None


def convert_kv2_to_binary9(input_kv2: Path, output_bin9: Path) -> None:
    # Preferred path: external converter (Datamodel.NET) with schema normalization.
    dotnet = find_dotnet_executable()
    dll = find_bin9_converter_dll()
    if dotnet and dll:
        output_bin9.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [dotnet, dll, str(input_kv2), str(output_bin9)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return

    # Fallback: srctools binary writer.
    with input_kv2.open("rb") as f:
        parsed = Element.parse(f)
    if isinstance(parsed, tuple):
        root, fmt_name, fmt_ver = parsed
    else:
        root = parsed
        fmt_name = "vmap"
        fmt_ver = 40
    output_bin9.parent.mkdir(parents=True, exist_ok=True)
    with output_bin9.open("wb") as f:
        root.export_binary(f, fmt_name=fmt_name, fmt_ver=fmt_ver)


def load_parts_and_spawns(input_path: Path, forced_format: str) -> Tuple[List[s1.Part], List[Vector3]]:
    auto_rbxm = forced_format == "auto" and input_path.suffix.lower() == ".rbxm"
    if forced_format == "rbxm" or auto_rbxm:
        return s1.load_map_from_rbxm_binary(input_path)

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    format_name = s1.detect_input_format(raw, forced_format)
    parts = s1.load_parts(raw, format_name)
    if format_name == "snapshot":
        spawns = s1.extract_spawnpoints_from_snapshot(raw)
    elif format_name == "parts":
        spawns = s1.extract_spawnpoints_from_parts_json(raw)
    else:
        spawns = []
    return parts, s1.dedupe_points(spawns)


def convert(args: argparse.Namespace) -> Dict[str, object]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parts, spawns = load_parts_and_spawns(input_path, args.input_format)
    skip_parts = max(0, int(args.skip_parts))
    max_parts = int(args.max_parts)
    if skip_parts > 0:
        parts = parts[skip_parts:]
    if max_parts >= 0:
        parts = parts[:max_parts]

    map_name = args.map_name.strip() if args.map_name else input_path.stem

    textured_root = build_vmap(
        parts,
        spawns,
        textured=True,
        material_mode=str(args.material_mode),
        map_name=map_name,
        scale=float(args.scale),
        uv_studs_per_tile=float(args.uv_studs_per_tile),
        offset=s1.parse_offset(args.offset),
        cylinder_sides=max(6, int(args.cylinder_sides)),
        add_skybox=bool(args.add_skybox),
        skybox_margin=float(args.skybox_margin),
        skybox_thickness=float(args.skybox_thickness),
        spawn_height_source=float(args.spawn_height_source),
        lean_mesh=bool(args.lean_mesh),
        rectangles_only=bool(args.rectangles_only),
        merge_colors_into_vertex_tint=bool(args.merge_colors_into_vertex_tint),
        add_default_lighting=bool(args.add_default_lighting),
        merge_bricks=bool(args.merge_bricks),
    )

    no_texture_root = build_vmap(
        parts,
        spawns,
        textured=False,
        material_mode=CUSTOM_MATERIAL_MODE_STOCK,
        map_name=map_name,
        scale=float(args.scale),
        uv_studs_per_tile=float(args.uv_studs_per_tile),
        offset=s1.parse_offset(args.offset),
        cylinder_sides=max(6, int(args.cylinder_sides)),
        add_skybox=bool(args.add_skybox),
        skybox_margin=float(args.skybox_margin),
        skybox_thickness=float(args.skybox_thickness),
        spawn_height_source=float(args.spawn_height_source),
        lean_mesh=bool(args.lean_mesh),
        rectangles_only=bool(args.rectangles_only),
        merge_colors_into_vertex_tint=False,
        add_default_lighting=bool(args.add_default_lighting),
        merge_bricks=bool(args.merge_bricks),
    )

    textured_path = output_dir / f"{map_name}_cs2_textured.vmap"
    no_texture_path = output_dir / f"{map_name}_cs2_notexture.vmap"

    if bool(args.binary9):
        textured_kv2 = output_dir / f"{map_name}_cs2_textured_kv2.vmap"
        no_texture_kv2 = output_dir / f"{map_name}_cs2_notexture_kv2.vmap"
    else:
        textured_kv2 = textured_path
        no_texture_kv2 = no_texture_path

    write_vmap_kv2(
        textured_root,
        textured_kv2,
        fmt_ver=int(args.vmap_format_version),
        kv2_encoding_version=int(args.kv2_encoding_version),
    )
    write_vmap_kv2(
        no_texture_root,
        no_texture_kv2,
        fmt_ver=int(args.vmap_format_version),
        kv2_encoding_version=int(args.kv2_encoding_version),
    )

    if bool(args.binary9):
        convert_kv2_to_binary9(textured_kv2, textured_path)
        convert_kv2_to_binary9(no_texture_kv2, no_texture_path)
        if not bool(args.keep_kv2):
            for p in (textured_kv2, no_texture_kv2):
                if p.exists():
                    p.unlink()

    material_pack_path = None
    if str(args.material_mode) == CUSTOM_MATERIAL_MODE_ROBLOX:
        material_pack_path = emit_roblox_custom_material_pack(
            output_dir=output_dir,
            material_names=[p.material for p in parts],
            texture_source_override=str(args.roblox_texture_source or ""),
        )

    return {
        "textured": str(textured_path),
        "notexture": str(no_texture_path),
        "parts": len(parts),
        "spawns": len(spawns),
        "material_pack": str(material_pack_path) if material_pack_path else "",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert Roblox map data into native Source2 VMAP (keyvalues2).")
    p.add_argument("--input", required=False, default="", help="Input .rbxm or JSON")
    p.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "parts", "snapshot", "rbxm"],
        help="Input schema (auto detects .rbxm by extension).",
    )
    p.add_argument("--output-dir", default="out", help="Output directory")
    p.add_argument("--map-name", default="", help="Base map name (defaults to input stem)")
    p.add_argument(
        "--material-mode",
        default=CUSTOM_MATERIAL_MODE_ROBLOX_LIBRARY,
        choices=[CUSTOM_MATERIAL_MODE_STOCK, CUSTOM_MATERIAL_MODE_ROBLOX, CUSTOM_MATERIAL_MODE_ROBLOX_LIBRARY],
        help="Textured map material source: stock CS2 materials or generated Roblox custom materials.",
    )
    p.add_argument(
        "--roblox-texture-source",
        default="",
        help="Optional path to Roblox-Materials/Modern for custom material pack generation.",
    )
    p.add_argument(
        "--generate-roblox-material-library",
        action="store_true",
        default=False,
        help="Generate the full Roblox material library pack once and exit.",
    )
    p.add_argument(
        "--library-output-dir",
        default="",
        help="Override output directory used by --generate-roblox-material-library (defaults to --output-dir).",
    )
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE, help="Stud -> Source scale")
    p.add_argument(
        "--uv-studs-per-tile",
        type=float,
        default=DEFAULT_UV_STUDS_PER_TILE,
        help="Material UV tiling density in Roblox studs per tile (lower = bigger texture on surface).",
    )
    p.add_argument("--offset", default="0,0,0", help="World offset (x,y,z in Source units)")
    p.add_argument("--cylinder-sides", type=int, default=DEFAULT_CYLINDER_SIDES, help="Cylinder side count")
    p.add_argument("--vmap-format-version", type=int, default=40, help="DMX vmap format version in header")
    p.add_argument(
        "--kv2-encoding-version",
        type=int,
        default=4,
        help="DMX keyvalues2 encoding version in header (Hammer commonly expects 4).",
    )
    p.add_argument(
        "--binary9",
        action="store_true",
        default=True,
        help="Convert output to DMX binary v9 (default on).",
    )
    p.add_argument(
        "--no-binary9",
        dest="binary9",
        action="store_false",
        help="Keep outputs as keyvalues2 text VMAP.",
    )
    p.add_argument(
        "--keep-kv2",
        action="store_true",
        default=False,
        help="When binary9 is enabled, also keep intermediate *_kv2.vmap files.",
    )

    p.add_argument("--add-skybox", action="store_true", default=True)
    p.add_argument("--no-skybox", dest="add_skybox", action="store_false")
    p.add_argument("--skybox-margin", type=float, default=1024.0)
    p.add_argument("--skybox-thickness", type=float, default=128.0)
    p.add_argument("--spawn-height-source", type=float, default=48.0)
    p.add_argument("--add-default-lighting", action="store_true", default=True, help="Add template env_sky + light_environment entities at map center.")
    p.add_argument("--no-default-lighting", dest="add_default_lighting", action="store_false", help="Do not add default env_sky/light_environment entities.")
    p.add_argument("--skip-parts", type=int, default=0, help="Skip N parts from the beginning before export.")
    p.add_argument("--max-parts", type=int, default=-1, help="Cap exported part count (-1 = all).")
    p.add_argument(
        "--rectangles-only",
        action="store_true",
        default=True,
        help="Force non-block parts to export as rectangular solids for brush-like geometry.",
    )
    p.add_argument(
        "--allow-non-rect-shapes",
        dest="rectangles_only",
        action="store_false",
        help="Allow non-rectangular shape approximation (e.g., cylinders).",
    )
    p.add_argument(
        "--merge-colors-into-vertex-tint",
        action="store_true",
        default=False,
        help="Reduce mesh count by grouping by material and encoding Roblox part colors in vertex tint.",
    )
    p.add_argument(
        "--separate-mesh-per-color",
        dest="merge_colors_into_vertex_tint",
        action="store_false",
        help="Keep separate meshes for each material+color combination.",
    )
    p.add_argument(
        "--merge-bricks",
        action="store_true",
        default=True,
        help="Merge perfectly touching axis-aligned block parts with identical material+color.",
    )
    p.add_argument(
        "--no-merge-bricks",
        dest="merge_bricks",
        action="store_false",
        help="Disable strict safe brick merging.",
    )
    p.add_argument(
        "--lean-mesh",
        action="store_true",
        default=True,
        help="Emit slimmer polygon mesh streams for better Hammer save/compile stability.",
    )
    p.add_argument(
        "--full-mesh-streams",
        dest="lean_mesh",
        action="store_false",
        help="Emit full mesh streams (texcoord1/vertex paint/tangent) for maximum compatibility.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if bool(args.generate_roblox_material_library):
        try:
            target_dir = Path(args.library_output_dir.strip()) if str(args.library_output_dir).strip() else Path(args.output_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            pack = emit_roblox_custom_material_pack(
                output_dir=target_dir,
                material_names=[],
                texture_source_override=str(args.roblox_texture_source or ""),
                generate_all=True,
            )
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        print(f"[ok] roblox_material_library={str(pack) if pack else ''}")
        return 0

    if not str(args.input or "").strip():
        print("[error] --input is required unless --generate-roblox-material-library is used", file=sys.stderr)
        return 1

    try:
        result = convert(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(
        "[ok] vmap "
        f"textured={result['textured']} "
        f"notexture={result['notexture']} "
        f"parts={result['parts']} "
        f"spawns={result['spawns']} "
        f"material_pack={result.get('material_pack', '')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
