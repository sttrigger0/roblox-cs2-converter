# Roblox -> CS2 Map Converter

Convert Roblox maps (`.rbxm` or JSON snapshots/part lists) into Source 2 `.vmap` files for CS2 Workshop Tools.

This repository includes:
- Command-line converter (`roblox_to_vmap.py`)
- Windows UI launcher source (`ui/windows_portable/`)
- Optional Binary9 normalizer (`tools/vmap_bin9_converter/`)
- Editable Roblox material library (`materials/roblox_generated/`)

## Quick Usage (CLI)

### Linux
1. Install dependencies:
```bash
sudo apt-get install -y nodejs npm python3 python3-pip
cd rbxm_vmf_converter
npm ci
python3 -m pip install srctools pillow attrs typing_extensions useful_types
```
2. Run conversion:
```bash
python3 roblox_to_vmap.py \
  --input /path/to/map.rbxm \
  --input-format auto \
  --output-dir out \
  --map-name my_map \
  --material-mode roblox_library
```

### Windows
1. Install dependencies:
- Python 3.10+ (with `pip`)
- Node.js 20+
2. In repo folder:
```powershell
npm ci
py -3 -m pip install srctools pillow attrs typing_extensions useful_types
```
3. Run conversion:
```powershell
py -3 roblox_to_vmap.py --input C:\path\map.rbxm --input-format auto --output-dir out --map-name my_map --material-mode roblox_library
```

Output files:
- `*_cs2_textured.vmap`
- `*_cs2_notexture.vmap`

## Quick Usage (Windows UI / Portable)

The UI source is in `ui/windows_portable/`.

To build a portable app zip from Linux:
```bash
./scripts/build_windows_portable.sh
```

It generates:
- `dist/RobloxCS2MapConverter_windows_portable.zip`

Portable app behavior:
- You choose RBXM input + CS2 addon content directory
- It runs conversion with the project defaults
- It copies map files into `<addon>/maps`
- It copies local editable materials from app folder into `<addon>/materials/roblox_generated`

## Post-Conversion Workflow (Hammer)
1. Open the generated map in Hammer.
2. Add both gameplay spawns manually:
- `info_player_counterterrorist`
- `info_player_terrorist`
3. Build the map (can take a long time on large Roblox maps).
4. If textures still appear missing/wrong: open each material in Material Manager once and press `Ctrl+S` so Hammer re-registers it.

## Main CLI Parameters

### Required/Primary
- `--input`: input file path (`.rbxm` or JSON)
- `--input-format`: `auto|parts|snapshot|rbxm`
- `--output-dir`: output folder
- `--map-name`: output filename prefix
- `--material-mode`: `stock|roblox_custom|roblox_library`

### Geometry / Scale
- `--scale` (default `12.8`): stud-to-source scale
- `--uv-studs-per-tile` (default `10.0`): UV tiling density
- `--offset` (default `0,0,0`): map offset
- `--rectangles-only` / `--allow-non-rect-shapes`
- `--merge-bricks` / `--no-merge-bricks`
- `--lean-mesh` / `--full-mesh-streams`
- `--skip-parts` / `--max-parts`

### World Helpers
- `--add-skybox` / `--no-skybox`
- `--skybox-margin`
- `--skybox-thickness`
- `--spawn-height-source`
- `--add-default-lighting` / `--no-default-lighting`

### Binary / Format
- `--binary9` / `--no-binary9`
- `--keep-kv2`
- `--vmap-format-version`
- `--kv2-encoding-version`

### Materials
- `--roblox-texture-source`: optional path to `Roblox-Materials/Modern`
- `--generate-roblox-material-library`: generate full material pack and exit
- `--library-output-dir`: output override for library generation

## Binary9 Converter Notes

`roblox_to_vmap.py` tries this order:
1. External converter (`tools/vmap_bin9_converter`) via `dotnet` (preferred)
2. `srctools` binary export fallback

If you need consistent schema normalization, keep `tools/vmap_bin9_converter` build artifacts available.

## Repository Layout
- `roblox_to_vmap.py`: main CS2 converter
- `roblox_to_vmf.py`: shared parsing/math helpers
- `ui/windows_portable/`: UI launcher source
- `scripts/build_windows_portable.sh`: builds portable Windows zip
- `tools/vmap_bin9_converter/`: optional Binary9 normalization tool
- `materials/roblox_generated/`: editable base material library

## Publishing / GitHub

Initialize and push:
```bash
git init
git add .
git commit -m "Initial Roblox->CS2 converter (CLI + UI)"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

If you use releases for the portable app, upload:
- `dist/RobloxCS2MapConverter_windows_portable.zip`
