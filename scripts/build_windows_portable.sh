#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="RobloxCS2MapConverter"
DIST_DIR="$ROOT_DIR/dist/$APP_NAME"
ZIP_OUT="$ROOT_DIR/dist/${APP_NAME}_windows_portable.zip"

PY_EMBED_URL="${PY_EMBED_URL:-https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip}"
NODE_URL="${NODE_URL:-https://nodejs.org/dist/v20.19.0/node-v20.19.0-win-x64.zip}"

mkdir -p "$ROOT_DIR/dist"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR" "$DIST_DIR/tools" "$DIST_DIR/input" "$DIST_DIR/output" "$DIST_DIR/materials"

cp "$ROOT_DIR/roblox_to_vmap.py" "$DIST_DIR/tools/"
cp "$ROOT_DIR/roblox_to_vmf.py" "$DIST_DIR/tools/"
cp "$ROOT_DIR/package.json" "$DIST_DIR/tools/"
cp "$ROOT_DIR/package-lock.json" "$DIST_DIR/tools/"
cp -r "$ROOT_DIR/ui/windows_portable/converter_gui.ps1" "$DIST_DIR/converter_gui.ps1"
cp -r "$ROOT_DIR/ui/windows_portable/launch_gui.bat" "$DIST_DIR/launch_gui.bat"
cp "$ROOT_DIR/ui/windows_portable/README_PORTABLE.txt" "$DIST_DIR/README.txt"

if [ -d "$ROOT_DIR/node_modules" ]; then
  cp -r "$ROOT_DIR/node_modules" "$DIST_DIR/tools/"
else
  echo "[info] node_modules missing, running npm ci..."
  (cd "$ROOT_DIR" && npm ci)
  cp -r "$ROOT_DIR/node_modules" "$DIST_DIR/tools/"
fi

cp -r "$ROOT_DIR/materials/roblox_generated" "$DIST_DIR/materials/"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
export DIST_DIR TMP_DIR ZIP_OUT

curl -fL "$PY_EMBED_URL" -o "$TMP_DIR/python_embed.zip"
python3 - << 'PY'
import os, zipfile
root = os.environ['DIST_DIR']
out = os.path.join(root, 'python')
os.makedirs(out, exist_ok=True)
with zipfile.ZipFile(os.path.join(os.environ['TMP_DIR'], 'python_embed.zip')) as zf:
    zf.extractall(out)
PY

mkdir -p "$DIST_DIR/python/Lib/site-packages"
python3 - << 'PY'
import importlib, pathlib, shutil, os
mods=['srctools','attrs','attr','typing_extensions','useful_types']
site=pathlib.Path(os.environ['DIST_DIR'])/'python'/'Lib'/'site-packages'
site.mkdir(parents=True, exist_ok=True)
for m in mods:
    mod=importlib.import_module(m)
    p=pathlib.Path(mod.__file__).resolve()
    if p.name=='__init__.py':
        dst=site/p.parent.name
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(p.parent,dst)
    else:
        shutil.copy2(p, site/p.name)
for base in [pathlib.Path('/usr/local/lib/python3.10/dist-packages'), pathlib.Path('/usr/lib/python3/dist-packages')]:
    if not base.exists():
        continue
    for pattern in ['srctools-*.dist-info','attrs-*.dist-info','typing_extensions-*.dist-info','useful_types-*.dist-info']:
        for di in base.glob(pattern):
            dst=site/di.name
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(di,dst)
PY

cat > "$DIST_DIR/python/python311._pth" << 'PTH'
python311.zip
.
Lib
Lib/site-packages
..\tools
import site
PTH

curl -fL "$NODE_URL" -o "$TMP_DIR/node.zip"
python3 - << 'PY'
import os, zipfile, shutil
root=os.environ['DIST_DIR']
tmp=os.environ['TMP_DIR']
out=os.path.join(root,'node')
os.makedirs(out, exist_ok=True)
extract=os.path.join(tmp,'node_extract')
os.makedirs(extract, exist_ok=True)
with zipfile.ZipFile(os.path.join(tmp,'node.zip')) as zf:
    zf.extractall(extract)
entries=[e for e in os.listdir(extract) if e.startswith('node-v')]
if not entries:
    raise SystemExit('node zip missing node-v* dir')
shutil.move(os.path.join(extract, entries[0]), os.path.join(out, entries[0]))
PY

rm -f "$ZIP_OUT"
python3 - << 'PY'
import os, zipfile
root=os.environ['DIST_DIR']
zip_out=os.environ['ZIP_OUT']
base=os.path.dirname(root)
with zipfile.ZipFile(zip_out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
    for d, _, files in os.walk(root):
        for f in files:
            full=os.path.join(d,f)
            rel=os.path.relpath(full, base)
            zf.write(full, rel)
print(zip_out)
PY

echo "[ok] built $ZIP_OUT"
