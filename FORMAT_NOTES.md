# RBXM / RBXMX / VMF Syntax Notes

This project converts Roblox-exported structure data to Valve VMF.

## 1) Roblox format reality check

- Roblox Creator docs identify model/place extensions:
  - `.rbxm`, `.rbxl` = binary
  - `.rbxmx`, `.rbxlx` = XML
- Roblox does **not** publish a full official binary grammar for `.rbxm` in Creator docs.
- The practical, widely used reference for exact syntax is the reverse-engineered `rbx-dom` docs and implementations.

## 2) RBXMX (XML) syntax (exact structure)

From `rbx-dom` XML docs, the root structure is:

- Root element: `<roblox version="4"> ... </roblox>`
- Instances are nested as `<Item class="..." referent="...">`
- Properties live under `<Properties>` using typed tags like:
  - `<string name="Name">...</string>`
  - `<bool name="Anchored">true</bool>`
  - `<Vector3 name="size"><X>..</X><Y>..</Y><Z>..</Z></Vector3>`
  - `<CoordinateFrame name="CFrame">` with `X,Y,Z,R00..R22`
  - `<Ref name="SomeProperty">RBX...</Ref>` for instance references

This is the reliable syntax target if we add native `.rbxmx` input parsing.

## 3) RBXM (binary) syntax (what is known)

Community-documented binary layout (reverse engineered, used by tools) is chunk-based.
Common chunks include:

- `INST` (instance/class records)
- `PROP` (property payload)
- `PRNT` (parent relationships)
- `META` (metadata)
- `SSTR` (shared strings)
- `SIGN` (signature, when present)
- `END\0` terminator

So, exact `.rbxm` parsing means implementing chunk decoding, type decoding, and referent resolution. This converter currently expects already-decoded JSON (`parts` or snapshot schema), not raw `.rbxm` bytes.

## 4) VMF syntax used by this project

VMF is a text KeyValue format with nested blocks. Canonical shape:

```text
versioninfo
{
    "editorversion" "400"
    ...
}
world
{
    "id" "1"
    "classname" "worldspawn"
    solid
    {
        "id" "2"
        side
        {
            "id" "3"
            "plane" "(x y z) (x y z) (x y z)"
            "material" "TOOLS/TOOLSNODRAW"
            "uaxis" "[1 0 0 0] 0.25"
            "vaxis" "[0 -1 0 0] 0.25"
            "rotation" "0"
            "lightmapscale" "16"
            "smoothing_groups" "0"
        }
    }
}
```

The converter writes exactly this `world -> solid -> side` structure and formats `plane` as 3 points, with `uaxis/vaxis` entries per side.

## 5) Source links

- Roblox docs (extensions overview):
  - https://create.roblox.com/docs/reference/engine/enums/ModelFormat
- `rbx-dom` format docs:
  - https://dom.rojo.space/xml.html
  - https://github.com/rojo-rbx/rbx-dom/blob/master/docs/rbxlx.md
  - https://github.com/rojo-rbx/rbx-dom/blob/master/docs/rbxl.md
- RBXM binary chunk discussion/reference:
  - https://rojoblox.com/2021/04/20/rusting-roblox/
- VMF reference pages:
  - https://developer.valvesoftware.com/wiki/VMF_%28Valve_Map_Format%29
  - https://developer.valvesoftware.com/wiki/VMFs
- Additional VMF syntax breakdown:
  - https://wiki.alliedmods.net/VMF_%28Valve_Map_Format%29
