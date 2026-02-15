using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Datamodel;

if (args.Length >= 2 && args[0] == "--inspect")
{
    var inspectPath = args[1];
    if (!File.Exists(inspectPath))
    {
        Console.Error.WriteLine($"input not found: {inspectPath}");
        return 2;
    }

    using var inInspect = File.OpenRead(inspectPath);
    using var inspectDm = Datamodel.Datamodel.Load(inInspect, Datamodel.Codecs.DeferredMode.Disabled);

    Datamodel.Element? FindFirstByType(string typeName)
    {
        return inspectDm.AllElements.FirstOrDefault(el => string.Equals(el.ClassName, typeName, StringComparison.Ordinal));
    }

    void DumpElement(string title, Datamodel.Element? el)
    {
        Console.WriteLine($"== {title} ==");
        if (el == null)
        {
            Console.WriteLine("(missing)");
            return;
        }
        Console.WriteLine($"name={el.Name} class={el.ClassName}");
        foreach (var kv in el)
        {
            Console.WriteLine($"{kv.Key}: {kv.Value.GetType().Name}");
        }
    }

    var mesh = FindFirstByType("CMapMesh");
    var poly = FindFirstByType("CDmePolygonMesh");
    DumpElement("CMapMesh", mesh);
    DumpElement("CDmePolygonMesh", poly);

    if (poly != null && poly.ContainsKey("faceVertexData") && poly["faceVertexData"] is Datamodel.Element faceVertexData)
    {
        if (faceVertexData.ContainsKey("streams") && faceVertexData["streams"] is Datamodel.ElementArray streams)
        {
            Console.WriteLine("== faceVertexData streams ==");
            foreach (var streamObj in streams)
            {
                if (streamObj is Datamodel.Element stream)
                {
                    var n = stream.ContainsKey("name") ? stream["name"].ToString() : "(unnamed)";
                    var san = stream.ContainsKey("standardAttributeName") ? stream["standardAttributeName"].ToString() : "";
                    Console.WriteLine($"{n} | {san}");
                }
            }
        }
    }
    return 0;
}

if (args.Length >= 2 && args[0] == "--dump-lighting")
{
    var inspectPath = args[1];
    if (!File.Exists(inspectPath))
    {
        Console.Error.WriteLine($"input not found: {inspectPath}");
        return 2;
    }

    using var inInspect = File.OpenRead(inspectPath);
    using var inspectDm = Datamodel.Datamodel.Load(inInspect, Datamodel.Codecs.DeferredMode.Disabled);

    foreach (var el in inspectDm.AllElements)
    {
        if (!string.Equals(el.ClassName, "CMapEntity", StringComparison.Ordinal))
        {
            continue;
        }
        if (!el.ContainsKey("entity_properties") || el["entity_properties"] is not Datamodel.Element props)
        {
            continue;
        }
        if (!props.ContainsKey("classname"))
        {
            continue;
        }
        var cls = props["classname"]?.ToString() ?? "";
        if (!string.Equals(cls, "env_sun", StringComparison.Ordinal) &&
            !string.Equals(cls, "env_sky", StringComparison.Ordinal) &&
            !string.Equals(cls, "light_environment", StringComparison.Ordinal))
        {
            continue;
        }

        Console.WriteLine($"== {cls} ==");
        foreach (var kv in props)
        {
            Console.WriteLine($"{kv.Key}={kv.Value}");
        }
    }
    return 0;
}

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: vmap_bin9_converter <input.vmap> <output.vmap>");
    return 1;
}

var inputPath = args[0];
var outputPath = args[1];
if (!File.Exists(inputPath))
{
    Console.Error.WriteLine($"input not found: {inputPath}");
    return 2;
}

try
{
    using var inStream = File.OpenRead(inputPath);
    using var dm = Datamodel.Datamodel.Load(inStream, Datamodel.Codecs.DeferredMode.Disabled);

    int ToInt(object attr)
    {
        switch (attr)
        {
            case int i:
                return i;
            case long l:
                return (int)Math.Clamp(l, int.MinValue, int.MaxValue);
            case ulong ul:
                return (int)Math.Min((ulong)int.MaxValue, ul);
            case uint ui:
                return (int)Math.Min((uint)int.MaxValue, ui);
            case bool b:
                return b ? 1 : 0;
            case float f:
                return (int)f;
            case double d:
                return (int)d;
            default:
                return 0;
        }
    }

    void EnsureString(Element el, string key, string value = "")
    {
        if (!el.ContainsKey(key))
        {
            el.Add(key, value);
        }
    }

    void EnsureInt(Element el, string key, int value = 0)
    {
        if (!el.ContainsKey(key))
        {
            el.Add(key, value);
        }
    }

    void EnsureBool(Element el, string key, bool value = false)
    {
        if (!el.ContainsKey(key))
        {
            el.Add(key, value);
        }
    }

    void EnsureFloat(Element el, string key, float value = 0f)
    {
        if (!el.ContainsKey(key))
        {
            el.Add(key, value);
        }
    }

    void NormalizeGeneratedSchema(Datamodel.Datamodel dm)
    {
        foreach (var el in dm.AllElements)
        {
            // Hammer expects uint64 here in template maps.
            if (el.ContainsKey("referenceID"))
            {
                var asU64 = (ulong)Math.Max(0, ToInt(el["referenceID"]));
                el["referenceID"] = asU64;
            }

            if (el.ClassName == "CMapMesh")
            {
                // Match common template field shapes/types.
                if (el.ContainsKey("disableShadows"))
                {
                    el["disableShadows"] = ToInt(el["disableShadows"]);
                }
                else
                {
                    el.Add("disableShadows", 0);
                }

                EnsureString(el, "customVisGroup", "");
                EnsureInt(el, "randomSeed", 0);
                EnsureBool(el, "emissiveLightingEnabled", false);
                EnsureFloat(el, "emissiveLightingBoost", 0f);
                EnsureBool(el, "lightingDummy", false);

                // Template uses element arrays, not strings, for these.
                if (!el.ContainsKey("physicsIncludedDetailLayers") || el["physicsIncludedDetailLayers"] is not Datamodel.ElementArray)
                {
                    el["physicsIncludedDetailLayers"] = new Datamodel.ElementArray();
                }
                if (!el.ContainsKey("physicsMissingDetailLayers") || el["physicsMissingDetailLayers"] is not Datamodel.ElementArray)
                {
                    el["physicsMissingDetailLayers"] = new Datamodel.ElementArray();
                }

                if (el.ContainsKey("useAsOccluder"))
                {
                    // Non-template key; drop to reduce save-path surprises.
                    el.Remove("useAsOccluder");
                }
            }

            if (el.ClassName == "CDmePolygonMesh")
            {
                // Template polygon mesh elements do not include mapnode transform metadata.
                foreach (var key in new[] {
                    "origin", "angles", "scales", "nodeID", "referenceID", "children",
                    "editorOnly", "force_hidden", "transformLocked", "variableTargetKeys", "variableNames"
                })
                {
                    if (el.ContainsKey(key))
                    {
                        el.Remove(key);
                    }
                }
            }
        }
    }

    NormalizeGeneratedSchema(dm);

    var materialRefs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    foreach (var el in dm.AllElements)
    {
        if (!el.ContainsKey("materials"))
        {
            continue;
        }
        if (el["materials"] is not Datamodel.StringArray materials)
        {
            continue;
        }
        foreach (var mat in materials)
        {
            if (!string.IsNullOrWhiteSpace(mat))
            {
                materialRefs.Add(mat.Trim());
            }
        }
    }

    if (materialRefs.Count > 0)
    {
        var refs = new Datamodel.StringArray(materialRefs.OrderBy(s => s));
        if (dm.PrefixAttributes.ContainsKey("map_asset_references"))
        {
            dm.PrefixAttributes["map_asset_references"] = refs;
        }
        else
        {
            dm.PrefixAttributes.Add("map_asset_references", refs);
        }
    }

    var outDir = Path.GetDirectoryName(outputPath);
    if (!string.IsNullOrEmpty(outDir))
    {
        Directory.CreateDirectory(outDir);
    }

    using var outStream = File.Create(outputPath);
    dm.Save(outStream, "binary", 9);

    Console.WriteLine($"[ok] wrote binary9 vmap: {outputPath}");
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"[error] {ex.GetType().Name}: {ex.Message}");
    return 3;
}
