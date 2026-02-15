using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Datamodel;

if (args.Length < 1)
{
    Console.Error.WriteLine("usage: vmap_inspect <input.vmap>");
    return 1;
}

using var fs = File.OpenRead(args[0]);
using var dm = Datamodel.Datamodel.Load(fs, Datamodel.Codecs.DeferredMode.Disabled);
var root = dm.Root;

Console.WriteLine($"File={args[0]}");
Console.WriteLine($"Encoding={dm.Encoding} {dm.EncodingVersion}, Format={dm.Format} {dm.FormatVersion}");
Console.WriteLine($"Root={root.Name}:{root.ClassName}");
Console.WriteLine($"Prefix keys: {string.Join(", ", dm.PrefixAttributes.Keys)}");
if (dm.PrefixAttributes.ContainsKey("map_asset_references"))
{
    var refs = dm.PrefixAttributes["map_asset_references"] as Datamodel.StringArray;
    Console.WriteLine($"map_asset_references count={refs?.Count ?? 0}");
    if (refs != null)
    {
        foreach (var r in refs) Console.WriteLine($"  ref={r}");
    }
}

var world = root.Get<Element>("world");
if (world == null)
{
    Console.WriteLine("world missing");
    return 0;
}

var children = world["children"] as Datamodel.ElementArray;
Console.WriteLine($"world child count={children?.Count ?? 0}");

var mats = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
var tints = new Dictionary<string,int>();
int meshCount = 0;
if (children != null)
{
    foreach (var obj in children)
    {
        if (obj is not Element e || e.ClassName != "CMapMesh") continue;
        meshCount++;
        var md = e.Get<Element>("meshData");
        var matArr = md?["materials"] as Datamodel.StringArray;
        if (matArr != null)
            foreach (var m in matArr) mats.Add(m);

        var tint = e.ContainsKey("tintColor") ? e["tintColor"]?.ToString() ?? "<null>" : "<missing>";
        tints[tint] = tints.TryGetValue(tint, out var n) ? n+1 : 1;
    }
}

Console.WriteLine($"meshes={meshCount}");
Console.WriteLine($"unique materials={mats.Count}");
foreach (var m in mats.OrderBy(x=>x)) Console.WriteLine($"  mat={m}");
Console.WriteLine("top tints:");
foreach (var kv in tints.OrderByDescending(kv=>kv.Value).Take(20))
    Console.WriteLine($"  {kv.Key} x{kv.Value}");

return 0;
