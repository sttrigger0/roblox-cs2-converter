Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ErrorActionPreference = "Stop"

$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $appRoot "python\python.exe"
$converterPy = Join-Path $appRoot "tools\roblox_to_vmap.py"
$inputDir = Join-Path $appRoot "input"
$outputDir = Join-Path $appRoot "output"
$materialsDir = Join-Path $appRoot "materials"

New-Item -ItemType Directory -Force -Path $inputDir, $outputDir, $materialsDir | Out-Null

if (-not (Test-Path $pythonExe)) {
    [System.Windows.Forms.MessageBox]::Show("Embedded Python not found: $pythonExe", "Startup Error", "OK", "Error") | Out-Null
    exit 1
}
if (-not (Test-Path $converterPy)) {
    [System.Windows.Forms.MessageBox]::Show("Converter script not found: $converterPy", "Startup Error", "OK", "Error") | Out-Null
    exit 1
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "Roblox -> CS2 Converter"
$form.Size = New-Object System.Drawing.Size(900, 640)
$form.StartPosition = "CenterScreen"
$form.Font = New-Object System.Drawing.Font("Segoe UI", 9)

function Add-Label($text, $x, $y, $w=160, $h=24) {
    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Text = $text
    $lbl.Location = New-Object System.Drawing.Point($x, $y)
    $lbl.Size = New-Object System.Drawing.Size($w, $h)
    $form.Controls.Add($lbl)
    return $lbl
}
function Add-TextBox($x, $y, $w=580, $h=24) {
    $tb = New-Object System.Windows.Forms.TextBox
    $tb.Location = New-Object System.Drawing.Point($x, $y)
    $tb.Size = New-Object System.Drawing.Size($w, $h)
    $form.Controls.Add($tb)
    return $tb
}
function Add-Button($text, $x, $y, $w=100, $h=28) {
    $btn = New-Object System.Windows.Forms.Button
    $btn.Text = $text
    $btn.Location = New-Object System.Drawing.Point($x, $y)
    $btn.Size = New-Object System.Drawing.Size($w, $h)
    $form.Controls.Add($btn)
    return $btn
}

Add-Label "RBXM Input" 20 20
$tbInput = Add-TextBox 180 20 560
$tbInput.Text = Join-Path $inputDir "map.rbxm"
$btnBrowseInput = Add-Button "Browse" 750 18 110

Add-Label "CS2 Addon Content Dir" 20 58
$tbAddon = Add-TextBox 180 58 560
$tbAddon.Text = "D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\content\csgo_addons\your_addon"
$btnBrowseAddon = Add-Button "Browse" 750 56 110

Add-Label "Map Name" 20 96
$tbMap = Add-TextBox 180 96 260
$tbMap.Text = "test_map"

Add-Label "Output Folder" 460 96
$tbOut = Add-TextBox 560 96 180
$tbOut.Text = $outputDir
$btnBrowseOut = Add-Button "Browse" 750 94 110

$group = New-Object System.Windows.Forms.GroupBox
$group.Text = "Settings"
$group.Location = New-Object System.Drawing.Point(20, 136)
$group.Size = New-Object System.Drawing.Size(840, 150)
$form.Controls.Add($group)

$chkBinary9 = New-Object System.Windows.Forms.CheckBox
$chkBinary9.Text = "Binary9 output (default)"
$chkBinary9.Checked = $true
$chkBinary9.Location = New-Object System.Drawing.Point(20, 28)
$chkBinary9.Size = New-Object System.Drawing.Size(240, 24)
$group.Controls.Add($chkBinary9)

$chkSkybox = New-Object System.Windows.Forms.CheckBox
$chkSkybox.Text = "Add skybox"
$chkSkybox.Checked = $true
$chkSkybox.Location = New-Object System.Drawing.Point(280, 28)
$chkSkybox.Size = New-Object System.Drawing.Size(120, 24)
$group.Controls.Add($chkSkybox)

$chkLighting = New-Object System.Windows.Forms.CheckBox
$chkLighting.Text = "Add default env_sky + light_environment"
$chkLighting.Checked = $true
$chkLighting.Location = New-Object System.Drawing.Point(420, 28)
$chkLighting.Size = New-Object System.Drawing.Size(360, 24)
$group.Controls.Add($chkLighting)

$chkRect = New-Object System.Windows.Forms.CheckBox
$chkRect.Text = "Rectangles-only solids"
$chkRect.Checked = $true
$chkRect.Location = New-Object System.Drawing.Point(20, 60)
$chkRect.Size = New-Object System.Drawing.Size(200, 24)
$group.Controls.Add($chkRect)

$chkMerge = New-Object System.Windows.Forms.CheckBox
$chkMerge.Text = "Merge touching bricks"
$chkMerge.Checked = $true
$chkMerge.Location = New-Object System.Drawing.Point(240, 60)
$chkMerge.Size = New-Object System.Drawing.Size(180, 24)
$group.Controls.Add($chkMerge)

$chkLean = New-Object System.Windows.Forms.CheckBox
$chkLean.Text = "Lean mesh streams"
$chkLean.Checked = $true
$chkLean.Location = New-Object System.Drawing.Point(440, 60)
$chkLean.Size = New-Object System.Drawing.Size(160, 24)
$group.Controls.Add($chkLean)

$chkCopyMats = New-Object System.Windows.Forms.CheckBox
$chkCopyMats.Text = "Copy local materials folder into addon (recommended)"
$chkCopyMats.Checked = $true
$chkCopyMats.Location = New-Object System.Drawing.Point(20, 92)
$chkCopyMats.Size = New-Object System.Drawing.Size(420, 24)
$group.Controls.Add($chkCopyMats)

Add-Label "Scale" 20 302 80
$tbScale = Add-TextBox 90 302 80
$tbScale.Text = "12.8"
Add-Label "UV studs/tile" 190 302 100
$tbUv = Add-TextBox 290 302 80
$tbUv.Text = "10"

$btnConvert = Add-Button "Convert" 20 340 140 32
$btnOpenOut = Add-Button "Open Output" 170 340 140 32
$btnOpenMat = Add-Button "Open Materials" 320 340 140 32

$log = New-Object System.Windows.Forms.TextBox
$log.Multiline = $true
$log.ScrollBars = "Vertical"
$log.Location = New-Object System.Drawing.Point(20, 390)
$log.Size = New-Object System.Drawing.Size(840, 190)
$log.ReadOnly = $true
$form.Controls.Add($log)

function Write-Log([string]$msg) {
    $log.AppendText("$msg`r`n")
}

$btnBrowseInput.Add_Click({
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Filter = "Roblox files (*.rbxm;*.json)|*.rbxm;*.json|All files (*.*)|*.*"
    $dlg.InitialDirectory = $inputDir
    if ($dlg.ShowDialog() -eq "OK") {
        $tbInput.Text = $dlg.FileName
        if ([string]::IsNullOrWhiteSpace($tbMap.Text) -or $tbMap.Text -eq "test_map") {
            $tbMap.Text = [System.IO.Path]::GetFileNameWithoutExtension($dlg.FileName)
        }
    }
})

$btnBrowseAddon.Add_Click({
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    if ($dlg.ShowDialog() -eq "OK") {
        $tbAddon.Text = $dlg.SelectedPath
    }
})

$btnBrowseOut.Add_Click({
    $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
    $dlg.SelectedPath = $tbOut.Text
    if ($dlg.ShowDialog() -eq "OK") {
        $tbOut.Text = $dlg.SelectedPath
    }
})

$btnOpenOut.Add_Click({
    $p = $tbOut.Text
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force -Path $p | Out-Null }
    Start-Process explorer.exe $p
})

$btnOpenMat.Add_Click({
    if (-not (Test-Path $materialsDir)) { New-Item -ItemType Directory -Force -Path $materialsDir | Out-Null }
    Start-Process explorer.exe $materialsDir
})

$btnConvert.Add_Click({
    try {
        $input = $tbInput.Text.Trim()
        $addon = $tbAddon.Text.Trim()
        $mapName = $tbMap.Text.Trim()
        $outDir = $tbOut.Text.Trim()

        if (-not (Test-Path $input)) { throw "Input file not found: $input" }
        if ([string]::IsNullOrWhiteSpace($mapName)) { throw "Map name is required." }
        if ([string]::IsNullOrWhiteSpace($addon)) { throw "Addon content directory is required." }

        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        $addonMaps = Join-Path $addon "maps"
        $addonMaterials = Join-Path $addon "materials"
        New-Item -ItemType Directory -Force -Path $addonMaps, $addonMaterials | Out-Null

        $args = @(
            $converterPy,
            "--input", $input,
            "--input-format", "auto",
            "--output-dir", $outDir,
            "--map-name", $mapName,
            "--material-mode", "roblox_library",
            "--scale", $tbScale.Text.Trim(),
            "--uv-studs-per-tile", $tbUv.Text.Trim(),
            "--spawn-height-source", "48"
        )

        if ($chkBinary9.Checked) { $args += "--binary9" } else { $args += "--no-binary9" }
        if ($chkSkybox.Checked) { $args += "--add-skybox" } else { $args += "--no-skybox" }
        if ($chkLighting.Checked) { $args += "--add-default-lighting" } else { $args += "--no-default-lighting" }
        if ($chkRect.Checked) { $args += "--rectangles-only" } else { $args += "--allow-non-rect-shapes" }
        if ($chkMerge.Checked) { $args += "--merge-bricks" } else { $args += "--no-merge-bricks" }
        if ($chkLean.Checked) { $args += "--lean-mesh" } else { $args += "--full-mesh-streams" }

        Write-Log "Running converter..."
        Write-Log ("`"" + $pythonExe + "`" " + ($args -join ' '))

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $pythonExe
        $psi.WorkingDirectory = $appRoot
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.Arguments = (($args | ForEach-Object {
            if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
        }) -join ' ')

        $nodeDir = Join-Path $appRoot "node\node-v20.19.0-win-x64"
        if (Test-Path (Join-Path $nodeDir "node.exe")) {
            $basePath = [System.Environment]::GetEnvironmentVariable("PATH")
            $psi.EnvironmentVariables["PATH"] = "$nodeDir;$basePath"
        }

        $p = [System.Diagnostics.Process]::Start($psi)
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()

        if ($stdout) { Write-Log $stdout.TrimEnd() }
        if ($stderr) { Write-Log $stderr.TrimEnd() }

        if ($p.ExitCode -ne 0) {
            throw "Converter failed with exit code $($p.ExitCode)."
        }

        $textured = Join-Path $outDir ("{0}_cs2_textured.vmap" -f $mapName)
        $notexture = Join-Path $outDir ("{0}_cs2_notexture.vmap" -f $mapName)
        if (-not (Test-Path $textured)) { throw "Expected output missing: $textured" }

        Copy-Item -Force $textured (Join-Path $addonMaps ([System.IO.Path]::GetFileName($textured)))
        if (Test-Path $notexture) {
            Copy-Item -Force $notexture (Join-Path $addonMaps ([System.IO.Path]::GetFileName($notexture)))
        }

        if ($chkCopyMats.Checked) {
            $srcMats = Join-Path $materialsDir "roblox_generated"
            if (-not (Test-Path $srcMats)) {
                throw "Missing local material library at $srcMats"
            }
            $dstMats = Join-Path $addonMaterials "roblox_generated"
            if (Test-Path $dstMats) { Remove-Item -Recurse -Force $dstMats }
            Copy-Item -Recurse -Force $srcMats $dstMats
            Write-Log "Copied materials to addon: $dstMats"
        }

        $msg = @"
Conversion finished.

Next steps:
1) Open the generated map in Hammer.
2) Manually place BOTH spawns: info_player_counterterrorist and info_player_terrorist.
3) Build the map.
4) If textures still show wrong/missing in Hammer: open each material once in Material Manager and press Ctrl+S so Hammer re-registers them.

Warning: build time can be very long on large Roblox maps.
"@
        [System.Windows.Forms.MessageBox]::Show($msg, "Done", "OK", "Information") | Out-Null
    }
    catch {
        Write-Log ("ERROR: " + $_.Exception.Message)
        [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "Conversion Error", "OK", "Error") | Out-Null
    }
})

[void]$form.ShowDialog()
