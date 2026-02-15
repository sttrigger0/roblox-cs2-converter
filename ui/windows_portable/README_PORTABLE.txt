Roblox -> CS2 Converter (Portable)

How to use:
1) Double-click launch_gui.bat
2) Pick your .rbxm input
3) Pick your CS2 addon CONTENT folder (example: ...\content\csgo_addons\myaddon)
4) Click Convert

App folders (next to launch_gui.bat):
- input: put RBXM files here (optional)
- output: generated VMAP files
- materials\roblox_generated: editable material library copied into addon on each conversion

Important:
- After conversion, open the map in Hammer.
- Add BOTH player spawn entities manually:
  - info_player_counterterrorist
  - info_player_terrorist
- Then build the map.
- Build time may be very long for large Roblox maps.
