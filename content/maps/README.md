# Drop-in arenas

Every `*.json` file in this folder is loaded as an additional arena, next to the built-in ones in
`content/maps_registry.py`. The format is the same nested dictionary, written as JSON (lists instead
of tuples). The file name is the arena key unless the file sets `"key"`.

```json
{
  "name": "MY ARENA",
  "description": "Shown in the arena selection.",
  "mirror": true,
  "environment": {"clouds": 0.4, "sun_dir": [0.4, -0.8, 0.45]},
  "center": [[0, -1, 0, 40, 1, 30, "floor", -1]],
  "half": [[-15, 0, 0, 1, 1, 1, "spawn", 0], [-8, 0, 4, 1.4, 1.2, 1.4, "crate", -1]]
}
```

Each object is `[x, y, z, scale_x, scale_y, scale_z, texture_type, spawn_team]`: `(x, z)` is the centre of
the footprint, `y` the bottom, scales are full sizes in metres. With `"mirror": true` the `half` list is
copied to the other side of the arena (rotated by 180 degrees, teams swapped). Run
`python dedicated_server.py --list-maps` to validate. Every player in a LAN match needs the same file;
the handshake compares checksums.
