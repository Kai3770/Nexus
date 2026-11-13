# Texture overrides

`<material>.png` replaces the albedo of a material (an alpha channel, if present, is its gloss) and
`<material>_n.png` its tangent-space normal map. Material names: concrete, floor, metal, rack, crate, hazard,
grate, roof, brick, paint, glow, glass, container, tower, armor. Images are resampled to the texture size of
the current graphics preset. JPG, TGA and BMP work as well.
