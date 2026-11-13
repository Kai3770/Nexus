"""
NEXUS-3D-EXTREME :: engine/textures.py
======================================
Procedural surface materials and panoramic skies, generated with NumPy.

The renderer has no art pipeline to lean on, so every surface texture is
synthesised here at load time (and can be overridden by image files, see
``engine/renderer.py``).  Nothing in this module needs a window:

* :func:`material_layers` returns one tileable texture per entry of
  ``maps.TEXTURE_LAYERS``: an albedo + gloss array and a tangent-space normal +
  cavity array, ready to be uploaded as two ``sampler2DArray`` textures.
* :func:`sky_panorama` paints an equirectangular sky (atmosphere gradient, sun
  halo, a perspective-correct cloud deck) for an arena's environment block.

All noise is shaped in the frequency domain, which makes every layer tile
seamlessly without any blending tricks.
"""
from __future__ import annotations

import numpy as np

from . import maps

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
def _frequencies(size: int):
    k = np.fft.fftfreq(size) * size
    return np.meshgrid(k, k, indexing="xy")


def tile_noise(size: int, beta: float = 2.0, seed: int = 0, stretch: float = 1.0) -> np.ndarray:
    """
    Seamlessly tileable fractal noise in [0, 1] (spectral synthesis: white
    noise shaped by ``1 / f**beta``).  ``stretch`` > 1 smears the pattern
    along u, which gives brushed-metal and wood-grain streaks.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft2(rng.standard_normal((size, size)))
    kx, ky = _frequencies(size)
    radius = np.sqrt((kx * stretch) ** 2 + ky ** 2)
    radius[0, 0] = 1.0
    shaped = spectrum / radius ** (beta * 0.5)
    shaped[0, 0] = 0.0
    field = np.real(np.fft.ifft2(shaped))
    low, high = np.percentile(field, (0.5, 99.5))
    return np.clip((field - low) / max(high - low, 1.0e-9), 0.0, 1.0)


def blur(field: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic Gaussian blur (``sigma`` in texels)."""
    size = field.shape[0]
    kx, ky = _frequencies(size)
    kernel = np.exp(-2.0 * (np.pi * sigma / size) ** 2 * (kx * kx + ky * ky))
    return np.real(np.fft.ifft2(np.fft.fft2(field) * kernel))


def _uv(size: int):
    axis = (np.arange(size) + 0.5) / size
    return np.meshgrid(axis, axis, indexing="xy")          # u varies along columns, v along rows (row 0 = v 0)


def _edge(coord: np.ndarray, count: float, width: float, soft: float = 0.004) -> np.ndarray:
    """1 on the ``count`` grid lines per repeat (each ``width`` wide, in uv units), 0 elsewhere."""
    distance = np.abs((coord * count + 0.5) % 1.0 - 0.5) / count
    return np.clip((width * 0.5 + soft - distance) / soft, 0.0, 1.0)


def _cell_random(u: np.ndarray, v: np.ndarray, nu: int, nv: int, seed: int) -> np.ndarray:
    table = np.random.default_rng(seed).random((nv, nu))
    return table[np.floor(v * nv).astype(np.int64) % nv, np.floor(u * nu).astype(np.int64) % nu]


def _disc(u, v, cu, cv, radius, soft=0.003) -> np.ndarray:
    du = np.abs((u - cu + 0.5) % 1.0 - 0.5)
    dv = np.abs((v - cv + 0.5) % 1.0 - 0.5)
    return np.clip((radius + soft - np.sqrt(du * du + dv * dv)) / soft, 0.0, 1.0)


def _rgb(value, tint) -> np.ndarray:
    return np.asarray(value)[..., None] * np.asarray(tint, dtype=np.float64)[None, None, :]


def _mix(a, b, t) -> np.ndarray:
    t = np.asarray(t)
    if a.ndim == 3 and t.ndim == 2:
        t = t[..., None]
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
#  Materials: each returns (albedo rgb, height, gloss), all in [0, 1]
# --------------------------------------------------------------------------- #
def _concrete(size, seed):
    u, v = _uv(size)
    cloud, grain = tile_noise(size, 2.6, seed), tile_noise(size, 0.9, seed + 1)
    stain = np.clip((tile_noise(size, 3.2, seed + 2) - 0.58) * 4.0, 0.0, 1.0)
    joint = np.maximum(_edge(u, 2, 0.010), _edge(v, 2, 0.010))
    pores = (tile_noise(size, 0.2, seed + 3) > 0.90).astype(np.float64)
    base = 0.62 + 0.13 * (cloud - 0.5) + 0.07 * (grain - 0.5) - 0.07 * stain - 0.05 * pores
    albedo = _rgb(base * (1.0 - 0.45 * joint), (1.00, 1.00, 1.03))
    height = 0.55 + 0.25 * (cloud - 0.5) + 0.20 * (grain - 0.5) - 0.50 * joint - 0.12 * pores
    return albedo, height, 0.10 + 0.08 * cloud


def _floor(size, seed):
    u, v = _uv(size)
    grout = np.maximum(_edge(u, 4, 0.014), _edge(v, 4, 0.014))
    shade = 0.52 + 0.12 * _cell_random(u, v, 4, 4, seed)
    veins = np.clip(1.0 - np.abs(tile_noise(size, 2.2, seed + 1) - 0.5) * 14.0, 0.0, 1.0)
    polish = tile_noise(size, 1.6, seed + 2, stretch=6.0)
    base = shade + 0.05 * veins + 0.04 * (polish - 0.5)
    albedo = _mix(_rgb(base, (0.96, 0.99, 1.06)), _rgb(np.full_like(u, 0.16), (1.0, 1.0, 1.0)), grout)
    height = 0.80 - 0.70 * blur(grout, size / 256.0)
    return albedo, height, _mix(0.72 + 0.10 * polish, np.full_like(u, 0.08), grout)


def _metal(size, seed):
    u, v = _uv(size)
    streak = tile_noise(size, 1.3, seed, stretch=14.0)
    fine = tile_noise(size, 0.6, seed + 1, stretch=30.0)
    seam = np.maximum(_edge(u, 2, 0.010), _edge(v, 2, 0.010))
    rivet = np.zeros_like(u)
    for cu in (0.045, 0.455, 0.545, 0.955):
        for cv in (0.045, 0.455, 0.545, 0.955):
            rivet = np.maximum(rivet, _disc(u, v, cu, cv, 0.011))
    scuff = np.clip((tile_noise(size, 2.8, seed + 2) - 0.62) * 3.0, 0.0, 1.0)
    base = 0.60 + 0.12 * (streak - 0.5) + 0.06 * (fine - 0.5) - 0.07 * scuff + 0.14 * rivet
    albedo = _rgb(base * (1.0 - 0.60 * seam), (0.97, 1.00, 1.06))
    height = 0.60 + 0.06 * (streak - 0.5) - 0.55 * seam + 0.35 * rivet
    return albedo, height, np.clip(0.78 - 0.30 * scuff - 0.45 * seam, 0.0, 1.0)


def _rack(size, seed):
    u, v = _uv(size)
    post = _edge(u, 2, 0.050)
    shelf = _edge(v, 8, 0.012)
    slot = ((v * 48.0) % 1.0 < 0.55) & (np.abs((u * 2.0) % 1.0 - 0.55) < 0.30)
    slot = slot.astype(np.float64) * (1.0 - post) * (1.0 - shelf)
    grain = tile_noise(size, 1.0, seed, stretch=8.0)
    base = 0.24 + 0.04 * (grain - 0.5) + 0.12 * post + 0.05 * shelf - 0.13 * slot
    albedo = _rgb(base, (0.95, 1.00, 1.10))
    cell_v = (v * 8.0) % 1.0
    cell_u = (u * 2.0) % 1.0
    lamp = ((np.abs(cell_v - 0.5) < 0.16) & (np.abs(cell_u - 0.14) < 0.022)).astype(np.float64)
    status = _cell_random(u, v, 2, 8, seed + 1)
    lamp_rgb = np.where((status > 0.82)[..., None], (1.0, 0.62, 0.15), (0.20, 1.00, 0.45))
    albedo = _mix(albedo, lamp_rgb * (0.35 + 0.65 * (status > 0.25))[..., None], lamp)
    height = 0.45 + 0.40 * post + 0.20 * shelf - 0.40 * slot + 0.10 * lamp
    return albedo, height, 0.42 + 0.20 * post


def _crate(size, seed):
    u, v = _uv(size)
    border = np.clip((0.085 - np.minimum(np.minimum(u, 1.0 - u), np.minimum(v, 1.0 - v))) / 0.006, 0.0, 1.0)
    brace = np.clip((0.062 - np.abs(u - v)) / 0.006, 0.0, 1.0)
    frame = np.maximum(border, brace)
    gap = _edge(v, 5, 0.012) * (1.0 - frame)
    grain = tile_noise(size, 1.5, seed, stretch=12.0)
    knots = np.clip((tile_noise(size, 3.0, seed + 1) - 0.72) * 5.0, 0.0, 1.0)
    plank = 0.60 + 0.16 * (grain - 0.5) + 0.06 * _cell_random(u, v, 1, 5, seed + 2) - 0.10 * knots
    wood = _rgb(plank, (1.00, 0.78, 0.50))
    edge = _rgb(0.70 + 0.12 * (tile_noise(size, 1.4, seed + 3, stretch=0.1) - 0.5), (1.00, 0.84, 0.60))
    albedo = _mix(wood, edge, frame) * (1.0 - 0.60 * gap)[..., None]
    bolt = np.zeros_like(u)
    for cu in (0.043, 0.957):
        for cv in (0.043, 0.957):
            bolt = np.maximum(bolt, _disc(u, v, cu, cv, 0.016))
    albedo = _mix(albedo, _rgb(np.full_like(u, 0.34), (1.0, 1.0, 1.05)), bolt)
    height = 0.45 + 0.35 * blur(frame, size / 256.0) - 0.35 * gap + 0.05 * (grain - 0.5) + 0.15 * bolt
    return albedo, height, 0.16 + 0.30 * bolt


def _hazard(size, seed):
    u, v = _uv(size)
    stripe = np.clip((np.abs(((u + v) * 4.0) % 1.0 - 0.5) - 0.25) / 0.012 + 0.5, 0.0, 1.0)
    wear = tile_noise(size, 2.0, seed)
    chip = np.clip((tile_noise(size, 1.2, seed + 1) - 0.74) * 8.0, 0.0, 1.0)
    albedo = _mix(_rgb(np.full_like(u, 1.0), (0.10, 0.10, 0.11)), _rgb(np.full_like(u, 1.0), (0.96, 0.72, 0.08)), stripe)
    albedo = albedo * (0.84 + 0.22 * wear)[..., None]
    albedo = _mix(albedo, _rgb(np.full_like(u, 0.42), (1.0, 1.0, 1.02)), chip * 0.8)
    return albedo, 0.55 + 0.10 * (wear - 0.5) - 0.25 * chip, 0.34 - 0.2 * chip


def _grate(size, seed):
    u, v = _uv(size)
    hole_u = np.abs((u * 8.0) % 1.0 - 0.5)
    hole_v = np.abs((v * 8.0) % 1.0 - 0.5)
    hole = np.clip((0.33 - np.maximum(hole_u, hole_v)) / 0.03, 0.0, 1.0)
    streak = tile_noise(size, 1.2, seed, stretch=10.0)
    bar = 0.52 + 0.10 * (streak - 0.5)
    albedo = _mix(_rgb(bar, (0.96, 1.00, 1.07)), _rgb(np.full_like(u, 0.035), (1.0, 1.0, 1.0)), hole)
    return albedo, 0.85 - 0.80 * blur(hole, size / 200.0), _mix(np.full_like(u, 0.68), np.full_like(u, 0.02), hole)


def _roof(size, seed):
    u, v = _uv(size)
    grit = tile_noise(size, 0.35, seed)
    blotch = tile_noise(size, 2.8, seed + 1)
    speck = (tile_noise(size, 0.05, seed + 2) > 0.93).astype(np.float64)
    seam = _edge(u, 1, 0.006)
    crack = np.clip(1.0 - np.abs(tile_noise(size, 2.4, seed + 3) - 0.5) * 60.0, 0.0, 1.0) * (blotch > 0.55)
    base = 0.36 + 0.08 * (grit - 0.5) + 0.07 * (blotch - 0.5) + 0.14 * speck - 0.12 * seam - 0.12 * crack
    albedo = _rgb(base, (1.00, 1.00, 1.04))
    return albedo, 0.5 + 0.22 * (grit - 0.5) + 0.12 * speck - 0.35 * crack - 0.3 * seam, 0.08 + 0.10 * blotch


def _brick(size, seed):
    u, v = _uv(size)
    rows, per_row = 8, 4
    row = np.floor(v * rows).astype(np.int64)
    shifted = u * per_row + 0.5 * (row % 2)
    mortar_v = np.abs((v * rows + 0.5) % 1.0 - 0.5) * 2.0 / rows
    mortar_u = np.abs((shifted + 0.5) % 1.0 - 0.5) * 2.0 / per_row
    mortar = np.maximum(np.clip((0.014 - mortar_v) / 0.004, 0.0, 1.0), np.clip((0.014 - mortar_u) / 0.004, 0.0, 1.0))
    ident = np.random.default_rng(seed).random((rows, per_row))[row % rows, np.floor(shifted).astype(np.int64) % per_row]
    rough = tile_noise(size, 1.0, seed + 1)
    blotch = tile_noise(size, 2.6, seed + 2)
    tone = 0.80 + 0.34 * ident + 0.16 * (rough - 0.5) + 0.12 * (blotch - 0.5)
    brick = _rgb(tone, (0.62, 0.33, 0.25)) + _rgb(ident - 0.5, (0.05, 0.03, -0.02))
    grout = _rgb(0.64 + 0.10 * (rough - 0.5), (1.00, 0.98, 0.93))
    albedo = _mix(brick, grout, mortar)
    height = 0.70 + 0.16 * (rough - 0.5) + 0.06 * (ident - 0.5) - 0.50 * blur(mortar, size / 300.0)
    return albedo, height, 0.08 + 0.06 * rough


def _paint(size, seed):
    u, v = _uv(size)
    cloud = tile_noise(size, 2.8, seed)
    roller = tile_noise(size, 0.7, seed + 1, stretch=0.2)
    line = _edge(v, 1, 0.008)
    scratch = np.clip(1.0 - np.abs(tile_noise(size, 2.0, seed + 2, stretch=7.0) - 0.5) * 90.0, 0.0, 1.0) * 0.5
    base = 0.84 + 0.06 * (cloud - 0.5) + 0.03 * (roller - 0.5) - 0.14 * line - 0.06 * scratch
    return _rgb(base, (1.00, 1.00, 1.01)), 0.55 + 0.10 * (roller - 0.5) - 0.35 * line - 0.10 * scratch, 0.30 + 0.1 * cloud


def _glow(size, seed):
    u, v = _uv(size)
    divider = np.maximum(_edge(u, 4, 0.016), _edge(v, 4, 0.016))
    cell_u = np.abs((u * 4.0) % 1.0 - 0.5)
    cell_v = np.abs((v * 4.0) % 1.0 - 0.5)
    hot = 1.0 - 0.30 * (cell_u * cell_u + cell_v * cell_v) * 2.0
    dots = 0.5 + 0.5 * np.cos(u * TWO_PI * 32.0) * np.cos(v * TWO_PI * 32.0)
    base = (0.86 + 0.08 * dots) * hot * (1.0 - 0.45 * divider)
    return _rgb(base, (1.0, 1.0, 1.0)), 0.5 + 0.3 * divider, np.full_like(u, 0.25)


def _glass(size, seed):
    u, v = _uv(size)
    mullion = np.maximum(_edge(u, 2, 0.028), _edge(v, 2, 0.022))
    pane_v = (v * 2.0) % 1.0
    streak = tile_noise(size, 1.8, seed, stretch=0.15)
    cloud = tile_noise(size, 3.0, seed + 1)
    tint = 0.46 + 0.26 * pane_v + 0.07 * (cloud - 0.5) + 0.04 * (streak - 0.5)
    pane = _rgb(tint, (0.74, 0.94, 1.12)) + _rgb(_cell_random(u, v, 2, 2, seed + 2) - 0.5, (0.04, 0.05, 0.06))
    frame = _rgb(np.full_like(u, 0.20), (0.95, 1.00, 1.08))
    albedo = _mix(pane, frame, mullion)
    return albedo, 0.35 + 0.55 * blur(mullion, size / 300.0), _mix(np.full_like(u, 0.95), np.full_like(u, 0.45), mullion)


def _container(size, seed):
    u, v = _uv(size)
    wave = 0.5 + 0.5 * np.cos(u * TWO_PI * 12.0)
    rib = np.clip(wave * 1.8 - 0.4, 0.0, 1.0)
    rail = np.clip((0.045 - np.minimum(v, 1.0 - v)) / 0.006, 0.0, 1.0)
    rust_mask = np.clip((tile_noise(size, 2.4, seed) - 0.50 - 0.30 * v) * 5.0, 0.0, 1.0)
    grime = tile_noise(size, 1.6, seed + 1, stretch=0.12)
    scratch = np.clip(1.0 - np.abs(tile_noise(size, 2.2, seed + 2, stretch=9.0) - 0.5) * 80.0, 0.0, 1.0)
    shade = (0.74 + 0.20 * rib) * (0.92 + 0.14 * (grime - 0.5)) * (1.0 - 0.30 * rail)
    albedo = _rgb(shade, (1.0, 1.0, 1.0))
    albedo = _mix(albedo, _rgb(0.55 + 0.2 * grime, (0.80, 0.50, 0.36)), rust_mask * 0.75)
    albedo = albedo + (0.10 * scratch * (1.0 - rust_mask))[..., None]
    height = 0.20 + 0.65 * rib * (1.0 - rail) + 0.55 * rail - 0.08 * rust_mask
    return albedo, height, np.clip(0.42 - 0.34 * rust_mask + 0.10 * scratch, 0.0, 1.0)


def _tower(size, seed):
    u, v = _uv(size)
    bays, floors = 4, 4
    cell_u, cell_v = (u * bays) % 1.0, (v * floors) % 1.0
    window = (np.clip((0.34 - np.abs(cell_u - 0.5)) / 0.02, 0.0, 1.0)
              * np.clip((0.30 - np.abs(cell_v - 0.52)) / 0.02, 0.0, 1.0))
    state = _cell_random(u, v, bays, floors, seed)
    warm = (state > 0.70)[..., None]
    cool = ((state > 0.58) & (state <= 0.70))[..., None]
    pane = np.where(warm, (0.62, 0.52, 0.34), np.where(cool, (0.42, 0.54, 0.62), (0.07, 0.10, 0.15)))
    pane = pane * (0.82 + 0.18 * cell_v)[..., None]
    grain = tile_noise(size, 1.6, seed + 1)
    facade = _rgb(0.20 + 0.05 * (grain - 0.5) - 0.05 * _edge(v, floors, 0.02), (0.96, 1.00, 1.08))
    albedo = _mix(facade, pane, window)
    return albedo, 0.65 - 0.35 * window, _mix(np.full_like(u, 0.15), np.full_like(u, 0.85), window)


def _armor(size, seed):
    u, v = _uv(size)
    seam = np.maximum(_edge(u, 2, 0.020), _edge(v, 3, 0.020))
    weave = 0.5 + 0.25 * np.cos(u * TWO_PI * 40.0) + 0.25 * np.cos(v * TWO_PI * 40.0)
    scuff = np.clip((tile_noise(size, 2.2, seed) - 0.60) * 3.5, 0.0, 1.0)
    vent = (np.abs((u * 2.0) % 1.0 - 0.5) < 0.16) & (np.abs((v * 3.0) % 1.0 - 0.80) < 0.07) & ((v * 36.0) % 1.0 < 0.5)
    vent = vent.astype(np.float64)
    base = 0.80 + 0.05 * (weave - 0.5) - 0.10 * scuff - 0.35 * seam - 0.30 * vent
    albedo = _rgb(base, (1.0, 1.0, 1.0))
    return albedo, 0.60 - 0.45 * blur(seam, size / 300.0) - 0.30 * vent + 0.04 * (weave - 0.5), 0.50 - 0.25 * scuff


# name -> (generator, relief strength of the normal map)
_GENERATORS = {
    "concrete": (_concrete, 2.2), "floor": (_floor, 2.6), "metal": (_metal, 2.4), "rack": (_rack, 3.0),
    "crate": (_crate, 3.0), "hazard": (_hazard, 1.2), "grate": (_grate, 4.0), "roof": (_roof, 1.6),
    "brick": (_brick, 3.6), "paint": (_paint, 1.2), "glow": (_glow, 0.8), "glass": (_glass, 2.4),
    "container": (_container, 4.2), "tower": (_tower, 1.6), "armor": (_armor, 2.6),
}


def height_to_normal(height: np.ndarray, strength: float) -> np.ndarray:
    """Tangent-space normal map (x along +u, y along +v, z out of the surface) from a tileable height field."""
    scale = strength * height.shape[0] / 256.0
    du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5 * scale * 6.0
    dv = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5 * scale * 6.0
    normal = np.stack([-du, -dv, np.ones_like(height)], axis=-1)
    return normal / np.sqrt(np.sum(normal * normal, axis=-1, keepdims=True))


def cavity_from_height(height: np.ndarray) -> np.ndarray:
    """Cheap ambient occlusion: texels lying below their blurred surroundings are shaded."""
    size = height.shape[0]
    return np.clip(1.0 - (blur(height, size / 40.0) - height) * 2.2, 0.35, 1.0)


def material_layer(name: str, size: int, seed: int = 7):
    """One material as ``(albedo_gloss, normal_cavity)`` uint8 arrays of shape ``(size, size, 4)``."""
    generator, strength = _GENERATORS[name]
    albedo, height, gloss = generator(size, seed + 101 * sorted(_GENERATORS).index(name))
    height = np.clip(height, 0.0, 1.0)
    normal = height_to_normal(height, strength)
    colour = np.concatenate([np.clip(albedo, 0.0, 1.0), np.clip(np.asarray(gloss), 0.0, 1.0)[..., None]], axis=-1)
    relief = np.concatenate([normal * 0.5 + 0.5, cavity_from_height(height)[..., None]], axis=-1)
    return np.round(colour * 255.0).astype(np.uint8), np.round(relief * 255.0).astype(np.uint8)


def material_layers(size: int, seed: int = 7):
    """
    Every layer of ``maps.TEXTURE_LAYERS`` stacked for two texture arrays:
    ``albedo[layer, v, u] = (r, g, b, gloss)`` and ``relief[layer, v, u] =
    (nx, ny, nz, cavity)``, both uint8 of shape ``(layers, size, size, 4)``.
    """
    albedo = np.zeros((len(maps.TEXTURE_LAYERS), size, size, 4), dtype=np.uint8)
    relief = np.zeros_like(albedo)
    for index, name in enumerate(maps.TEXTURE_LAYERS):
        albedo[index], relief[index] = material_layer(name, size, seed)
    return albedo, relief


# --------------------------------------------------------------------------- #
#  Panoramic sky
# --------------------------------------------------------------------------- #
def _sample_wrapped(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear lookup into a tileable field with coordinates in repeats."""
    size = field.shape[0]
    fx, fy = x * size - 0.5, y * size - 0.5
    x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    tx, ty = (fx - x0).astype(np.float32), (fy - y0).astype(np.float32)
    x0, y0, x1, y1 = x0 % size, y0 % size, (x0 + 1) % size, (y0 + 1) % size
    top = field[y0, x0] * (1.0 - tx) + field[y0, x1] * tx
    bottom = field[y1, x0] * (1.0 - tx) + field[y1, x1] * tx
    return top * (1.0 - ty) + bottom * ty


def sky_directions(width: int, height: int):
    """Unit view directions of an equirectangular image (row 0 = straight down, column 0 = azimuth -pi)."""
    azimuth = ((np.arange(width, dtype=np.float32) + 0.5) / width - 0.5) * TWO_PI
    elevation = ((np.arange(height, dtype=np.float32) + 0.5) / height - 0.5) * np.pi
    cos_e, sin_e = np.cos(elevation)[:, None], np.sin(elevation)[:, None]
    return cos_e * np.sin(azimuth)[None, :], np.broadcast_to(sin_e, (height, width)), cos_e * np.cos(azimuth)[None, :]


def sky_panorama(width: int, environment: dict, seed: int = 0) -> np.ndarray:
    """
    Equirectangular sky ``(width / 2, width, 4)`` uint8.  RGB holds the
    atmosphere gradient, the sun's halo and the cloud deck; alpha holds how
    much of the sky dome is still visible through the clouds (the shader uses
    it to hide stars and the sun disc behind them).
    """
    height = width // 2
    dx, dy, dz = sky_directions(width, height)
    top = np.asarray(environment["sky_top"], dtype=np.float32)
    horizon = np.asarray(environment["sky_horizon"], dtype=np.float32)
    haze = np.asarray(environment["fog_color"], dtype=np.float32)
    sun_color = np.asarray(environment["sun_color"], dtype=np.float32)
    sun = -np.asarray(environment["sun_dir"], dtype=np.float32)
    sun = sun / np.sqrt(np.sum(sun * sun))

    climb = np.clip((dy + 0.04) / 0.70, 0.0, 1.0)
    climb = climb * climb * (3.0 - 2.0 * climb)
    sky = horizon[None, None, :] + (top - horizon)[None, None, :] * climb[..., None]
    below = np.clip(-dy / 0.35, 0.0, 1.0)[..., None]
    sky = sky + (haze * 0.55 - sky) * below
    facing = np.clip(dx * sun[0] + dy * sun[1] + dz * sun[2], 0.0, 1.0)
    band = np.exp(-np.abs(dy) * 7.0)
    sky = sky + sun_color[None, None, :] * (facing ** 8 * 0.22 + facing ** 64 * 0.30)[..., None]
    sky = sky + (sun_color * 0.5 + horizon * 0.5)[None, None, :] * (band * facing ** 3 * 0.25)[..., None]

    cover = float(environment.get("clouds", 0.0))
    visible = np.ones((height, width), dtype=np.float32)
    if cover > 0.01:
        base = tile_noise(512, 3.0, seed + 11).astype(np.float32)
        detail = tile_noise(512, 2.3, seed + 12).astype(np.float32)
        rows = slice(height // 2, height)                          # the cloud deck only exists above the horizon
        ux, uy, uz, toward_sun = dx[rows], dy[rows], dz[rows], facing[rows]
        lift = uy + 0.09
        px, pz = ux / lift * 0.085, uz / lift * 0.085

        def density(ox: float, oz: float) -> np.ndarray:
            field = (0.68 * _sample_wrapped(base, px + ox, pz + oz)
                     + 0.24 * _sample_wrapped(detail, px * 2.3 + 0.37 + ox * 2.3, pz * 2.3 + 0.11 + oz * 2.3)
                     + 0.08 * _sample_wrapped(detail, px * 5.1 + 0.71, pz * 5.1 + 0.53))
            return np.clip((field - (0.64 - 0.34 * cover)) / 0.28, 0.0, 1.0)

        cloud = density(0.0, 0.0)
        cloud = cloud * cloud * (3.0 - 2.0 * cloud)
        horizontal = np.sqrt(sun[0] * sun[0] + sun[2] * sun[2]) + 1.0e-6
        shade = density(sun[0] / horizontal * 0.012, sun[2] / horizontal * 0.012)
        lit = np.clip(0.66 + (cloud - shade) * 0.8 + 0.22 * (1.0 - cloud), 0.0, 1.0)
        daylight = float(np.clip(np.max(sun_color) * (0.35 + 0.65 * max(float(sun[1]), 0.0)) + 0.25 * float(np.mean(top)),
                                 0.08, 1.0))
        bright = (sun_color * 0.55 + 0.45) * daylight
        dark = (horizon * 0.55 + top * 0.45) * (0.70 + 0.30 * daylight)
        tone = dark[None, None, :] + (bright - dark)[None, None, :] * lit[..., None]
        tone = tone + sun_color[None, None, :] * (toward_sun ** 12 * 0.35 * (1.0 - cloud * 0.5))[..., None]
        fade = np.clip((uy - 0.015) / 0.20, 0.0, 1.0)
        alpha = cloud * fade * fade * (3.0 - 2.0 * fade)
        sky[rows] = sky[rows] + (tone - sky[rows]) * alpha[..., None]
        visible[rows] = 1.0 - alpha
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[..., :3] = np.round(np.clip(sky, 0.0, 1.0) * 255.0).astype(np.uint8)
    out[..., 3] = np.round(np.clip(visible, 0.0, 1.0) * 255.0).astype(np.uint8)
    return out
