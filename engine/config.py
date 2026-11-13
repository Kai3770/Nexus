"""
NEXUS-3D-EXTREME :: engine/config.py
====================================
Single source of truth for every tuning value: simulation clock, netcode,
movement, the weapon balance matrices, audio gain staging, graphics quality
presets and frame pacing.

Coordinate convention (Ursina 8, Panda3D in ``y-up-left``):

    +X  right        +Y  up        +Z  forward

Angles are radians inside the simulation.  ``yaw`` turns clockwise seen from
above (0 looks down +Z), ``pitch`` is positive looking up:

    forward = ( sin(yaw) cos(pitch),  sin(pitch),  cos(yaw) cos(pitch) )
    right   = ( cos(yaw),             0,          -sin(yaw)            )

Frame-rate independence
-----------------------
Gameplay is simulated on a fixed 60 Hz tick (``TICK_DT``) because client
prediction, reconciliation and server rollback need bit-identical steps on
both ends.  The renderer is decoupled from it: it runs at any frame rate and
blends the two newest ticks, while every purely visual system (camera feel,
particles, tracers, HUD animation) integrates the measured frame delta time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
#  Identity and filesystem
# --------------------------------------------------------------------------- #
GAME_TITLE = "NEXUS-3D-EXTREME"
GAME_VERSION = "1.1.0"

BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = Path(os.environ.get("NEXUS_SAVE_DIR", str(BASE_DIR / "saves")))
CACHE_DIR = Path(os.environ.get("NEXUS_CACHE_DIR", str(BASE_DIR / "cache")))
SFX_DIR = CACHE_DIR / "sfx"
BAKE_DIR = CACHE_DIR / "bake"
CONTENT_DIR = BASE_DIR / "content"
MAPS_DIR = CONTENT_DIR / "maps"          # drop-in *.json arenas
MODELS_DIR = CONTENT_DIR / "models"      # drop-in *.obj props

# --------------------------------------------------------------------------- #
#  Simulation clock (60 Hz tick steps)
# --------------------------------------------------------------------------- #
TICK_RATE = 60
TICK_DT = 1.0 / TICK_RATE
TICK_MS = 1000.0 / TICK_RATE
MAX_TICKS_PER_FRAME = 6                  # spiral-of-death guard when a frame takes > 100 ms
MAX_FRAME_DT = 0.1                       # longest frame delta any visual integrator accepts
MAX_PLAYERS = 16

# --------------------------------------------------------------------------- #
#  Frame pacing
# --------------------------------------------------------------------------- #
VSYNC_DEFAULT = False                    # double-buffer vsync halves 59 fps to 30; a limiter does not
FPS_LIMIT_DEFAULT = 144                  # 0 = uncapped
TARGET_FPS = 60.0
DYNAMIC_RESOLUTION_DEFAULT = True        # trade render resolution for frame time on slow GPUs
DYNRES_STEPS = (0.50, 0.60, 0.70, 0.85, 1.00)
DYNRES_DROP_MS = 1000.0 / TARGET_FPS * 1.03     # smoothed frame time above this -> one step down
DYNRES_RAISE_MS = 1000.0 / TARGET_FPS * 0.72    # below this -> one step up
DYNRES_COOLDOWN = 1.5                    # seconds between resolution changes
QUALITY_FALLBACK_SECONDS = 4.0           # lowest render scale still too slow for this long -> next lower preset
THREAD_SWITCH_INTERVAL = 0.001           # s; interpreter hand-over between the render loop and a listen server

# --------------------------------------------------------------------------- #
#  Graphics quality presets
# --------------------------------------------------------------------------- #
QUALITY_LOW, QUALITY_MEDIUM, QUALITY_HIGH, QUALITY_ULTRA = range(4)
QUALITY_NAMES = ("LOW", "MEDIUM", "HIGH", "ULTRA")
QUALITY_DEFAULT = QUALITY_HIGH
GRAPHICS_PRESETS = (
    # render_scale: ceiling of the 3D resolution scale      shadow_size: sun shadow map edge (0 = off)
    # shadow_taps: PCF kernel (1, 4 or 9)                   texture_size: material texture resolution
    # light_cell: baked-light tessellation in metres        sky_size: panoramic sky texture width
    # normal_maps: per-pixel relief                         bloom: glow pass       particles: emission scale
    {"render_scale": 0.70, "shadow_size": 0, "shadow_taps": 1, "anisotropy": 1, "texture_size": 128,
     "light_cell": 3.0, "sky_size": 1024, "particles": 0.5, "normal_maps": False, "bloom": False},
    {"render_scale": 0.85, "shadow_size": 1024, "shadow_taps": 1, "anisotropy": 4, "texture_size": 256,
     "light_cell": 2.0, "sky_size": 2048, "particles": 0.75, "normal_maps": True, "bloom": False},
    {"render_scale": 1.00, "shadow_size": 2048, "shadow_taps": 4, "anisotropy": 8, "texture_size": 256,
     "light_cell": 1.25, "sky_size": 2048, "particles": 1.0, "normal_maps": True, "bloom": True},
    {"render_scale": 1.00, "shadow_size": 4096, "shadow_taps": 9, "anisotropy": 16, "texture_size": 512,
     "light_cell": 0.8, "sky_size": 4096, "particles": 1.0, "normal_maps": True, "bloom": True},
)
MAX_DYNAMIC_LIGHTS = 8                   # muzzle flashes / explosions evaluated per pixel
MAX_STATIC_LIGHTS = 12                   # level mood lights, baked into vertices at load
MUZZLE_FLASH_TIME = 0.03                 # seconds of muzzle point-light life
SHADOW_DEPTH_BIAS = 0.05                 # metres of depth bias along the sun direction
SHADOW_NORMAL_OFFSET = 1.6               # shadow lookups are pushed off the surface by this many shadow texels
SHADOW_DYNAMIC_SIZE = 1024               # edge of the per-frame shadow map that holds the characters
BLOOM_STRENGTH = 0.55
SKY_DOME_RADIUS = 900.0
TEXTURES_DIR = CONTENT_DIR / "textures"  # drop-in <material>.png / <material>_n.png overrides
SKY_DIR = CONTENT_DIR / "sky"            # drop-in <arena key>.png / .jpg equirectangular panoramas
TEXTURE_WORLD_SCALE = 1.0                # multiplies every material's tiling size

# --------------------------------------------------------------------------- #
#  Networking
# --------------------------------------------------------------------------- #
DISCOVERY_PORT = 28015
GAME_PORT = 28016
GAME_PORT_SEARCH = 16
BROADCAST_ADDRESS = "255.255.255.255"
BEACON_INTERVAL = 1.0
BEACON_EXPIRY = 3.5
PROTOCOL_MAGIC = 0x584E
PROTOCOL_VERSION = 11
MAX_DATAGRAM = 1200
RECV_BUFFER = 4096
CONNECT_RETRY_INTERVAL = 0.25
CONNECT_TIMEOUT = 5.0
CONNECTION_TIMEOUT = 6.0
KEEPALIVE_INTERVAL = 1.0                 # s; sent by the receive thread while the main thread is busy loading

HISTORY_TICKS = 30                       # lag compensation ring: 30 ticks = 500 ms
HISTORY_MS = HISTORY_TICKS * TICK_MS
INTERP_DELAY_MS = 100.0
INTERP_BUFFER = 64
RECONCILE_THRESHOLD = 0.05
RECONCILE_VELOCITY_THRESHOLD = 0.75
SMOOTH_CORRECTION_LIMIT = 2.5
SNAPSHOT_INTERVAL_TICKS = 1
SNAPSHOT_RING = 128
CMD_REDUNDANCY = 3
CMD_QUEUE_LIMIT = 24
CMD_BUDGET_MAX = 8.0
MAX_CMDS_PER_TICK = 4
PENDING_CMD_CAPACITY = 256
EVENT_BYTES_PER_SNAPSHOT = 520
ROSTER_INTERVAL = 1.0

POS_QUANT = 128.0
VEL_QUANT = 256.0
ANGLE_QUANT = 65536.0 / (2.0 * np.pi)
WORLD_LIMIT = 250.0

# --------------------------------------------------------------------------- #
#  Input bit mask (bits 0..7 in wire order W, A, S, D, Slide, Jump, Shoot, Ability)
# --------------------------------------------------------------------------- #
BTN_FORWARD = 1 << 0
BTN_LEFT = 1 << 1
BTN_BACK = 1 << 2
BTN_RIGHT = 1 << 3
BTN_SLIDE = 1 << 4
BTN_JUMP = 1 << 5
BTN_SHOOT = 1 << 6
BTN_ABILITY = 1 << 7
BTN_SPRINT = 1 << 8
BTN_RELOAD = 1 << 9
BTN_ADS = 1 << 10
BTN_SWAP = 1 << 11
BTN_FIRED = 1 << 12
BTN_TACTICAL = 1 << 13                    # use the tactical equipment (stim, shock, smoke)

BUTTON_NAMES = {
    BTN_FORWARD: "forward", BTN_LEFT: "left", BTN_BACK: "back", BTN_RIGHT: "right",
    BTN_SLIDE: "slide", BTN_JUMP: "jump", BTN_SHOOT: "shoot", BTN_ABILITY: "ability",
    BTN_SPRINT: "sprint", BTN_RELOAD: "reload", BTN_ADS: "ads", BTN_SWAP: "swap", BTN_FIRED: "fired",
    BTN_TACTICAL: "tactical",
}

# Movement state flags ---------------------------------------------------------
MF_ON_GROUND = 1 << 0
MF_CROUCHED = 1 << 1
MF_SLIDING = 1 << 2
MF_SPRINTING = 1 << 3
MF_ADS = 1 << 4
MF_JUMPED = 1 << 5
MF_LANDED = 1 << 6
MF_STEPPED = 1 << 7
MF_PERSISTENT = MF_ON_GROUND | MF_CROUCHED | MF_SLIDING | MF_SPRINTING | MF_ADS

# --------------------------------------------------------------------------- #
#  Kinematics (per second units; the simulation multiplies by TICK_DT)
# --------------------------------------------------------------------------- #
GRAVITY = 24.0
TERMINAL_VELOCITY = 55.0
MAX_HORIZONTAL_SPEED = 40.0

PLAYER_RADIUS = 0.35
PLAYER_HEIGHT_STAND = 1.80
PLAYER_HEIGHT_CROUCH = 1.10
EYE_HEIGHT_STAND = 1.62
CROUCH_CAMERA_DROP = 0.40
EYE_HEIGHT_CROUCH = EYE_HEIGHT_STAND * (1.0 - CROUCH_CAMERA_DROP)
CROUCH_TRANSITION_TIME = 0.22

WALK_SPEED = 5.6
SPRINT_SPEED = 8.2
CROUCH_SPEED = 2.9
ADS_SPEED_SCALE = 0.62
GROUND_ACCEL = 12.0
GROUND_FRICTION = 8.0
STOP_SPEED = 2.0
AIR_ACCEL = 2.4
JUMP_SPEED = 7.4
COYOTE_TIME = 0.09
JUMP_BUFFER_TIME = 0.10
STEP_HEIGHT = 0.42

SLIDE_MIN_SPEED = 6.4
SLIDE_START_SPEED = 11.2
SLIDE_IMPULSE = 2.6
SLIDE_SPEED_CAP = 13.5
SLIDE_FRICTION_K = 1.45                  # V_next = V * exp(-k * dt)
SLIDE_END_SPEED = 4.4
SLIDE_MAX_TIME = 1.10
SLIDE_MIN_TIME = 0.22
SLIDE_COOLDOWN = 0.45

COLLISION_EPS = 1.0e-5
CANDIDATE_MARGIN = 1.0
KILL_PLANE_Y = -30.0

# --------------------------------------------------------------------------- #
#  Camera feel (integrated with the measured frame delta time)
# --------------------------------------------------------------------------- #
BASE_FOV = 90.0
SPRINT_FOV_BONUS = 5.0
SLIDE_FOV_BONUS = 9.0
VIEWMODEL_FOV = 62.0
BANK_STRAFE_DEG = 2.4
BANK_SLIDE_DEG = 6.5
BOB_FREQUENCY = 1.55
BOB_AMPLITUDE = 0.045
SWAY_AMPLITUDE = 0.012
LAND_DIP = 0.16
MOUSE_SENSITIVITY_DEFAULT = 1.0
PITCH_LIMIT = np.radians(88.0)

# --------------------------------------------------------------------------- #
#  Vital ceilings
# --------------------------------------------------------------------------- #
BASE_HEALTH = 100.0                      # every operator has exactly 100 health; no upgrade raises it
HEALTH_REGEN_DELAY = 5.0
HEALTH_REGEN_RATE = 28.0
RESPAWN_TIME = 3.0
SPAWN_PROTECTION = 1.5
ZONE_HEAD = 0
ZONE_TORSO = 1
ZONE_LIMB = 2
ZONE_MULTIPLIER = np.array([3.0, 1.0, 0.75], dtype=np.float64)
ZONE_NAMES = ("head", "torso", "limb")
MAX_BULLET_RANGE = 220.0

# --------------------------------------------------------------------------- #
#  Weapon balance matrices
# --------------------------------------------------------------------------- #
# Balancing model (enforced by ``engine.weapons.validate_balance`` and the test-suite):
#
#   * Time-to-kill against the 100 health ceiling, torso hits, every round landing, inside the
#     effective range:  TTK_MIN_MS <= TTK <= TTK_MAX_MS  for every weapon, stock AND fully upgraded.
#   * Sorting the arsenal by damage per trigger pull, the fire interval, the recoil and the reload
#     time all grow monotonically: more damage per shot always costs rate of fire, control and
#     reload speed.
#   * Recoil is not hand tuned; it is RECOIL_PITCH_PER_DAMAGE * damage per trigger pull.
TTK_MIN_MS = 300.0
TTK_MAX_MS = 500.0
RECOIL_PITCH_PER_DAMAGE = np.radians(0.025)     # radians of climb per point of damage per trigger pull
RECOIL_YAW_RATIO = 0.45                         # horizontal scatter relative to the climb

WEAPON_CLASSES = ("ASSAULT RIFLE", "SMG", "SHOTGUN", "MARKSMAN", "PISTOL", "LMG", "SNIPER")
WEAPON_COLUMNS = (
    "key", "name", "klass", "auto", "damage", "pellets", "interval", "mag", "reload_s", "swap",
    "hip_deg", "ads_deg", "bloom_shot_deg", "bloom_max_deg", "bloom_recover_deg",
    "range_start", "range_end", "range_min", "recoil_recover", "mobility", "ads_zoom",
    "unlock_cost", "unlock_level", "tier", "description",
)
# damage: per pellet      interval: ticks between rounds      swap: ticks to raise the weapon
# hip / ads: cone half-angles in degrees      range_min: damage fraction beyond range_end
WEAPON_TABLE = (
    ("vanguard", "VK-9 VANGUARD", 0, True, 21.0, 1, 6.0, 30, 1.9, 18,
     2.0, 0.22, 0.22, 1.8, 9.0, 28.0, 60.0, 0.62, 7.0, 1.00, 1.35, 0.0, 1, 1.0,
     "Balanced full-auto rifle. Five rounds to the chest at any practical range."),
    ("hornet", "HX-4 HORNET", 1, True, 17.0, 1, 4.0, 30, 1.5, 14,
     2.4, 0.55, 0.20, 2.2, 11.0, 12.0, 30.0, 0.55, 9.0, 1.06, 1.20, 0.0, 1, 1.0,
     "Fastest time-to-kill up close, fades quickly with distance."),
    ("breacher", "BREACHER-12", 2, False, 8.5, 8, 26.0, 6, 2.9, 20,
     3.6, 3.0, 0.0, 0.0, 0.0, 8.0, 18.0, 0.30, 5.0, 0.98, 1.15, 2500.0, 3, 1.6,
     "Semi-automatic shotgun. Two full patterns at arm's length."),
    ("longbow", "LONGBOW DMR", 3, False, 40.0, 1, 12.0, 12, 2.7, 20,
     3.2, 0.08, 0.80, 2.4, 6.0, 50.0, 100.0, 0.75, 5.5, 0.96, 2.20, 6000.0, 6, 2.4,
     "Marksman rifle. Three body hits or one clean headshot."),
    ("sidekick", "P9 SIDEKICK", 4, False, 20.0, 1, 5.0, 12, 1.6, 10,
     1.6, 0.30, 0.50, 2.2, 10.0, 14.0, 34.0, 0.55, 9.0, 1.08, 1.15, 0.0, 1, 0.8,
     "Sidearm with a fast draw. As quick as your trigger finger."),
    ("bulwark", "BULWARK LMG", 5, True, 28.0, 1, 8.0, 60, 2.6, 26,
     3.4, 0.50, 0.20, 2.8, 7.0, 32.0, 70.0, 0.65, 5.0, 0.88, 1.35, 15000.0, 10, 3.6,
     "Sixty round belt and four-hit kills, paid for with mobility and control."),
    ("aurora", "AURORA RAIL", 6, False, 80.0, 1, 28.0, 5, 3.1, 28,
     7.0, 0.0, 0.0, 0.0, 0.0, 110.0, 200.0, 0.85, 3.5, 0.92, 4.00, 40000.0, 15, 5.5,
     "Rail rifle. Two to the body, one to the head, at any distance."),
    ("viper", "VX-3 VIPER", 4, True, 14.4, 1, 3.3, 24, 1.3, 9,
     2.2, 0.70, 0.30, 2.8, 12.0, 9.0, 22.0, 0.55, 10.0, 1.08, 1.12, 4000.0, 4, 1.4,
     "Full-auto machine pistol. Seven hits to the body, but they arrive fast."),
    ("wraith", "WR-5 WRAITH PDW", 1, True, 20.5, 1, 5.5, 40, 1.75, 15,
     2.3, 0.40, 0.18, 2.0, 10.0, 16.0, 38.0, 0.58, 8.5, 1.04, 1.25, 5500.0, 5, 1.8,
     "Forty-round PDW. Steadier and longer reaching than the Hornet, a little slower to kill."),
    ("verdict", "VERDICT REVOLVER", 4, False, 34.0, 1, 10.0, 6, 2.65, 12,
     2.0, 0.15, 0.90, 2.6, 7.0, 22.0, 45.0, 0.70, 5.0, 1.05, 1.30, 7000.0, 7, 2.0,
     "Six-shot hand cannon. Three to the body or one to the head."),
    ("kestrel", "KR-11 KESTREL", 0, True, 25.5, 1, 7.0, 20, 2.2, 20,
     2.4, 0.18, 0.28, 2.0, 8.0, 34.0, 75.0, 0.68, 6.0, 0.97, 1.50, 9000.0, 8, 2.6,
     "Battle rifle. Four hits at almost any range; twenty rounds to make them count."),
    ("cyclone", "CYCLONE AUTO-8", 2, True, 7.0, 8, 20.0, 8, 2.8, 22,
     4.2, 3.4, 0.0, 0.0, 0.0, 6.0, 14.0, 0.25, 4.5, 0.95, 1.10, 11000.0, 9, 2.8,
     "Drum-fed automatic shotgun. Hold the trigger: two patterns end it up close."),
    ("midge", "MX-2 MIDGE", 1, True, 14.3, 1, 3.2, 32, 1.25, 8,
     2.8, 0.75, 0.24, 2.6, 13.0, 8.0, 18.0, 0.50, 10.5, 1.10, 1.12, 1200.0, 2, 0.9,
     "Micro SMG. The fastest cyclic rate in the armory and the shortest reach."),
    ("shrike", "TK-7 SHRIKE", 1, True, 16.8, 1, 3.9, 30, 1.4, 13,
     2.5, 0.50, 0.20, 2.2, 11.0, 11.0, 26.0, 0.55, 9.5, 1.07, 1.20, 2800.0, 3, 1.5,
     "Compact SMG with a stabilised grip. Hornet pace with a tighter cone."),
    ("warden", "WD-17 WARDEN", 4, False, 17.3, 1, 4.4, 17, 1.55, 10,
     1.5, 0.28, 0.40, 2.0, 11.0, 14.0, 32.0, 0.55, 9.5, 1.08, 1.15, 1600.0, 2, 0.9,
     "Seventeen-round service pistol. Six quick taps from a forgiving cone."),
    ("axiom", "AX-5 AXIOM", 0, True, 20.2, 1, 5.2, 30, 1.65, 16,
     2.1, 0.30, 0.20, 1.9, 10.0, 20.0, 45.0, 0.60, 8.0, 1.03, 1.30, 4500.0, 4, 1.6,
     "Carbine. Handles quicker than a full rifle and gives up a little reach."),
    ("sparrow", "BP-4 SPARROW", 0, True, 20.8, 1, 5.8, 30, 1.8, 18,
     2.2, 0.24, 0.21, 1.8, 9.5, 26.0, 55.0, 0.62, 7.5, 1.01, 1.40, 5800.0, 5, 1.9,
     "Bullpup rifle. A long barrel in a short body; accurate on the move."),
    ("tempest", "TR-6 TEMPEST", 0, True, 21.4, 1, 6.5, 45, 2.0, 20,
     2.4, 0.26, 0.24, 2.2, 8.5, 30.0, 62.0, 0.63, 7.0, 0.98, 1.35, 7500.0, 7, 2.2,
     "Forty-five round rifle. Keeps firing while others reload."),
    ("heron", "LM-3 HERON", 5, True, 21.7, 1, 6.8, 100, 2.1, 24,
     3.2, 0.45, 0.18, 2.6, 7.5, 30.0, 65.0, 0.64, 5.5, 0.90, 1.30, 12000.0, 9, 2.9,
     "Light machine gun with a hundred-round box. Suppression first, speed second."),
    ("basalt", "BR-14 BASALT", 0, True, 25.0, 1, 6.9, 25, 2.15, 21,
     2.4, 0.20, 0.27, 2.0, 8.0, 36.0, 80.0, 0.68, 6.0, 0.97, 1.55, 18000.0, 11, 3.4,
     "Heavy battle rifle. Four hits out to eighty metres."),
    ("sentry", "DM-7 SENTRY", 3, False, 26.5, 1, 7.5, 20, 2.35, 20,
     3.0, 0.10, 0.55, 2.2, 7.0, 45.0, 90.0, 0.72, 6.0, 0.97, 1.80, 16000.0, 10, 3.2,
     "Semi-automatic tactical rifle. Four fast, precise taps at range."),
    ("mammoth", "MG-6 MAMMOTH", 5, True, 27.5, 1, 7.8, 80, 2.5, 28,
     3.6, 0.55, 0.22, 3.0, 6.5, 34.0, 72.0, 0.66, 4.8, 0.86, 1.30, 28000.0, 14, 4.0,
     "Drum-fed heavy machine gun. Brutal, slow to raise and slower to move."),
    ("anvil", "HP-50 ANVIL", 4, False, 28.5, 1, 9.0, 7, 2.62, 13,
     2.0, 0.20, 0.85, 2.6, 7.5, 20.0, 42.0, 0.68, 5.5, 1.05, 1.25, 6500.0, 6, 2.1,
     "Heavy semi-automatic pistol. Four body hits; the recoil earns its name."),
    ("ranger", "LV-36 RANGER", 3, False, 36.0, 1, 11.0, 8, 2.67, 18,
     3.0, 0.10, 0.70, 2.4, 6.5, 45.0, 95.0, 0.74, 5.5, 0.98, 1.90, 21000.0, 12, 3.6,
     "Lever-action rifle. Three body hits or one headshot, cycled fast."),
    ("osprey", "SC-43 OSPREY", 6, False, 43.0, 1, 13.5, 6, 2.75, 22,
     5.0, 0.03, 0.0, 0.0, 0.0, 70.0, 150.0, 0.80, 4.5, 0.97, 3.00, 44000.0, 16, 5.6,
     "Light bolt-action sniper. Three to the body or one to the head, and still mobile."),
    ("badger", "SL-1 BADGER", 2, False, 52.0, 1, 19.5, 5, 2.78, 22,
     3.0, 0.40, 0.0, 0.0, 0.0, 18.0, 40.0, 0.55, 5.0, 0.96, 1.25, 9500.0, 8, 2.5,
     "Slug gun. One heavy round instead of a pattern: two hits at mid range."),
    ("gavel", "PX-10 GAVEL", 2, False, 6.0, 10, 22.0, 7, 2.85, 22,
     4.0, 3.2, 0.0, 0.0, 0.0, 7.0, 16.0, 0.28, 5.0, 0.97, 1.15, 24000.0, 13, 3.8,
     "Pump shotgun with a ten-pellet load. Two patterns at close range."),
    ("twinfang", "DB-2 TWINFANG", 2, False, 6.0, 12, 27.0, 2, 3.0, 18,
     4.6, 3.8, 0.0, 0.0, 0.0, 6.0, 13.0, 0.25, 4.5, 1.00, 1.10, 50000.0, 18, 5.8,
     "Double barrel. Both shells kill at arm's length; then you reload."),
    ("halberd", "HB-84 HALBERD", 6, False, 84.0, 1, 29.0, 4, 3.4, 32,
     7.5, 0.0, 0.0, 0.0, 0.0, 130.0, 250.0, 0.88, 3.2, 0.88, 4.50, 60000.0, 20, 6.2,
     "Anti-materiel rifle. Two hits from anywhere on the map, if you can carry it."),
)
DEFAULT_LOADOUT_KEYS = ("vanguard", "sidekick")

# Bullet Spread Matrix: multiplier on the base cone, rows = weapon class,
# columns = movement state (standing, walking, sprinting, crouched, sliding, airborne).
SPREAD_MATRIX = (
    (1.00, 1.30, 1.80, 0.72, 1.75, 2.40),   # assault rifle
    (1.00, 1.15, 1.40, 0.80, 1.30, 1.80),   # smg
    (1.00, 1.05, 1.20, 0.90, 1.10, 1.30),   # shotgun
    (1.00, 1.60, 2.40, 0.60, 2.20, 3.20),   # marksman
    (1.00, 1.20, 1.50, 0.78, 1.35, 1.90),   # pistol
    (1.00, 1.50, 2.20, 0.62, 2.10, 3.00),   # lmg
    (1.00, 1.80, 2.80, 0.55, 2.60, 3.60),   # sniper
)

# Per-weapon upgrade tracks.  Damage and fire-rate ceilings are sized so that a fully upgraded
# weapon keeps its shots-to-kill and stays inside the TTK band.
UPGRADE_COLUMNS = ("key", "name", "max_level", "effect", "base_cost", "cost_multiplier", "description")
UPGRADE_TABLE = (
    ("damage", "DAMAGE", 10, 0.015, 150.0, 1.55, "+1.5% bullet damage per level"),
    ("fire_rate", "FIRE RATE", 10, 0.010, 220.0, 1.60, "-1% time between shots per level"),
    ("magazine", "MAGAZINE", 10, 0.050, 180.0, 1.45, "+5% magazine capacity per level"),
    ("reload", "RELOAD SPEED", 10, 0.030, 160.0, 1.40, "-3% reload time per level"),
    ("stabilizer", "STABILIZER", 5, 0.090, 400.0, 1.85, "Attachment: -9% recoil per level"),
    ("laser", "LASER MODULE", 5, 0.080, 350.0, 1.80, "Attachment: -8% hip-fire spread per level"),
    ("barrel", "LONG BARREL", 5, 0.100, 450.0, 1.90, "Attachment: +10% damage range per level"),
    ("stock", "FEATHER STOCK", 5, 0.015, 300.0, 1.75, "Attachment: +1.5% movement speed per level"),
)
# Operator upgrades never touch the health ceiling or bullet damage.
OPERATOR_TABLE = (
    ("plating", "BLAST PLATING", 5, 0.06, 500.0, 1.60, "-6% explosive damage taken per level"),
    ("capacitor", "PULSE CAPACITOR", 10, 0.05, 650.0, 1.55, "-5% equipment cooldown per level"),
    ("bounty", "BOUNTY UPLINK", 20, 0.10, 300.0, 1.42, "+10% combat credits per level"),
    ("nanites", "MEDIC NANITES", 8, 0.06, 450.0, 1.65, "-6% health regeneration delay per level"),
)

# Equipment: one LETHAL (key G) and one TACTICAL (key F) per player ---------------------------------
# Chosen in the EQUIPMENT menu; a change made during a match takes effect at the next respawn.
EQUIP_LETHAL, EQUIP_TACTICAL = 0, 1
EQUIP_SLOT_NAMES = ("LETHAL", "TACTICAL")
EQUIPMENT_KINDS = ("frag", "sticky", "fire", "axe", "stun", "smoke", "stim")
EQUIPMENT_COLUMNS = ("key", "name", "slot", "kind", "cooldown", "speed", "gravity", "fuse", "radius", "damage",
                     "duration", "strength", "unlock_cost", "unlock_level", "description")
# cooldown s | speed m/s, gravity scale (thrown items) | fuse s: frag / stun / smoke detonate after it; a sticky
# charge after it sticks; a fire flask bursts on impact or after it; an axe falls out of the air after it
# radius m | damage: at the centre (frag, sticky), per second (fire), on a direct hit (axe), healed (stim)
# duration s: fire / smoke lifetime, longest stun, stim healing time | strength: movement factor while stunned
EQUIPMENT_TABLE = (
    ("pulse", "PULSE GRENADE", 0, "frag", 12.0, 19.0, 1.0, 1.6, 6.5, 120.0, 0.0, 0.0, 0.0, 1,
     "Bouncing fragmentation grenade on a 1.6 second fuse."),
    ("hatchet", "RAZOR HATCHET", 0, "axe", 9.0, 26.0, 0.55, 2.0, 0.0, 150.0, 0.0, 0.0, 2000.0, 3,
     "Thrown axe. A direct hit is lethal, a miss is just a miss."),
    ("cling", "CLING CHARGE", 0, "sticky", 16.0, 17.0, 1.0, 1.2, 4.5, 140.0, 0.0, 0.0, 3000.0, 5,
     "Sticks to walls and to people, then detonates. Whoever wears it does not walk away."),
    ("thermite", "THERMITE FLASK", 0, "fire", 15.0, 16.0, 1.0, 2.5, 3.2, 38.0, 4.0, 0.0, 4500.0, 8,
     "Shatters on impact into a fire pool: 38 damage per second for four seconds."),
    ("stim", "NANO STIM", 1, "stim", 16.0, 0.0, 0.0, 0.0, 0.0, 50.0, 1.0, 0.0, 0.0, 1,
     "Injects 50 health over one second and restarts regeneration at once."),
    ("shock", "SHOCK GRENADE", 1, "stun", 14.0, 18.0, 1.0, 1.0, 7.0, 0.0, 2.5, 0.55, 1800.0, 2,
     "Stuns everyone it reaches: slower movement and a shaky aim for up to 2.5 seconds."),
    ("smoke", "SMOKE CANISTER", 1, "smoke", 18.0, 17.0, 1.0, 1.2, 5.0, 0.0, 8.0, 0.0, 2600.0, 4,
     "Fills a five metre sphere with smoke for eight seconds; nobody sees through it."),
)
DEFAULT_EQUIPMENT_KEYS = ("pulse", "stim")
EQUIP_KILL_BASE = 200                    # kill-feed weapon id of an equipment kill: base + equipment index
STICK_DAMAGE_MULT = 1.5                  # a charge stuck to a player hits that player this much harder
STICK_FLIGHT = 1.8                       # extra seconds a sticky charge may fly before it detonates anyway
STUN_OWNER_SCALE = 0.5                   # your own shock grenade stuns you for half as long
STUN_EDGE_SCALE = 0.4                    # stun duration at the edge of the blast, relative to the centre
STUN_LOOK_SCALE = 0.35                   # mouse sensitivity while stunned (client side)
FIRE_HEIGHT = 1.9                        # a fire pool burns players whose feet are this close above it
ZONE_CAPACITY = 16
STATUS_STUN, STATUS_STIM = 1, 2

# Derived names kept for the pulse grenade (the default lethal).
ABILITY_COOLDOWN = EQUIPMENT_TABLE[0][4]
GRENADE_SPEED = EQUIPMENT_TABLE[0][5]
GRENADE_FUSE = EQUIPMENT_TABLE[0][7]
GRENADE_RADIUS = EQUIPMENT_TABLE[0][8]
GRENADE_DAMAGE = EQUIPMENT_TABLE[0][9]   # at the centre; falls off with (1 - d / radius)^2
GRENADE_RESTITUTION = 0.42
GRENADE_CAPACITY = 32
EXPLOSION_KNOCKBACK = 11.0

# --------------------------------------------------------------------------- #
#  Audio gain staging
# --------------------------------------------------------------------------- #
# final gain = clamp(master, 0, 1) * ceiling[category] * clamp(cue volume, 0, 1) * distance falloff
# Distance falloff is the inverse-square law around a per-category reference distance:
#     falloff = 1 / max(distance / reference, 1) ** 2
AUDIO_MASTER_DEFAULT = 0.5
AUDIO_CEILING = {
    "weapon": 0.25,       # gunfire
    "impact": 0.20,       # hits taken, bullet impacts
    "explosion": 0.30,    # loudest cue in the game
    "handling": 0.16,     # reload, swap, dry fire
    "movement": 0.14,     # steps, jump, land, slide, throw
    "feedback": 0.22,     # hit / headshot / kill confirmation
    "interface": 0.20,    # menu clicks
    "announcer": 0.24,    # level up, round result, respawn
}
AUDIO_REFERENCE_DISTANCE = {"weapon": 10.0, "impact": 5.0, "explosion": 14.0, "handling": 3.0,
                            "movement": 4.0, "feedback": 4.0, "interface": 1.0, "announcer": 1.0}
AUDIO_CATEGORY = {
    "shot": "weapon", "explosion": "explosion", "hurt": "impact", "reload": "handling", "empty": "handling",
    "swap": "handling", "jump": "movement", "land": "movement", "slide": "movement", "step": "movement",
    "throw": "movement", "hit": "feedback", "headshot": "feedback", "kill": "feedback",
    "ui_click": "interface", "ui_buy": "interface", "ui_error": "interface",
    "respawn": "announcer", "level_up": "announcer", "victory": "announcer", "defeat": "announcer",
    "hatchet": "movement", "hatchet_hit": "impact", "stick": "feedback", "fire": "explosion", "shock": "explosion",
    "smoke": "explosion", "stim": "feedback", "stun_ring": "feedback",
}
AUDIO_MIN_GAIN = 0.003                   # quieter cues are not started at all
AUDIO_DUCK_WINDOW = 0.07                 # seconds; overlapping cues of one category are attenuated
AUDIO_DUCK_FACTOR = 0.7
AUDIO_PAN_STRENGTH = 0.8

# --------------------------------------------------------------------------- #
#  Match rules
# --------------------------------------------------------------------------- #
MODE_TDM = 0
MODE_FFA = 1
MODE_NAMES = ("TEAM DEATHMATCH", "FREE FOR ALL")
PHASE_WARMUP = 0
PHASE_LIVE = 1
PHASE_ENDED = 2
TDM_SCORE_LIMIT = 75
FFA_SCORE_LIMIT = 30
MATCH_TIME_LIMIT = 600.0
INTERMISSION_TIME = 12.0
TEAM_NAMES = ("VANTA", "HELIX")
TEAM_COLORS = np.array([[0.95, 0.36, 0.20], [0.20, 0.62, 0.98]], dtype=np.float32)

LAYER_PLAYER = 1
LAYER_BULLET = 2
LAYER_ALL = LAYER_PLAYER | LAYER_BULLET


def ensure_directories() -> None:
    """Create the writable directories used by the game (idempotent)."""
    for folder in (SAVE_DIR, CACHE_DIR, SFX_DIR, BAKE_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def is_frozen() -> bool:
    """True when running from a packaged executable."""
    return bool(getattr(sys, "frozen", False))
