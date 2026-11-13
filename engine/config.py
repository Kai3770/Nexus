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
PROTOCOL_VERSION = 9
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

BUTTON_NAMES = {
    BTN_FORWARD: "forward", BTN_LEFT: "left", BTN_BACK: "back", BTN_RIGHT: "right",
    BTN_SLIDE: "slide", BTN_JUMP: "jump", BTN_SHOOT: "shoot", BTN_ABILITY: "ability",
    BTN_SPRINT: "sprint", BTN_RELOAD: "reload", BTN_ADS: "ads", BTN_SWAP: "swap", BTN_FIRED: "fired",
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
    ("capacitor", "PULSE CAPACITOR", 10, 0.05, 650.0, 1.55, "-5% pulse grenade cooldown per level"),
    ("bounty", "BOUNTY UPLINK", 20, 0.10, 300.0, 1.42, "+10% combat credits per level"),
    ("nanites", "MEDIC NANITES", 8, 0.06, 450.0, 1.65, "-6% health regeneration delay per level"),
)

# Ability: pulse grenade -------------------------------------------------------
ABILITY_COOLDOWN = 12.0
GRENADE_FUSE = 1.6
GRENADE_SPEED = 19.0
GRENADE_RADIUS = 6.5
GRENADE_DAMAGE = 120.0                   # at the centre; falls off with (1 - d / radius)^2
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
