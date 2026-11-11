"""
NEXUS-3D-EXTREME :: engine/config.py
====================================
Single source of truth for every tuning constant shared by the authoritative
server, the predicting client, the renderer and the test-suite.

Coordinate convention (identical to Ursina 8.x, which runs Panda3D in the
``y-up-left`` coordinate system):

    +X  right        +Y  up        +Z  forward

Angles are stored in radians inside the simulation.  ``yaw`` rotates clockwise
when seen from above (yaw = 0 looks down +Z, yaw = +90deg looks down +X) and
``pitch`` is positive when looking up.  The matching unit vectors are

    forward = ( sin(yaw) * cos(pitch),  sin(pitch),  cos(yaw) * cos(pitch) )
    right   = ( cos(yaw),               0,          -sin(yaw)              )

Stock Panda3D is Z-up; Ursina remaps the world so that "up" is +Y.  Every place
in this code base that says "world up" therefore means the +Y axis, which is the
axis stock Panda3D documentation calls the global Z axis.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
#  Identity
# --------------------------------------------------------------------------- #
GAME_TITLE = "NEXUS-3D-EXTREME"
GAME_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
#  Filesystem layout
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = Path(os.environ.get("NEXUS_SAVE_DIR", str(BASE_DIR / "saves")))
CACHE_DIR = Path(os.environ.get("NEXUS_CACHE_DIR", str(BASE_DIR / "cache")))
SFX_DIR = CACHE_DIR / "sfx"

# --------------------------------------------------------------------------- #
#  Simulation clock
# --------------------------------------------------------------------------- #
TICK_RATE = 60                       # authoritative simulation frequency (Hz)
TICK_DT = 1.0 / TICK_RATE            # seconds per tick
TICK_MS = 1000.0 / TICK_RATE         # milliseconds per tick
MAX_TICKS_PER_FRAME = 6              # client spiral-of-death guard
MAX_PLAYERS = 16                     # entity capacity of every SoA store

# --------------------------------------------------------------------------- #
#  Networking
# --------------------------------------------------------------------------- #
DISCOVERY_PORT = 28015               # LAN beacon port (UDP broadcast)
GAME_PORT = 28016                    # default gameplay port (UDP unicast)
GAME_PORT_SEARCH = 16                # how many consecutive ports to try when hosting
BROADCAST_ADDRESS = "255.255.255.255"
BEACON_INTERVAL = 1.0                # seconds between server beacons
BEACON_EXPIRY = 3.5                  # seconds until a silent server leaves the lobby
PROTOCOL_MAGIC = 0x584E              # 'NX' little endian
PROTOCOL_VERSION = 7
MAX_DATAGRAM = 1200                  # stay below the common 1500 byte MTU
RECV_BUFFER = 4096
CONNECT_RETRY_INTERVAL = 0.25
CONNECT_TIMEOUT = 5.0
CONNECTION_TIMEOUT = 6.0             # silence before a peer is dropped

HISTORY_TICKS = 30                   # lag-compensation ring size (30 ticks == 500 ms @ 60 Hz)
HISTORY_MS = HISTORY_TICKS * TICK_MS
INTERP_DELAY_MS = 100.0              # fixed remote-entity interpolation delay
INTERP_BUFFER = 64                   # snapshots kept by the client interpolation ring
RECONCILE_THRESHOLD = 0.05           # world units of divergence that force a rewind + replay
RECONCILE_VELOCITY_THRESHOLD = 0.75  # units / second
SMOOTH_CORRECTION_LIMIT = 2.5        # larger errors are snapped visually as well
SNAPSHOT_INTERVAL_TICKS = 1          # 60 Hz state stream (LAN)
SNAPSHOT_RING = 128                  # per-tick quantised world states kept for delta baselines
CMD_REDUNDANCY = 3                   # previous commands re-sent in each client datagram
CMD_QUEUE_LIMIT = 24                 # server side per-client command backlog cap
CMD_BUDGET_MAX = 8.0                 # speed-hack token bucket (commands)
MAX_CMDS_PER_TICK = 4                # server catch-up steps for one client per tick
PENDING_CMD_CAPACITY = 256           # client ring of unacknowledged commands
EVENT_BYTES_PER_SNAPSHOT = 520       # reliable event budget per datagram
ROSTER_INTERVAL = 1.0                # seconds between roster / scoreboard datagrams

# Wire quantisation ------------------------------------------------------------
POS_QUANT = 128.0                    # int16 position units per world unit (+-255.99)
VEL_QUANT = 256.0                    # int16 velocity units per world unit / second
ANGLE_QUANT = 65536.0 / (2.0 * np.pi)
WORLD_LIMIT = 250.0                  # maps must stay inside +-WORLD_LIMIT on every axis

# --------------------------------------------------------------------------- #
#  Input bit-mask (bits 0..7 are exactly the wire order W, A, S, D, Slide,
#  Jump, Shoot, Ability; the upper byte carries the extended inputs)
# --------------------------------------------------------------------------- #
BTN_FORWARD = 1 << 0
BTN_LEFT = 1 << 1
BTN_BACK = 1 << 2
BTN_RIGHT = 1 << 3
BTN_SLIDE = 1 << 4      # crouch / slide key
BTN_JUMP = 1 << 5
BTN_SHOOT = 1 << 6
BTN_ABILITY = 1 << 7
BTN_SPRINT = 1 << 8
BTN_RELOAD = 1 << 9
BTN_ADS = 1 << 10
BTN_SWAP = 1 << 11
BTN_FIRED = 1 << 12     # set by the owner on the command whose tick released a round

BUTTON_NAMES = {
    BTN_FORWARD: "forward", BTN_LEFT: "left", BTN_BACK: "back", BTN_RIGHT: "right",
    BTN_SLIDE: "slide", BTN_JUMP: "jump", BTN_SHOOT: "shoot", BTN_ABILITY: "ability",
    BTN_SPRINT: "sprint", BTN_RELOAD: "reload", BTN_ADS: "ads", BTN_SWAP: "swap",
    BTN_FIRED: "fired",
}

# --------------------------------------------------------------------------- #
#  Movement state flags (uint16 per entity)
# --------------------------------------------------------------------------- #
MF_ON_GROUND = 1 << 0
MF_CROUCHED = 1 << 1
MF_SLIDING = 1 << 2
MF_SPRINTING = 1 << 3
MF_ADS = 1 << 4
MF_JUMPED = 1 << 5      # transient: the jump impulse was applied this tick
MF_LANDED = 1 << 6      # transient: touched the ground this tick
MF_STEPPED = 1 << 7     # transient: a stair step was climbed this tick
MF_PERSISTENT = MF_ON_GROUND | MF_CROUCHED | MF_SLIDING | MF_SPRINTING | MF_ADS

# --------------------------------------------------------------------------- #
#  Kinematics
# --------------------------------------------------------------------------- #
GRAVITY = 24.0
TERMINAL_VELOCITY = 55.0
MAX_HORIZONTAL_SPEED = 40.0          # hard safety clamp (explosive knock-back etc.)

PLAYER_RADIUS = 0.35                 # half width of the square collision footprint
PLAYER_HEIGHT_STAND = 1.80
PLAYER_HEIGHT_CROUCH = 1.10
EYE_HEIGHT_STAND = 1.62
CROUCH_CAMERA_DROP = 0.40            # the camera sinks by 40 % while crouched / sliding
EYE_HEIGHT_CROUCH = EYE_HEIGHT_STAND * (1.0 - CROUCH_CAMERA_DROP)
CROUCH_TRANSITION_TIME = 0.22        # seconds for the cosine eased camera height blend

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

SLIDE_MIN_SPEED = 6.4                # horizontal speed threshold that arms the slide
SLIDE_START_SPEED = 11.2             # minimum launch speed of a slide
SLIDE_IMPULSE = 2.6                  # speed added on top of the entry speed
SLIDE_SPEED_CAP = 13.5
SLIDE_FRICTION_K = 1.45              # V_next = V * exp(-k * dt)
SLIDE_END_SPEED = 4.4
SLIDE_MAX_TIME = 1.10
SLIDE_MIN_TIME = 0.22
SLIDE_COOLDOWN = 0.45

COLLISION_EPS = 1.0e-5
CANDIDATE_MARGIN = 1.0               # broad-phase cache inflation (world units)
KILL_PLANE_Y = -30.0

# --------------------------------------------------------------------------- #
#  Camera feel (client only)
# --------------------------------------------------------------------------- #
BASE_FOV = 90.0
SPRINT_FOV_BONUS = 5.0
SLIDE_FOV_BONUS = 9.0
VIEWMODEL_FOV = 62.0
BANK_STRAFE_DEG = 2.4
BANK_SLIDE_DEG = 6.5
BOB_FREQUENCY = 1.55                 # stride cycles per world unit travelled (radians scale)
BOB_AMPLITUDE = 0.045
SWAY_AMPLITUDE = 0.012
LAND_DIP = 0.16
MOUSE_SENSITIVITY_DEFAULT = 1.0
PITCH_LIMIT = np.radians(88.0)

# --------------------------------------------------------------------------- #
#  Combat
# --------------------------------------------------------------------------- #
BASE_HEALTH = 100.0
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

ABILITY_COOLDOWN = 12.0
GRENADE_FUSE = 1.6
GRENADE_SPEED = 19.0
GRENADE_RADIUS = 6.5
GRENADE_DAMAGE = 150.0
GRENADE_RESTITUTION = 0.42
GRENADE_CAPACITY = 32
EXPLOSION_KNOCKBACK = 11.0

MUZZLE_FLASH_TIME = 0.03             # seconds of muzzle point-light life
MAX_DYNAMIC_LIGHTS = 8

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

# Collision / ray layers -------------------------------------------------------
LAYER_PLAYER = 1                     # blocks kinematic bodies
LAYER_BULLET = 2                     # blocks hitscan and grenades
LAYER_ALL = LAYER_PLAYER | LAYER_BULLET


def ensure_directories() -> None:
    """Create the writable directories used by the game (idempotent)."""
    for folder in (SAVE_DIR, CACHE_DIR, SFX_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def is_frozen() -> bool:
    """True when running from a packaged executable."""
    return bool(getattr(sys, "frozen", False))
