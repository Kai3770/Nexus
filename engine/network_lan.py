"""
NEXUS-3D-EXTREME :: engine/network_lan.py              (MODULE A)
=================================================================
60 Hz rollback netcode and zero-configuration LAN discovery.

Subsystems
----------
1. **LAN discovery** -- ``BeaconBroadcaster`` (server) sends an encrypted
   beacon to 255.255.255.255:28015 once per second; ``LanBrowser`` (client)
   listens on that port in a background thread and keeps a live server list.
   Nobody types an IP address.
2. **Binary protocol** -- every datagram is hand packed with ``struct``.
   Input commands are delta compressed against the previous command, world
   snapshots are delta compressed against the last snapshot the client
   acknowledged (per-entity change masks + packed NumPy record arrays).
3. **Lag compensation** -- ``LagCompensator`` keeps a ring of the last 30
   ticks (500 ms) of every player's root transform.  A FIRE record carries
   the client-side execution timestamp; the server rewinds all hitboxes to
   that instant, ray casts, registers the hit and restores the present.
4. **Prediction / reconciliation** -- ``Predictor`` simulates the local
   player immediately, keeps every unacknowledged command, and when the
   authoritative state diverges by more than 0.05 units snaps to it and
   replays the pending commands inside the same frame.
5. **Entity interpolation** -- ``InterpolationBuffer`` renders remote players
   100 ms in the past, LERPing positions and SLERPing orientation quaternions
   between the two snapshots that bracket the render time.

The gameplay socket uses UDP port 28016 (next free port if busy) so that it
never competes with the discovery listeners bound to 28015.
"""
from __future__ import annotations

import heapq
import random
import socket
import struct
import threading
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np

from . import bots as B
from . import config as C
from . import crypto
from . import equipment as EQ
from . import maps
from . import mathx as M
from . import simulation as S
from . import weapons as W
from .simulation import PlayerStore, simulate_commands

# =========================================================================== #
#  Wire formats
# =========================================================================== #
HEADER = struct.Struct("<HBI")                      # magic, packet type, session token
PKT_CONNECT_REQ, PKT_CONNECT_ACK, PKT_CONNECT_DENY, PKT_DISCONNECT, PKT_CLIENT_FRAME, PKT_SNAPSHOT, PKT_ROSTER = range(1, 8)
DENY_FULL, DENY_VERSION, DENY_BAD_REQUEST = 1, 2, 3
DENY_REASONS = {DENY_FULL: "server is full", DENY_VERSION: "protocol version mismatch", DENY_BAD_REQUEST: "bad request"}
CONTENT_MISMATCH = "the server runs an arena that is not installed here (or a different version of it)"

BEACON_MAGIC = b"NXB1"
BEACON_KEY = b"NX3D-lan-beacon-key-9d41f0c2b7"
BEACON_STRUCT = struct.Struct("<HHBBBBBBIIB")      # protocol, port, humans, max, bots, map, mode, tick rate, uid, map crc, name length

CONNECT_REQ = struct.Struct("<HIBB")               # protocol, nonce, preferred team, name length (+ name + loadout)
CONNECT_ACK = struct.Struct("<IBIBBBBIBBHHI")      # nonce, id, tick, rate, map, mode, team, token, max, upgrades, score, time, map crc
CONNECT_DENY = struct.Struct("<IB")
FRAME_HEADER = struct.Struct("<IIIIHBBBB")         # ack snapshot, ack event, client ms, newest cmd, ping, n cmds, n fire,
#                                                    requested lethal, requested tactical
EQUIP_BLOB = struct.Struct("<BB")                  # lethal, tactical (appended to the connect request)
FIRE_RECORD = struct.Struct("<HIIB")               # shot seq, command tick, client view time (ms), weapon slot
SNAP_HEADER = struct.Struct("<IIIHIHBBB")          # tick, baseline, ack cmd, ack shot, echo ms, hold ms, flags, n rel, n transient
MATCH_STATE = struct.Struct("<HHHBB")              # score 0, score 1, seconds left, phase, winner
ENT_MASKS = struct.Struct("<HHHH")                 # active, transform changed, orientation changed, state changed
ROSTER_ENTRY = struct.Struct("<BBBHHIHB")          # id, team, flags, kills, deaths, score, ping, name length
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_F32 = struct.Struct("<f")
_VEL = struct.Struct("<hhh")

CMD_BUTTONS, CMD_YAW, CMD_PITCH, CMD_VEL = 1, 2, 4, 8
SNAP_HAS_OWNER = 1

ENT_T = np.dtype([("px", "<i2"), ("py", "<i2"), ("pz", "<i2"), ("vx", "<i2"), ("vy", "<i2"), ("vz", "<i2")])
ENT_O = np.dtype([("yaw", "<u2"), ("pitch", "<i2")])
ENT_S = np.dtype([("flags", "u1"), ("health", "u1"), ("weapon", "u1"), ("crouch", "u1"), ("seq", "u1"), ("team", "u1")])

EF_ALIVE, EF_GROUND, EF_CROUCH, EF_SLIDE, EF_SPRINT, EF_ADS, EF_RELOAD, EF_BOT = (1 << i for i in range(8))

_EVENT_FORMATS = {
    S.EV_HIT: struct.Struct("<BHBBhhh"),            # victim, damage*10, zone, killed, position
    S.EV_DAMAGE: struct.Struct("<BHBhhh"),          # attacker, damage*10, zone, source position
    S.EV_KILL: struct.Struct("<BBBB"),              # killer, victim, weapon, flags
    S.EV_SPAWN: struct.Struct("<BhhhH"),            # player, position, yaw
    S.EV_SHOT: struct.Struct("<BBhhhhhh"),          # shooter, weapon, origin, end
    S.EV_EXPLOSION: struct.Struct("<hhhBBB"),       # position, owner, grenade id, equipment
    S.EV_GRENADE: struct.Struct("<BBhhhhhhHIB"),    # id, owner, position, velocity, fuse, spawn tick, equipment
    S.EV_STICK: struct.Struct("<BBhhh"),            # grenade id, target (255: a wall), position
    S.EV_STATUS: struct.Struct("<BH"),              # status, ticks
    S.EV_REWARD: struct.Struct("<IHB"),             # credits, xp, reason
    S.EV_MATCH: struct.Struct("<BBBBI"),            # phase, winner, map, mode, map crc
}

assert C.MAX_PLAYERS <= 16, "entity masks are 16 bit"


def make_packet(packet_type: int, token: int, payload: bytes = b"") -> bytes:
    return HEADER.pack(C.PROTOCOL_MAGIC, packet_type, token & 0xFFFFFFFF) + payload


def now_ms() -> int:
    """Local monotonic clock in whole milliseconds, wrapped to 32 bits."""
    return int(time.perf_counter() * 1000.0) & 0xFFFFFFFF


def seq_newer(a: int, b: int) -> bool:
    """True when 16-bit sequence number ``a`` is ahead of ``b`` (wrap aware)."""
    return a != b and ((a - b) & 0xFFFF) < 0x8000


# ---- quantisation helpers ---------------------------------------------------
def quantize_position(position) -> tuple:
    q = np.clip(np.rint(np.asarray(position, dtype=np.float64) * C.POS_QUANT), -32767, 32767).astype(np.int64)
    return int(q[0]), int(q[1]), int(q[2])


def quantize_velocity(velocity) -> tuple:
    q = np.clip(np.rint(np.asarray(velocity, dtype=np.float64) * C.VEL_QUANT), -32767, 32767).astype(np.int64)
    return int(q[0]), int(q[1]), int(q[2])


def _unq(q, scale) -> tuple:
    return (q[0] / scale, q[1] / scale, q[2] / scale)


def mask_to_int(mask: np.ndarray) -> int:
    return int.from_bytes(np.packbits(mask.astype(np.uint8), bitorder="little").tobytes(), "little") & 0xFFFF


def int_to_rows(value: int, capacity: int) -> np.ndarray:
    return np.flatnonzero((value >> np.arange(capacity)) & 1)


# =========================================================================== #
#  Command codec (delta compressed against the previous command in the packet)
# =========================================================================== #
def encode_commands(commands) -> bytes:
    """
    ``commands`` is an iterable of ``(buttons, yaw, pitch, (vx, vy, vz))`` with
    quantised velocity, oldest first.  Each record starts with a flag byte that
    says which fields differ from the previous record; only those are written.
    """
    out = bytearray()
    previous = None
    for buttons, yaw, pitch, velocity in commands:
        if previous is None:
            flags = CMD_BUTTONS | CMD_YAW | CMD_PITCH | CMD_VEL
        else:
            flags = ((CMD_BUTTONS if buttons != previous[0] else 0) | (CMD_YAW if yaw != previous[1] else 0)
                     | (CMD_PITCH if pitch != previous[2] else 0) | (CMD_VEL if velocity != previous[3] else 0))
        out.append(flags)
        if flags & CMD_BUTTONS:
            out += _U16.pack(buttons)
        if flags & CMD_YAW:
            out += _F32.pack(yaw)
        if flags & CMD_PITCH:
            out += _F32.pack(pitch)
        if flags & CMD_VEL:
            out += _VEL.pack(*velocity)
        previous = (buttons, yaw, pitch, velocity)
    return bytes(out)


def decode_commands(data: bytes, offset: int, count: int):
    """Inverse of :func:`encode_commands`; returns ``(list, new_offset)``."""
    commands = []
    buttons, yaw, pitch, velocity = 0, 0.0, 0.0, (0, 0, 0)
    for _ in range(count):
        flags = data[offset]
        offset += 1
        if flags & CMD_BUTTONS:
            buttons = _U16.unpack_from(data, offset)[0]
            offset += 2
        if flags & CMD_YAW:
            yaw = _F32.unpack_from(data, offset)[0]
            offset += 4
        if flags & CMD_PITCH:
            pitch = _F32.unpack_from(data, offset)[0]
            offset += 4
        if flags & CMD_VEL:
            velocity = _VEL.unpack_from(data, offset)
            offset += 6
        commands.append((buttons, yaw, pitch, velocity))
    return commands, offset


# =========================================================================== #
#  Snapshot entity codec (delta compressed against an acknowledged baseline)
# =========================================================================== #
class WorldFrame:
    """Quantised state of every entity slot at one server tick."""

    __slots__ = ("tick", "active", "T", "O", "S")

    def __init__(self, capacity: int = C.MAX_PLAYERS):
        self.tick = -1
        self.active = np.zeros(capacity, dtype=bool)
        self.T = np.zeros(capacity, dtype=ENT_T)
        self.O = np.zeros(capacity, dtype=ENT_O)
        self.S = np.zeros(capacity, dtype=ENT_S)

    def copy(self) -> "WorldFrame":
        out = WorldFrame(self.active.shape[0])
        out.tick = self.tick
        out.active = self.active.copy()
        out.T = self.T.copy()
        out.O = self.O.copy()
        out.S = self.S.copy()
        return out


def quantize_world(store: PlayerStore, tick: int) -> WorldFrame:
    """Vectorised conversion of the authoritative float state to wire integers."""
    n = store.capacity
    frame = WorldFrame(n)
    frame.tick = tick
    act = store.active
    frame.active = act.copy()
    mv, ws = store.move, store.weapons
    pos = np.clip(np.rint(mv.pos * C.POS_QUANT), -32767, 32767).astype(np.int16)
    vel = np.clip(np.rint(mv.vel * C.VEL_QUANT), -32767, 32767).astype(np.int16)
    frame.T["px"], frame.T["py"], frame.T["pz"] = pos[:, 0], pos[:, 1], pos[:, 2]
    frame.T["vx"], frame.T["vy"], frame.T["vz"] = vel[:, 0], vel[:, 1], vel[:, 2]
    frame.O["yaw"] = (np.rint(M.wrap_angle(store.yaw) * C.ANGLE_QUANT).astype(np.int64) & 0xFFFF).astype(np.uint16)
    frame.O["pitch"] = np.clip(np.rint(store.pitch * C.ANGLE_QUANT), -32767, 32767).astype(np.int16)
    mf = mv.mflags
    flags = (np.where(store.alive, EF_ALIVE, 0) | np.where(mf & C.MF_ON_GROUND, EF_GROUND, 0)
             | np.where(mf & C.MF_CROUCHED, EF_CROUCH, 0) | np.where(mf & C.MF_SLIDING, EF_SLIDE, 0)
             | np.where(mf & C.MF_SPRINTING, EF_SPRINT, 0) | np.where(mf & C.MF_ADS, EF_ADS, 0)
             | np.where(ws.reload_end > 0, EF_RELOAD, 0) | np.where(store.is_bot, EF_BOT, 0))
    frame.S["flags"] = flags.astype(np.uint8)
    frame.S["health"] = np.clip(np.ceil(store.health), 0, 255).astype(np.uint8)
    frame.S["weapon"] = ws.weapon[np.arange(n), ws.slot].astype(np.uint8)
    frame.S["crouch"] = np.clip(np.rint(mv.crouch_t * 255.0), 0, 255).astype(np.uint8)
    frame.S["seq"] = (store.spawn_seq & 0xFF).astype(np.uint8)
    frame.S["team"] = (store.team & 0xFF).astype(np.uint8)
    idle = ~act
    frame.T[idle] = 0
    frame.O[idle] = 0
    frame.S[idle] = 0
    return frame


def encode_entities(frame: WorldFrame, baseline: WorldFrame | None) -> bytes:
    """
    Delta encode ``frame`` against ``baseline`` (``None`` produces a full
    snapshot).  Three 16-bit masks say which entities carry a transform,
    orientation or state record; the records follow as packed arrays.
    """
    act = frame.active
    if baseline is None:
        mt = mo = ms = act
    else:
        fresh = act & ~baseline.active
        mt = act & ((frame.T != baseline.T) | fresh)
        mo = act & ((frame.O != baseline.O) | fresh)
        ms = act & ((frame.S != baseline.S) | fresh)
    return (ENT_MASKS.pack(mask_to_int(act), mask_to_int(mt), mask_to_int(mo), mask_to_int(ms))
            + frame.T[mt].tobytes() + frame.O[mo].tobytes() + frame.S[ms].tobytes())


def decode_entities(data: bytes, offset: int, baseline: WorldFrame | None, capacity: int = C.MAX_PLAYERS):
    """Inverse of :func:`encode_entities`; returns ``(frame, new_offset)``."""
    active, mt, mo, ms = ENT_MASKS.unpack_from(data, offset)
    offset += ENT_MASKS.size
    frame = WorldFrame(capacity) if baseline is None else baseline.copy()
    frame.active = ((active >> np.arange(capacity)) & 1).astype(bool)
    for mask, name, dtype in ((mt, "T", ENT_T), (mo, "O", ENT_O), (ms, "S", ENT_S)):
        rows = int_to_rows(mask, capacity)
        if rows.size:
            getattr(frame, name)[rows] = np.frombuffer(data, dtype=dtype, count=rows.size, offset=offset)
            offset += rows.size * dtype.itemsize
    idle = ~frame.active
    frame.T[idle] = 0
    frame.O[idle] = 0
    frame.S[idle] = 0
    return frame, offset


# =========================================================================== #
#  Event codec
# =========================================================================== #
def encode_event(event: tuple) -> bytes:
    """Serialise one simulation event (type byte + fixed struct)."""
    kind = event[0]
    fmt = _EVENT_FORMATS[kind]
    if kind == S.EV_HIT:
        _, _shooter, victim, damage, zone, killed, point = event
        body = fmt.pack(victim, min(int(round(damage * 10.0)), 65535), zone, int(killed), *quantize_position(point))
    elif kind == S.EV_DAMAGE:
        _, _victim, attacker, damage, zone, source = event
        body = fmt.pack(attacker, min(int(round(damage * 10.0)), 65535), zone, *quantize_position(source))
    elif kind == S.EV_KILL:
        body = fmt.pack(event[1] & 0xFF, event[2], event[3] & 0xFF, event[4])
    elif kind == S.EV_SPAWN:
        body = fmt.pack(event[1], *quantize_position(event[2]),
                        int(round(float(M.wrap_angle(event[3])) * C.ANGLE_QUANT)) & 0xFFFF)
    elif kind == S.EV_SHOT:
        body = fmt.pack(event[1], event[2] & 0xFF, *quantize_position(event[3]), *quantize_position(event[4]))
    elif kind == S.EV_EXPLOSION:
        body = fmt.pack(*quantize_position(event[1]), event[2] & 0xFF, event[3] & 0xFF,
                        (event[4] if len(event) > 4 else 0) & 0xFF)
    elif kind == S.EV_GRENADE:
        body = fmt.pack(event[1] & 0xFF, event[2] & 0xFF, *quantize_position(event[3]), *quantize_velocity(event[4]),
                        min(int(event[5]), 65535), int(event[6]) & 0xFFFFFFFF, (event[7] if len(event) > 7 else 0) & 0xFF)
    elif kind == S.EV_STICK:
        body = fmt.pack(event[1] & 0xFF, event[2] & 0xFF, *quantize_position(event[3]))
    elif kind == S.EV_STATUS:
        body = fmt.pack(event[2] & 0xFF, min(int(event[3]), 65535))
    elif kind == S.EV_REWARD:
        body = fmt.pack(int(event[2]) & 0xFFFFFFFF, min(int(event[3]), 65535), event[4])
    elif kind == S.EV_MATCH:
        body = fmt.pack(event[1], event[2] & 0xFF, event[3], event[4], (event[5] if len(event) > 5 else 0) & 0xFFFFFFFF)
    else:
        raise ValueError("unknown event type %r" % (kind,))
    return bytes((kind,)) + body


def decode_event(data: bytes, offset: int):
    """Returns ``(event_tuple, new_offset)``.  Recipient-implied ids are omitted."""
    kind = data[offset]
    fmt = _EVENT_FORMATS[kind]
    f = fmt.unpack_from(data, offset + 1)
    offset += 1 + fmt.size
    if kind == S.EV_HIT:
        event = (kind, f[0], f[1] / 10.0, f[2], bool(f[3]), _unq(f[4:7], C.POS_QUANT))
    elif kind == S.EV_DAMAGE:
        event = (kind, f[0], f[1] / 10.0, f[2], _unq(f[3:6], C.POS_QUANT))
    elif kind == S.EV_KILL:
        event = (kind, f[0], f[1], f[2], f[3])
    elif kind == S.EV_SPAWN:
        event = (kind, f[0], _unq(f[1:4], C.POS_QUANT), float(M.wrap_angle(f[4] / C.ANGLE_QUANT)))
    elif kind == S.EV_SHOT:
        event = (kind, f[0], f[1], _unq(f[2:5], C.POS_QUANT), _unq(f[5:8], C.POS_QUANT))
    elif kind == S.EV_EXPLOSION:
        event = (kind, _unq(f[0:3], C.POS_QUANT), f[3], f[4], f[5])
    elif kind == S.EV_GRENADE:
        event = (kind, f[0], f[1], _unq(f[2:5], C.POS_QUANT), _unq(f[5:8], C.VEL_QUANT), f[8], f[9], f[10])
    elif kind == S.EV_STICK:
        event = (kind, f[0], f[1], _unq(f[2:5], C.POS_QUANT))
    elif kind == S.EV_STATUS:
        event = (kind, f[0], f[1])
    elif kind == S.EV_REWARD:
        event = (kind, f[0], f[1], f[2])
    else:
        event = (kind, f[0], f[1], f[2], f[3], f[4])
    return event, offset


# =========================================================================== #
#  1. LAN DISCOVERY
# =========================================================================== #
@dataclass
class ServerInfo:
    """One row of the LAN lobby."""
    address: str
    port: int
    name: str
    players: int
    max_players: int
    bots: int
    map_index: int
    mode: int
    tick_rate: int
    uid: int
    protocol: int
    last_seen: float = 0.0
    map_crc: int = 0

    @property
    def has_map(self) -> bool:
        """True when this installation has the very arena the server is running."""
        return local_map_matches(self.map_index, self.map_crc)

    @property
    def map_name(self) -> str:
        return maps.MAP_NAMES[self.map_index] if self.has_map else "CUSTOM ARENA"

    @property
    def mode_name(self) -> str:
        return C.MODE_NAMES[self.mode % len(C.MODE_NAMES)]

    @property
    def compatible(self) -> bool:
        return self.protocol == C.PROTOCOL_VERSION

    @property
    def joinable(self) -> bool:
        return self.compatible and self.has_map


def local_map_matches(map_index: int, checksum: int) -> bool:
    """Does the local map registry hold exactly the arena a server announced?"""
    return 0 <= map_index < maps.MAP_COUNT and maps.get_map_data(map_index).checksum == checksum


def encode_beacon(name: str, port: int, players: int, max_players: int, bots: int, map_index: int, mode: int,
                  uid: int, map_crc: int | None = None) -> bytes:
    """Sealed (encrypted + authenticated) server advertisement."""
    raw_name = name.encode("utf-8")[:32]
    if map_crc is None:
        map_crc = maps.get_map_data(map_index).checksum
    plain = BEACON_STRUCT.pack(C.PROTOCOL_VERSION, port, players, max_players, bots, map_index, mode,
                               C.TICK_RATE, uid & 0xFFFFFFFF, map_crc & 0xFFFFFFFF, len(raw_name)) + raw_name
    return BEACON_MAGIC + crypto.seal(BEACON_KEY, plain)


def decode_beacon(datagram: bytes, address: str) -> ServerInfo | None:
    """Parse a beacon; anything that is not a genuine sealed beacon yields ``None``."""
    if len(datagram) < len(BEACON_MAGIC) or datagram[:4] != BEACON_MAGIC:
        return None
    try:
        plain = crypto.unseal(BEACON_KEY, datagram[4:])
        (protocol, port, players, max_players, bots, map_index, mode, rate, uid, map_crc,
         length) = BEACON_STRUCT.unpack_from(plain, 0)
        name = plain[BEACON_STRUCT.size: BEACON_STRUCT.size + length].decode("utf-8", "replace")
    except (crypto.AuthenticationError, struct.error):
        return None
    return ServerInfo(address, port, name, players, max_players, bots, map_index, mode, rate, uid, protocol,
                      map_crc=map_crc)


def local_ipv4_addresses() -> list:
    """Best effort list of this machine's IPv4 addresses (no external packets are sent)."""
    found = set()
    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            found.add(address)
    except OSError:
        pass
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        found.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    return sorted(a for a in found if not a.startswith("127."))


def broadcast_targets(port: int = C.DISCOVERY_PORT) -> list:
    """Limited broadcast, every /24 directed broadcast of local interfaces, and loopback."""
    targets = [(C.BROADCAST_ADDRESS, port)]
    for address in local_ipv4_addresses():
        parts = address.split(".")
        if len(parts) == 4:
            targets.append((".".join(parts[:3] + ["255"]), port))
    targets.append(("127.0.0.1", port))
    return list(dict.fromkeys(targets))


class BeaconBroadcaster(threading.Thread):
    """Server side: advertise the match on the LAN every ``BEACON_INTERVAL`` seconds."""

    def __init__(self, describe, port: int = C.DISCOVERY_PORT, interval: float = C.BEACON_INTERVAL):
        super().__init__(name="BeaconBroadcaster", daemon=True)
        self.describe = describe                  # callable returning the beacon datagram
        self.port = port
        self.interval = float(interval)
        self._halt = threading.Event()
        self.sent = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def run(self) -> None:
        targets = broadcast_targets(self.port)
        refresh = 0
        while not self._halt.is_set():
            datagram = self.describe()
            for target in targets:
                try:
                    self.sock.sendto(datagram, target)
                    self.sent += 1
                except OSError:
                    continue
            refresh += 1
            if refresh % 15 == 0:
                targets = broadcast_targets(self.port)
            self._halt.wait(self.interval)
        self.sock.close()

    def stop(self) -> None:
        self._halt.set()


class LanBrowser(threading.Thread):
    """
    Client side: background listener on the discovery port.  ``servers()``
    returns the hosts heard from within the last ``BEACON_EXPIRY`` seconds.
    """

    def __init__(self, port: int = C.DISCOVERY_PORT):
        super().__init__(name="LanBrowser", daemon=True)
        self.port = port
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._servers: dict = {}
        self.error = ""
        self.received = 0
        self.sock = None

    def _open(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.bind(("", self.port))
            sock.settimeout(0.25)
            self.sock = sock
            self.error = ""
            return True
        except OSError as error:
            self.error = "cannot listen on UDP %d: %s" % (self.port, error)
            self.sock = None
            return False

    def run(self) -> None:
        while not self._halt.is_set():
            if self.sock is None and not self._open():
                self._halt.wait(1.0)
                continue
            try:
                datagram, sender = self.sock.recvfrom(C.RECV_BUFFER)
            except socket.timeout:
                continue
            except OSError:
                if self._halt.is_set():
                    break
                continue
            info = decode_beacon(datagram, sender[0])
            if info is None:
                continue
            info.last_seen = time.monotonic()
            self.received += 1
            with self._lock:
                known = self._servers.get(info.uid)
                if known is not None and known.address.startswith("127.") and not info.address.startswith("127."):
                    info.address = known.address          # same machine: keep the loopback route
                self._servers[info.uid] = info
        if self.sock is not None:
            self.sock.close()

    def servers(self) -> list:
        """Live server list, oldest entries pruned, sorted by name."""
        deadline = time.monotonic() - C.BEACON_EXPIRY
        with self._lock:
            for uid in [u for u, s in self._servers.items() if s.last_seen < deadline]:
                del self._servers[uid]
            return sorted(self._servers.values(), key=lambda s: (s.name.lower(), s.uid))

    def stop(self) -> None:
        self._halt.set()


# =========================================================================== #
#  Artificial network conditions (tests / debugging)
# =========================================================================== #
class NetConditioner(threading.Thread):
    """Delays and drops outgoing datagrams to emulate latency, jitter and loss."""

    def __init__(self, latency_ms: float = 0.0, jitter_ms: float = 0.0, loss: float = 0.0, seed: int = 1):
        super().__init__(name="NetConditioner", daemon=True)
        self.latency = latency_ms / 1000.0
        self.jitter = jitter_ms / 1000.0
        self.loss = float(loss)
        self._rng = random.Random(seed)
        self._heap: list = []
        self._cond = threading.Condition()
        self._halt = False
        self._order = 0
        self.dropped = 0

    @property
    def passive(self) -> bool:
        return self.latency <= 0.0 and self.jitter <= 0.0 and self.loss <= 0.0

    def send(self, sock, data: bytes, address) -> None:
        if self.passive:
            sock.sendto(data, address)
            return
        if self.loss > 0.0 and self._rng.random() < self.loss:
            self.dropped += 1
            return
        due = time.perf_counter() + max(self.latency + self._rng.uniform(-self.jitter, self.jitter), 0.0)
        with self._cond:
            self._order += 1
            heapq.heappush(self._heap, (due, self._order, sock, data, address))
            self._cond.notify()

    def run(self) -> None:
        while True:
            with self._cond:
                while not self._halt and not self._heap:
                    self._cond.wait(0.25)
                if self._halt:
                    return
                due = self._heap[0][0]
                wait = due - time.perf_counter()
                if wait > 0.0:
                    self._cond.wait(min(wait, 0.25))
                    continue
                _, _, sock, data, address = heapq.heappop(self._heap)
            try:
                sock.sendto(data, address)
            except OSError:
                pass

    def stop(self) -> None:
        with self._cond:
            self._halt = True
            self._cond.notify_all()


# =========================================================================== #
#  3. LAG COMPENSATION (authoritative server history rollback)
# =========================================================================== #
class LagCompensator:
    """
    Sliding ring buffer of the last ``HISTORY_TICKS`` (30 ticks = 500 ms)
    spatial states of every player: the exact 4x4 root transform plus the
    position / heading quaternion / crouch blend it was built from.

    ``rewind`` is a context manager.  Inside it the world's hitboxes are the
    ones that existed at the requested timestamp (interpolated between the two
    bracketing ticks exactly like the client interpolated them on screen);
    on exit the present-tick hitboxes are restored, even if the body raised.
    """

    def __init__(self, capacity: int = C.MAX_PLAYERS, history: int = C.HISTORY_TICKS):
        self.history = int(history)
        self.capacity = int(capacity)
        h, n = self.history, self.capacity
        self.ticks = np.full(h, -1, dtype=np.int64)
        self.root = np.tile(np.eye(4), (h, n, 1, 1))
        self.pos = np.zeros((h, n, 3), dtype=np.float64)
        self.quat = np.zeros((h, n, 4), dtype=np.float64)
        self.quat[..., 0] = 1.0
        self.crouch = np.zeros((h, n), dtype=np.float64)
        self.valid = np.zeros((h, n), dtype=bool)
        self.seq = np.zeros((h, n), dtype=np.int64)
        self.newest = -1
        self.rewinds = 0
        self.clamped = 0
        self.last_rewind_ticks = 0.0

    def record(self, tick: int, store: PlayerStore) -> None:
        """Store the present transforms as history slot ``tick``."""
        slot = tick % self.history
        self.ticks[slot] = tick
        self.root[slot] = store.hitboxes.root
        self.pos[slot] = store.move.pos
        self.quat[slot] = M.quat_from_yaw_pitch(store.yaw, 0.0)
        self.crouch[slot] = store.move.crouch_t
        self.valid[slot] = store.alive & store.active
        self.seq[slot] = store.spawn_seq
        self.newest = tick

    def sample(self, target_ms: float):
        """
        Historical ``(root, crouch, valid, seq)`` at ``target_ms`` on the
        server time line.  Requests outside the window are clamped to it.
        """
        h = self.history
        recorded = self.ticks[self.ticks >= 0]
        oldest = int(max(self.newest - (h - 1), recorded.min() if recorded.size else self.newest))
        want = target_ms / C.TICK_MS
        t = min(max(want, float(oldest)), float(self.newest))
        if t != want:
            self.clamped += 1
        self.last_rewind_ticks = self.newest - t
        t0 = int(np.floor(t))
        frac = t - t0
        s0 = t0 % h
        if frac < 1.0e-6 or t0 >= self.newest:
            return self.root[s0].copy(), self.crouch[s0].copy(), self.valid[s0].copy(), self.seq[s0].copy()
        s1 = (t0 + 1) % h
        p0, p1 = self.pos[s0], self.pos[s1]
        jumped = (self.seq[s0] != self.seq[s1]) | (np.sum((p1 - p0) ** 2, axis=1) > 36.0)
        w = np.where(jumped, 1.0 if frac >= 0.5 else 0.0, frac)
        pos = M.lerp(p0, p1, w[:, None])
        quat = M.quat_slerp(self.quat[s0], self.quat[s1], w)
        crouch = M.lerp(self.crouch[s0], self.crouch[s1], w)
        nearest = s1 if frac >= 0.5 else s0
        return M.rigid_from_pos_quat(pos, quat), crouch, self.valid[nearest].copy(), self.seq[nearest].copy()

    @contextmanager
    def rewind(self, store: PlayerStore, target_ms: float, shooter: int):
        """Temporarily roll every hitbox back to ``target_ms`` (the shooter keeps its present state)."""
        saved = store.hitboxes.snapshot()
        root, crouch, valid, seq = self.sample(target_ms)
        valid &= store.alive & store.active & (seq == store.spawn_seq)
        valid[shooter] = False
        store.hitboxes.set_from_matrices(root, crouch, valid)
        self.rewinds += 1
        try:
            yield self
        finally:
            store.hitboxes.restore(saved)


# =========================================================================== #
#  Server
# =========================================================================== #
BOT_NAMES = ("VEXA", "KODO", "RIVET", "NOVA", "ECHO", "SABLE", "PIKE", "ONYX", "FLUX", "TALON", "ZEPH", "MIRA",
             "DRIFT", "QUILL", "EMBER", "LYNX")
_PISTOL_CLASS = C.WEAPON_CLASSES.index("PISTOL")
BOT_PRIMARIES = tuple(key for i, key in enumerate(W.WEAPON_KEYS) if int(W.WEAPONS["klass"][i]) != _PISTOL_CLASS)
BOT_SECONDARIES = tuple(key for i, key in enumerate(W.WEAPON_KEYS) if int(W.WEAPONS["klass"][i]) == _PISTOL_CLASS)


@dataclass
class ServerConfig:
    name: str = "NEXUS SERVER"
    port: int = C.GAME_PORT
    map_key: str = "datacore"
    mode: int = C.MODE_TDM
    max_players: int = 12
    bot_fill: int = 8                 # total players the server tops up to with bots
    bot_difficulty: int = 1
    score_limit: int = 0              # 0 = mode default
    time_limit: float = C.MATCH_TIME_LIMIT
    upgrades_enabled: bool = True
    rotate_maps: bool = True
    beacon: bool = True
    seed: int = 1234
    fake_lag_ms: float = 0.0
    fake_jitter_ms: float = 0.0
    fake_loss: float = 0.0


class Connection:
    """Server side record of one remote client."""

    __slots__ = ("addr", "token", "pid", "nonce", "name", "last_recv", "cmds", "newest_cmd", "consumed", "budget",
                 "fire", "ack_shot", "ack_snapshot", "events", "next_event_seq", "transient", "echo_time",
                 "echo_recv", "ping_ms", "fire_missing", "client_velocity", "starved")

    def __init__(self, addr, token: int, pid: int, nonce: int, name: str, start_tick: int, now: float):
        self.addr = addr
        self.token = token
        self.pid = pid
        self.nonce = nonce
        self.name = name
        self.last_recv = now
        self.cmds: deque = deque()            # (tick, buttons, yaw, pitch)
        self.newest_cmd = start_tick
        self.consumed = start_tick
        self.budget = 2.0
        self.fire: dict = {}                  # command tick -> (shot sequence, client view time ms)
        self.ack_shot = 0
        self.ack_snapshot = 0
        self.events: deque = deque()          # (sequence, encoded bytes) awaiting acknowledgement
        self.next_event_seq = 1
        self.transient: list = []
        self.echo_time = 0
        self.echo_recv = now
        self.ping_ms = 0
        self.fire_missing = 0
        self.client_velocity = (0, 0, 0)
        self.starved = 0


class GameServer:
    """
    Authoritative 60 Hz game server.

    ``start()`` runs the receive loop and the tick loop on background threads
    (listen server inside the game process); ``serve_forever()`` runs the tick
    loop on the calling thread (dedicated server).
    """

    def __init__(self, config: ServerConfig | None = None):
        self.config = config or ServerConfig()
        cfg = self.config
        self.world = S.GameWorld(cfg.map_key, mode=cfg.mode, score_limit=cfg.score_limit or None,
                                 time_limit=cfg.time_limit, upgrades_enabled=cfg.upgrades_enabled, seed=cfg.seed)
        self.bots = B.BotController(self.world, cfg.bot_difficulty, seed=cfg.seed + 17)
        self.lag = LagCompensator(self.world.store.capacity)
        self.lock = threading.RLock()
        self.conns: dict = {}                 # address -> Connection
        self.by_pid: dict = {}                # player id -> Connection
        self.inbox: deque = deque()
        self.ring: list = [None] * C.SNAPSHOT_RING
        self.uid = random.SystemRandom().getrandbits(32)
        self.running = False
        self.port = 0
        self.sock = None
        self.beacon = None
        self.threads: list = []
        self.last_roster = 0.0
        self._rng = random.Random(cfg.seed)
        self.conditioner = NetConditioner(cfg.fake_lag_ms, cfg.fake_jitter_ms, cfg.fake_loss, cfg.seed)
        self.stats = {"ticks": 0, "malformed": 0, "bytes_in": 0, "bytes_out": 0, "max_tick_ms": 0.0,
                      "mean_tick_ms": 0.0, "fire_missing": 0, "connects": 0, "snapshots": 0, "full_snapshots": 0}
        self.lag.record(self.world.tick, self.world.store)

    # ---- lifecycle ---------------------------------------------------------------------
    def open(self) -> int:
        """Bind the gameplay socket (first free port at or above the configured one)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        last_error = None
        for candidate in range(self.config.port, self.config.port + C.GAME_PORT_SEARCH):
            try:
                sock.bind(("0.0.0.0", candidate))
                self.port = candidate
                break
            except OSError as error:
                last_error = error
        else:
            sock.close()
            raise OSError("no free UDP port for the game server: %s" % last_error)
        sock.settimeout(0.2)
        self.sock = sock
        self.running = True
        receiver = threading.Thread(target=self._receive_loop, name="ServerReceive", daemon=True)
        receiver.start()
        self.threads.append(receiver)
        if not self.conditioner.passive:
            self.conditioner.start()
        if self.config.beacon:
            self.beacon = BeaconBroadcaster(self._beacon_datagram)
            self.beacon.start()
        return self.port

    def start(self) -> int:
        """Open the socket and run the simulation on a background thread."""
        port = self.open()
        ticker = threading.Thread(target=self._tick_loop, name="ServerTick", daemon=True)
        ticker.start()
        self.threads.append(ticker)
        return port

    def serve_forever(self) -> None:
        if self.sock is None:
            self.open()
        self._tick_loop()

    def stop(self) -> None:
        if not self.running and self.sock is None:
            return
        self.running = False
        for conn in list(self.conns.values()):
            self._send_raw(make_packet(PKT_DISCONNECT, conn.token, b"\x00"), conn.addr)
        if self.beacon is not None:
            self.beacon.stop()
        self.conditioner.stop()
        current = threading.current_thread()
        for thread in self.threads:
            if thread is not current:
                thread.join(timeout=1.5)
        self.threads.clear()
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _beacon_datagram(self) -> bytes:
        world = self.world
        return encode_beacon(self.config.name, self.port, len(self.conns), self.config.max_players,
                             int(world.bots().size), world.map_index, world.mode, self.uid,
                             world.loaded.data.checksum)

    # ---- threads -----------------------------------------------------------------------------
    def _receive_loop(self) -> None:
        sock = self.sock
        while self.running:
            try:
                data, addr = sock.recvfrom(C.RECV_BUFFER)
            except socket.timeout:
                continue
            except OSError:
                if not self.running:
                    break
                continue
            self.inbox.append((data, addr, time.perf_counter()))

    def _tick_loop(self) -> None:
        next_time = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            remaining = next_time - now
            if remaining > 0.0:
                time.sleep(remaining - 0.001 if remaining > 0.002 else 0.0)
                continue
            if -remaining > 0.25:
                next_time = now                     # stalled: resynchronise instead of bursting
            self.tick_once()
            next_time += C.TICK_DT

    # ---- sending -----------------------------------------------------------------------------------
    def _send_raw(self, datagram: bytes, addr) -> None:
        if self.sock is None:
            return
        try:
            self.conditioner.send(self.sock, datagram, addr)
            self.stats["bytes_out"] += len(datagram)
        except OSError:
            pass

    # ---- inbound packets -------------------------------------------------------------------------------
    def _drain_inbox(self) -> None:
        inbox = self.inbox
        while inbox:
            data, addr, received = inbox.popleft()
            self.stats["bytes_in"] += len(data)
            try:
                self._handle_packet(data, addr, received)
            except (struct.error, IndexError, ValueError):
                self.stats["malformed"] += 1

    def _handle_packet(self, data: bytes, addr, received: float) -> None:
        magic, packet_type, token = HEADER.unpack_from(data, 0)
        if magic != C.PROTOCOL_MAGIC:
            return
        offset = HEADER.size
        if packet_type == PKT_CONNECT_REQ:
            self._handle_connect(data, offset, addr, received)
            return
        conn = self.conns.get(addr)
        if conn is None or conn.token != token:
            return
        conn.last_recv = received
        if packet_type == PKT_CLIENT_FRAME:
            self._handle_frame(conn, data, offset, received)
        elif packet_type == PKT_DISCONNECT:
            self._drop(conn)

    def _handle_connect(self, data: bytes, offset: int, addr, received: float) -> None:
        protocol, nonce, team_pref, name_len = CONNECT_REQ.unpack_from(data, offset)
        offset += CONNECT_REQ.size
        known = self.conns.get(addr)
        if known is not None and known.nonce == nonce:
            self._send_ack(known)                      # our earlier acknowledgement was lost
            return
        if protocol != C.PROTOCOL_VERSION:
            self._send_raw(make_packet(PKT_CONNECT_DENY, 0, CONNECT_DENY.pack(nonce, DENY_VERSION)), addr)
            return
        name = data[offset: offset + name_len].decode("utf-8", "replace")
        offset += name_len
        try:
            weapon_ids, levels, operator, prestige = W.decode_loadout(data[offset: offset + W.LOADOUT_BLOB_SIZE])
            offset += W.LOADOUT_BLOB_SIZE
            equipment = EQ.sanitize(EQUIP_BLOB.unpack_from(data, offset)) if len(data) >= offset + EQUIP_BLOB.size \
                else EQ.DEFAULT_EQUIPMENT
        except ValueError:
            self._send_raw(make_packet(PKT_CONNECT_DENY, 0, CONNECT_DENY.pack(nonce, DENY_BAD_REQUEST)), addr)
            return
        if known is not None:
            self._drop(known)
        world = self.world
        if len(self.conns) >= self.config.max_players:
            self._send_raw(make_packet(PKT_CONNECT_DENY, 0, CONNECT_DENY.pack(nonce, DENY_FULL)), addr)
            return
        if not np.any(~world.store.active):
            bots = world.bots()
            if bots.size:
                world.remove_player(int(bots[-1]))
        team = team_pref if team_pref in (0, 1) else None
        pid = world.add_player(name, False, weapon_ids, levels, operator, prestige, team, equipment)
        if pid < 0:
            self._send_raw(make_packet(PKT_CONNECT_DENY, 0, CONNECT_DENY.pack(nonce, DENY_FULL)), addr)
            return
        token = random.SystemRandom().getrandbits(32) or 1
        conn = Connection(addr, token, pid, nonce, world.store.names[pid], world.tick, received)
        self.conns[addr] = conn
        self.by_pid[pid] = conn
        self.stats["connects"] += 1
        self._send_ack(conn)
        self.last_roster = 0.0

    def _send_ack(self, conn: Connection) -> None:
        world = self.world
        payload = CONNECT_ACK.pack(conn.nonce, conn.pid, conn.consumed & 0xFFFFFFFF, C.TICK_RATE, world.map_index,
                                   world.mode, int(world.store.team[conn.pid]) & 0xFF, conn.token,
                                   self.config.max_players, int(world.upgrades_enabled),
                                   min(world.score_limit, 65535), min(int(world.time_limit), 65535),
                                   world.loaded.data.checksum)
        self._send_raw(make_packet(PKT_CONNECT_ACK, conn.token, payload), conn.addr)

    def _handle_frame(self, conn: Connection, data: bytes, offset: int, received: float) -> None:
        (ack_snapshot, ack_event, client_ms, newest, ping, n_cmds, n_fire, lethal,
         tactical) = FRAME_HEADER.unpack_from(data, offset)
        offset += FRAME_HEADER.size
        self.world.request_equipment(conn.pid, (lethal, tactical))
        conn.ack_snapshot = ack_snapshot
        conn.echo_time = client_ms
        conn.echo_recv = received
        conn.ping_ms = ping
        events = conn.events
        while events and events[0][0] <= ack_event:
            events.popleft()
        commands, offset = decode_commands(data, offset, n_cmds)
        first = newest - n_cmds + 1
        for index, (buttons, yaw, pitch, velocity) in enumerate(commands):
            tick = first + index
            if tick > conn.newest_cmd:
                conn.cmds.append((tick, buttons, yaw, pitch))
                conn.newest_cmd = tick
                conn.client_velocity = velocity
        while len(conn.cmds) > C.CMD_QUEUE_LIMIT:
            conn.cmds.popleft()
        for _ in range(n_fire):
            seq, tick, view_ms, _slot = FIRE_RECORD.unpack_from(data, offset)
            offset += FIRE_RECORD.size
            if seq_newer(seq, conn.ack_shot):
                conn.ack_shot = seq
            if tick > conn.consumed and tick not in conn.fire:
                conn.fire[tick] = (seq, view_ms)

    def _drop(self, conn: Connection) -> None:
        self.conns.pop(conn.addr, None)
        self.by_pid.pop(conn.pid, None)
        self.world.remove_player(conn.pid)
        self.last_roster = 0.0

    def kick_silent(self, now: float) -> None:
        for conn in [c for c in self.conns.values() if now - c.last_recv > C.CONNECTION_TIMEOUT]:
            self._send_raw(make_packet(PKT_DISCONNECT, conn.token, b"\x01"), conn.addr)
            self._drop(conn)

    # ---- bots ---------------------------------------------------------------------------------------------
    def _manage_bots(self) -> None:
        world = self.world
        humans = len(self.conns)
        target = int(np.clip(self.config.bot_fill - humans, 0, max(self.config.max_players - humans, 0)))
        bots = world.bots()
        if bots.size < target:
            index = int(np.flatnonzero(~world.store.active)[0]) if np.any(~world.store.active) else 0
            primary = W.WEAPON_INDEX[BOT_PRIMARIES[self._rng.randrange(len(BOT_PRIMARIES))]]
            secondary = W.WEAPON_INDEX[BOT_SECONDARIES[self._rng.randrange(len(BOT_SECONDARIES))]]
            equipment = (EQ.LETHALS[self._rng.randrange(len(EQ.LETHALS))],
                         EQ.TACTICALS[self._rng.randrange(len(EQ.TACTICALS))])
            world.add_player("BOT " + BOT_NAMES[index % len(BOT_NAMES)], True, (primary, secondary), equipment=equipment)
            self.last_roster = 0.0
        elif bots.size > target:
            teams = world.store.team[bots]
            crowded = int(np.argmax(np.bincount(teams[teams < 2], minlength=2))) if np.any(teams < 2) else -1
            pick = bots[teams == crowded] if crowded >= 0 and np.any(teams == crowded) else bots
            world.remove_player(int(pick[-1]))
            self.last_roster = 0.0

    # ---- lag compensated fire -------------------------------------------------------------------------------
    def _rewind_for(self, row: int):
        """
        Rollback context for the round player ``row`` has just fired.  The
        FIRE record that travelled with the command supplies the client-side
        execution timestamp; without one (lost or never sent) the time is
        estimated from the client's measured ping.  Bots fire in the present.
        """
        conn = self.by_pid.get(row)
        if conn is None:
            return nullcontext()
        store = self.world.store
        record = conn.fire.pop(int(store.cmd_tick[row]), None)
        if record is None:
            conn.fire_missing += 1
            self.stats["fire_missing"] += 1
            view_ms = self.lag.newest * C.TICK_MS - C.INTERP_DELAY_MS - conn.ping_ms * 0.5
        else:
            view_ms = float(record[1])
        return self.lag.rewind(store, view_ms, row)

    # ---- one simulation tick ---------------------------------------------------------------------------------
    def tick_once(self) -> None:
        started = time.perf_counter()
        with self.lock:
            self._drain_inbox()
            self.kick_silent(started)
            self._manage_bots()
            world = self.world
            bot_rows = world.bots()
            if bot_rows.size:
                bot_buttons, bot_yaw, bot_pitch = self.bots.think(bot_rows)
            conns = list(self.conns.values())
            for conn in conns:
                # one command of budget per tick; a standing backlog drains 12 % faster than real time
                conn.budget = min(conn.budget + 1.0 + (0.12 if len(conn.cmds) > 2 else 0.0), C.CMD_BUDGET_MAX)
            for step in range(C.MAX_CMDS_PER_TICK):
                rows, buttons, yaws, pitches, ticks = [], [], [], [], []
                if step == 0 and bot_rows.size:
                    rows.extend(bot_rows.tolist())
                    buttons.extend(bot_buttons.tolist())
                    yaws.extend(bot_yaw.tolist())
                    pitches.extend(bot_pitch.tolist())
                    ticks.extend([world.tick + 1] * bot_rows.size)
                for conn in conns:
                    if conn.cmds and conn.budget >= 1.0:
                        tick, btn, yaw, pitch = conn.cmds.popleft()
                        conn.budget -= 1.0
                        conn.consumed = tick
                        rows.append(conn.pid)
                        buttons.append(btn)
                        yaws.append(yaw)
                        pitches.append(pitch)
                        ticks.append(tick)
                    elif step == 0:
                        conn.starved += 1
                if not rows:
                    break
                row_array = np.array(rows, dtype=np.intp)
                fired = world.step_commands(row_array, np.array(buttons, dtype=np.uint16), np.array(yaws),
                                            np.array(pitches), np.array(ticks, dtype=np.int64))
                if fired.any():
                    world.resolve_shots(row_array[fired], rewind=self._rewind_for)
            world.end_tick()
            if world.phase == C.PHASE_ENDED and world.next_map is None and self.config.rotate_maps:
                world.next_map = (world.map_index + 1) % maps.MAP_COUNT
            self.lag.record(world.tick, world.store)
            for conn in conns:
                if len(conn.fire) > 64:
                    for tick in [t for t in conn.fire if t <= conn.consumed]:
                        del conn.fire[tick]
            events = world.drain_events()
            self.bots.observe(events)
            self._route_events(events)
            self._send_snapshots(started)
            if started - self.last_roster >= C.ROSTER_INTERVAL:
                self._send_roster()
                self.last_roster = started
        elapsed = (time.perf_counter() - started) * 1000.0
        stats = self.stats
        stats["ticks"] += 1
        stats["max_tick_ms"] = max(stats["max_tick_ms"], elapsed)
        stats["mean_tick_ms"] += (elapsed - stats["mean_tick_ms"]) / min(stats["ticks"], 600)

    # ---- outbound state -----------------------------------------------------------------------------------------
    def _queue_reliable(self, conn: Connection, blob: bytes) -> None:
        conn.events.append((conn.next_event_seq, blob))
        conn.next_event_seq += 1
        if len(conn.events) > 512:
            conn.events.popleft()

    def _route_events(self, events: list) -> None:
        if not events or not self.conns:
            return
        by_pid = self.by_pid
        everyone = list(self.conns.values())
        for event in events:
            kind = event[0]
            if kind == S.EV_MATCH:
                event = event + (self.world.loaded.data.checksum,)
            blob = encode_event(event)
            if kind in (S.EV_HIT, S.EV_REWARD):
                conn = by_pid.get(event[1])
                if conn is not None:
                    self._queue_reliable(conn, blob)
            elif kind in (S.EV_DAMAGE, S.EV_STATUS):
                conn = by_pid.get(event[1])
                if conn is not None:
                    self._queue_reliable(conn, blob)
            elif kind == S.EV_SHOT:
                for conn in everyone:
                    if conn.pid != event[1] and len(conn.transient) < 24:
                        conn.transient.append(blob)
            else:
                for conn in everyone:
                    self._queue_reliable(conn, blob)

    def _send_snapshots(self, now: float) -> None:
        world = self.world
        store = world.store
        tick = world.tick
        ring = self.ring
        frame = quantize_world(store, tick)
        ring[tick % C.SNAPSHOT_RING] = frame
        if not self.conns:
            return
        winner = world.winner if world.phase == C.PHASE_ENDED else S.NO_PLAYER
        match = MATCH_STATE.pack(min(int(world.team_scores[0]), 65535), min(int(world.team_scores[1]), 65535),
                                 min(int(np.ceil(world.time_left)), 65535), world.phase, winner & 0xFF)
        encoded: dict = {}
        for conn in self.conns.values():
            baseline = ring[conn.ack_snapshot % C.SNAPSHOT_RING] if conn.ack_snapshot > 0 else None
            if baseline is not None and (baseline.tick != conn.ack_snapshot or tick - baseline.tick >= C.SNAPSHOT_RING - 1):
                baseline = None
            key = baseline.tick if baseline is not None else 0
            entities = encoded.get(key)
            if entities is None:
                entities = encoded[key] = encode_entities(frame, baseline)
            if key == 0:
                self.stats["full_snapshots"] += 1
            owner = store.pack_owner(conn.pid) if store.active[conn.pid] else b""
            room = C.MAX_DATAGRAM - (HEADER.size + SNAP_HEADER.size + MATCH_STATE.size + len(owner) + len(entities) + 4)
            reliable = bytearray()
            n_reliable = 0
            budget = min(C.EVENT_BYTES_PER_SNAPSHOT, room)
            for _, blob in conn.events:
                if len(reliable) + len(blob) > budget or n_reliable >= 255:
                    break
                reliable += blob
                n_reliable += 1
            room -= len(reliable)
            transient = bytearray()
            n_transient = 0
            for blob in conn.transient:
                if len(transient) + len(blob) > room or n_transient >= 255:
                    break
                transient += blob
                n_transient += 1
            conn.transient.clear()
            hold = int(min(max((now - conn.echo_recv) * 1000.0, 0.0), 65535.0))
            payload = SNAP_HEADER.pack(tick, key, conn.consumed & 0xFFFFFFFF, conn.ack_shot, conn.echo_time, hold,
                                       SNAP_HAS_OWNER if owner else 0, n_reliable, n_transient)
            payload += match + owner + entities
            if n_reliable:
                payload += _U32.pack(conn.events[0][0]) + bytes(reliable)
            payload += bytes(transient)
            self._send_raw(make_packet(PKT_SNAPSHOT, conn.token, payload), conn.addr)
            self.stats["snapshots"] += 1

    def _send_roster(self) -> None:
        if not self.conns:
            return
        store = self.world.store
        rows = store.active_rows()
        body = bytearray((rows.size,))
        for row in rows.tolist():
            conn = self.by_pid.get(row)
            name = store.names[row].encode("utf-8")[:16]
            flags = (1 if store.is_bot[row] else 0) | (2 if store.alive[row] else 0)
            body += ROSTER_ENTRY.pack(row, int(store.team[row]) & 0xFF, flags, min(int(store.kills[row]), 65535),
                                      min(int(store.deaths[row]), 65535), int(store.score[row]) & 0xFFFFFFFF,
                                      min(int(conn.ping_ms), 65535) if conn is not None else 0, len(name))
            body += name
        for conn in self.conns.values():
            self._send_raw(make_packet(PKT_ROSTER, conn.token, bytes(body)), conn.addr)

    def describe(self) -> dict:
        """Thread-safe summary for a host UI or the dedicated server console."""
        with self.lock:
            world = self.world
            return {"port": self.port, "tick": world.tick, "humans": len(self.conns), "bots": int(world.bots().size),
                    "map": maps.MAP_NAMES[world.map_index], "mode": C.MODE_NAMES[world.mode], "phase": world.phase,
                    "scores": world.team_scores.tolist(), "time_left": world.time_left,
                    "rewinds": self.lag.rewinds, "last_rewind_ticks": self.lag.last_rewind_ticks, **self.stats}


# =========================================================================== #
#  4. CLIENT SIDE PREDICTION & RECONCILIATION
# =========================================================================== #
_ROW0 = np.array([0], dtype=np.intp)


class Predictor:
    """
    Locally simulated copy of the player's own body.

    Every command is executed the moment it is sampled and kept, together
    with the state it produced, in a ring of ``PENDING_CMD_CAPACITY`` entries.
    ``reconcile`` compares the server's authoritative state for an
    acknowledged command with what was predicted for that same command; a
    position error above ``RECONCILE_THRESHOLD`` (0.05 units) -- or any
    disagreement in velocity, movement flags or weapon state -- makes the
    predictor adopt the server state and re-run every newer command at once.
    """

    def __init__(self, world, player_id: int, weapon_ids, levels, operator, prestige: int = 0):
        self.world = world
        self.player_id = int(player_id)
        self.store = PlayerStore(1)
        self.auth = PlayerStore(1)
        for store in (self.store, self.auth):
            store.active[0] = True
            store.weapons.set_loadout(0, weapon_ids, levels)
            store.apply_operator(0, operator, prestige)
        cap = C.PENDING_CMD_CAPACITY
        self.capacity = cap
        self.c_tick = np.full(cap, -1, dtype=np.int64)
        self.c_buttons = np.zeros(cap, dtype=np.uint16)
        self.c_yaw = np.zeros(cap, dtype=np.float64)
        self.c_pitch = np.zeros(cap, dtype=np.float64)
        self.c_pos = np.zeros((cap, 3), dtype=np.float64)
        self.c_vel = np.zeros((cap, 3), dtype=np.float64)
        self.c_flags = np.zeros(cap, dtype=np.uint16)
        self.c_weapon = np.zeros((cap, 4), dtype=np.int64)      # slot, ammo in hand, reload end, alive
        self.c_equip = np.zeros((cap, 4), dtype=np.int64)       # lethal ready, tactical ready, stun until, equipment
        self.newest_tick = -1
        self.acked_tick = -1
        self.visual_error = np.zeros(3, dtype=np.float64)
        self.corrections = 0
        self.resyncs = 0
        self.replayed = 0
        self.last_error = 0.0
        self.max_error = 0.0
        self.initialised = False

    def set_world(self, world) -> None:
        self.world = world
        self.store.move._cache_version = -1

    def _remember(self, index: int) -> None:
        st = self.store
        self.c_pos[index] = st.move.pos[0]
        self.c_vel[index] = st.move.vel[0]
        self.c_flags[index] = st.move.mflags[0]
        slot = int(st.weapons.slot[0])
        self.c_weapon[index] = (slot, st.weapons.ammo[0, slot], st.weapons.reload_end[0], int(st.alive[0]))
        self.c_equip[index] = (st.ability_ready[0], st.tactical_ready[0], st.stun_until[0],
                               int(st.equip[0, 0]) * 256 + int(st.equip[0, 1]))

    def predict(self, tick: int, buttons: int, yaw: float, pitch: float):
        """Execute one new command immediately; returns ``(fired, lethal_used, tactical_used)``."""
        index = tick % self.capacity
        self.c_tick[index] = tick
        self.c_buttons[index] = buttons
        self.c_yaw[index] = yaw
        self.c_pitch[index] = pitch
        fired, lethal, tactical = simulate_commands(self.store, self.world, _ROW0, buttons, yaw, pitch, tick)
        self._remember(index)
        self.newest_tick = tick
        return bool(fired[0]), bool(lethal[0]), bool(tactical[0])

    def reconcile(self, ack_tick: int, payload: bytes, offset: int = 0) -> bool:
        """
        Fold an authoritative owner state into the prediction.  Returns True
        when a rewind-and-replay correction was necessary.
        """
        auth, st = self.auth, self.store
        auth.unpack_owner(0, payload, offset)
        st.health[0] = auth.health[0]
        st.max_health[0] = auth.max_health[0]
        self.acked_tick = ack_tick
        index = ack_tick % self.capacity
        diverged = True
        comparable = self.initialised and self.c_tick[index] == ack_tick
        if comparable:
            delta = self.c_pos[index] - auth.move.pos[0]
            error = float(np.sqrt(np.dot(delta, delta)))
            dv = self.c_vel[index] - auth.move.vel[0]
            slot = int(auth.weapons.slot[0])
            same_flags = (int(self.c_flags[index]) & C.MF_PERSISTENT) == (int(auth.move.mflags[0]) & C.MF_PERSISTENT)
            same_weapon = (int(self.c_weapon[index, 0]) == slot
                           and int(self.c_weapon[index, 1]) == int(auth.weapons.ammo[0, slot])
                           and int(self.c_weapon[index, 2]) == int(auth.weapons.reload_end[0])
                           and int(self.c_weapon[index, 3]) == int(auth.alive[0]))
            same_equip = (int(self.c_equip[index, 0]) & 0xFFFFFFFF == int(auth.ability_ready[0])
                          and int(self.c_equip[index, 1]) & 0xFFFFFFFF == int(auth.tactical_ready[0])
                          and int(self.c_equip[index, 2]) & 0xFFFFFFFF == int(auth.stun_until[0])
                          and int(self.c_equip[index, 3]) == int(auth.equip[0, 0]) * 256 + int(auth.equip[0, 1]))
            self.last_error = error
            self.max_error = max(self.max_error, error)
            diverged = (error > C.RECONCILE_THRESHOLD or float(np.sqrt(np.dot(dv, dv))) > C.RECONCILE_VELOCITY_THRESHOLD
                        or not same_flags or not same_weapon or not same_equip)
        if not diverged:
            st.spawn_seq[0] = auth.spawn_seq[0]
            return False
        before = st.move.pos[0].copy()
        was_alive = bool(st.alive[0])
        respawned = int(st.spawn_seq[0]) != int(auth.spawn_seq[0])
        st.unpack_owner(0, payload, offset)
        st.move._cache_min[:] = np.inf
        st.move._cache_max[:] = -np.inf
        replayed = 0
        for tick in range(ack_tick + 1, self.newest_tick + 1):
            i = tick % self.capacity
            if self.c_tick[i] != tick:
                continue
            simulate_commands(st, self.world, _ROW0, self.c_buttons[i], self.c_yaw[i], self.c_pitch[i], tick)
            self._remember(i)
            replayed += 1
        if not comparable:
            self.resyncs += 1                     # nothing predicted for that tick yet (connect / long stall)
        if self.initialised:
            self.corrections += int(comparable)
            self.replayed += replayed
            shift = before - st.move.pos[0]
            if (respawned or was_alive != bool(st.alive[0])
                    or float(np.sqrt(np.dot(shift, shift))) > C.SMOOTH_CORRECTION_LIMIT):
                self.visual_error[:] = 0.0
            else:
                self.visual_error += shift
        self.initialised = True
        return True

    def decay_visual_error(self, dt: float) -> None:
        self.visual_error *= float(np.exp(-14.0 * dt))

    def render_position(self) -> np.ndarray:
        """Predicted feet position plus the fading visual correction offset."""
        return self.store.move.pos[0] + self.visual_error


# =========================================================================== #
#  5. ENTITY INTERPOLATION
# =========================================================================== #
class EntityView:
    """Interpolated remote entity state handed to the renderer."""

    __slots__ = ("pos", "vel", "yaw", "pitch", "crouch", "flags", "health", "weapon", "team", "active", "alive",
                 "render_ms", "fraction", "starved")

    def __init__(self, n: int):
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.yaw = np.zeros(n)
        self.pitch = np.zeros(n)
        self.crouch = np.zeros(n)
        self.flags = np.zeros(n, dtype=np.uint8)
        self.health = np.zeros(n)
        self.weapon = np.zeros(n, dtype=np.int64)
        self.team = np.zeros(n, dtype=np.int64)
        self.active = np.zeros(n, dtype=bool)
        self.alive = np.zeros(n, dtype=bool)
        self.render_ms = 0.0
        self.fraction = 0.0
        self.starved = False


class InterpolationBuffer:
    """
    Ring of the most recent snapshots in float form.  ``sample`` returns the
    world as it was ``INTERP_DELAY_MS`` (100 ms) ago: positions are linearly
    interpolated and orientations spherically interpolated (quaternion SLERP)
    between the two frames bracketing the render time, for all entities in
    one vectorised pass.
    """

    def __init__(self, capacity: int = C.MAX_PLAYERS, size: int = C.INTERP_BUFFER):
        k, n = int(size), int(capacity)
        self.size = k
        self.capacity = n
        self.time = np.full(k, -np.inf, dtype=np.float64)
        self.pos = np.zeros((k, n, 3), dtype=np.float64)
        self.vel = np.zeros((k, n, 3), dtype=np.float64)
        self.quat = np.zeros((k, n, 4), dtype=np.float64)
        self.quat[..., 0] = 1.0
        self.crouch = np.zeros((k, n), dtype=np.float64)
        self.state = np.zeros((k, n), dtype=ENT_S)
        self.active = np.zeros((k, n), dtype=bool)
        self.head = 0
        self.count = 0

    def clear(self) -> None:
        self.time[:] = -np.inf
        self.head = 0
        self.count = 0

    def push(self, frame: WorldFrame, time_ms: float) -> None:
        """Append a decoded snapshot (frames must arrive in increasing time)."""
        if self.count and time_ms <= self.time[(self.head - 1) % self.size]:
            return
        i = self.head
        t = frame.T
        self.time[i] = time_ms
        self.pos[i, :, 0] = t["px"] / C.POS_QUANT
        self.pos[i, :, 1] = t["py"] / C.POS_QUANT
        self.pos[i, :, 2] = t["pz"] / C.POS_QUANT
        self.vel[i, :, 0] = t["vx"] / C.VEL_QUANT
        self.vel[i, :, 1] = t["vy"] / C.VEL_QUANT
        self.vel[i, :, 2] = t["vz"] / C.VEL_QUANT
        self.quat[i] = M.quat_from_yaw_pitch(frame.O["yaw"] / C.ANGLE_QUANT, frame.O["pitch"] / C.ANGLE_QUANT)
        self.crouch[i] = frame.S["crouch"] / 255.0
        self.state[i] = frame.S
        self.active[i] = frame.active
        self.head = (i + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, render_ms: float, out: EntityView | None = None) -> EntityView:
        view = out if out is not None else EntityView(self.capacity)
        view.render_ms = render_ms
        view.starved = False
        if self.count == 0:
            view.active[:] = False
            view.alive[:] = False
            return view
        order = (self.head - self.count + np.arange(self.count)) % self.size
        times = self.time[order]
        k = int(np.searchsorted(times, render_ms, side="right")) - 1
        if k < 0:
            a = b = int(order[0])
            frac = 0.0
        elif k >= self.count - 1:
            a = b = int(order[-1])
            frac = 0.0
            view.starved = render_ms - times[-1] > 2.0 * C.TICK_MS
        else:
            a, b = int(order[k]), int(order[k + 1])
            frac = float((render_ms - times[k]) / max(times[k + 1] - times[k], 1.0e-6))
        view.fraction = frac
        sa, sb = self.state[a], self.state[b]
        both = self.active[a] & self.active[b]
        jumped = (sa["seq"] != sb["seq"]) | (np.sum((self.pos[b] - self.pos[a]) ** 2, axis=1) > 36.0)
        w = np.where(both & ~jumped, frac, np.where(self.active[b] & (~self.active[a] | (frac >= 0.5)), 1.0, 0.0))
        view.pos[:] = M.lerp(self.pos[a], self.pos[b], w[:, None])
        view.vel[:] = M.lerp(self.vel[a], self.vel[b], w[:, None])
        yaw, pitch = M.quat_to_yaw_pitch(M.quat_slerp(self.quat[a], self.quat[b], w))
        view.yaw[:] = yaw
        view.pitch[:] = pitch
        view.crouch[:] = M.lerp(self.crouch[a], self.crouch[b], w)
        near = np.where(w >= 0.5, sb, sa)
        view.flags[:] = near["flags"]
        view.health[:] = near["health"]
        view.weapon[:] = near["weapon"]
        view.team[:] = near["team"]
        view.active[:] = np.where(w >= 0.5, self.active[b], self.active[a])
        view.alive[:] = view.active & ((view.flags & EF_ALIVE) != 0)
        if a == b and render_ms > self.time[a]:
            ahead = min(render_ms - self.time[a], 100.0) / 1000.0     # bounded extrapolation when starved
            view.pos[view.alive] += view.vel[view.alive] * ahead
        return view


class TimeSync:
    """Estimates the server time line from snapshot arrival times (min-latency filter + slew)."""

    def __init__(self):
        self.offset = None
        self.samples: deque = deque(maxlen=40)

    def add(self, server_ms: float, local_ms: float) -> None:
        self.samples.append(server_ms - local_ms)
        target = max(self.samples)
        if self.offset is None or abs(target - self.offset) > 80.0:
            self.offset = target
        else:
            self.offset += min(max(target - self.offset, -0.4), 0.4)

    def server_ms(self, local_ms: float) -> float:
        return local_ms + (self.offset if self.offset is not None else 0.0)


# =========================================================================== #
#  Client
# =========================================================================== #
DISCONNECTED, CONNECTING, CONNECTED, FAILED = range(4)


class RosterEntry:
    __slots__ = ("pid", "team", "bot", "alive", "kills", "deaths", "score", "ping", "name")

    def __init__(self, pid, team, flags, kills, deaths, score, ping, name):
        self.pid, self.team, self.kills, self.deaths, self.score, self.ping, self.name = pid, team, kills, deaths, score, ping, name
        self.bot = bool(flags & 1)
        self.alive = bool(flags & 2)


class GameClient:
    """
    Network endpoint of one player: handshake, command upload with
    redundancy, snapshot decoding, prediction, interpolation and event
    delivery.  Drive it with ``update()`` once per rendered frame and
    ``send_command()`` once per simulation tick.
    """

    def __init__(self, name: str, weapon_ids, levels, operator, prestige: int = 0, fake_lag_ms: float = 0.0,
                 fake_jitter_ms: float = 0.0, fake_loss: float = 0.0, seed: int = 99, equipment=None):
        self.name = name[:16]
        self.equipment = EQ.sanitize(EQ.DEFAULT_EQUIPMENT if equipment is None else equipment)   # requested lethal, tactical
        self.weapon_ids = np.asarray(weapon_ids, dtype=np.int64)
        self.levels = np.asarray(levels, dtype=np.int64)
        self.operator = np.asarray(operator, dtype=np.int64)
        self.prestige = int(prestige)
        self.state = DISCONNECTED
        self.fail_reason = ""
        self.sock = None
        self.server = None
        self.token = 0
        self.nonce = random.SystemRandom().getrandbits(32)
        self.inbox: deque = deque()
        self.running = False
        self.thread = None
        self.conditioner = NetConditioner(fake_lag_ms, fake_jitter_ms, fake_loss, seed)
        # session ------------------------------------------------------------
        self.pid = -1
        self.team = 0
        self.map_index = 0
        self.mode = C.MODE_TDM
        self.max_players = C.MAX_PLAYERS
        self.upgrades_enabled = True
        self.score_limit = 0
        self.time_limit = 0
        self.map_changed = False
        self.loaded = None
        self.predictor = None
        self.interp = InterpolationBuffer()
        self.view = EntityView(C.MAX_PLAYERS)
        self.sync = TimeSync()
        self.frames: list = [None] * C.SNAPSHOT_RING
        self.last_snapshot_tick = 0
        self.need_full = True
        self.event_seq = 0
        self.events: list = []
        self.roster: dict = {}
        self.scores = [0, 0]
        self.time_left = 0
        self.phase = C.PHASE_LIVE
        self.winner = S.NO_PLAYER
        # commands -----------------------------------------------------------
        self.cmd_tick = 0
        self.shot_seq = 0
        self.fire_records: list = []          # (seq, tick, view_ms, slot) awaiting acknowledgement
        self.render_ms = 0.0
        self.ping_ms = 0.0
        self._connect_started = 0.0
        self._last_send = 0.0
        self._last_request = 0.0
        self._request = b""
        self.last_packet = 0.0
        self.stats = {"snapshots": 0, "dropped_snapshots": 0, "bytes_in": 0, "bytes_out": 0, "full_requests": 0,
                      "events": 0, "malformed": 0}

    # ---- connection --------------------------------------------------------------------
    def connect(self, host: str, port: int, team: int = 255) -> None:
        """Begin the handshake (non blocking; poll ``state`` while calling ``update``)."""
        self.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.2)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError:
            pass
        self.sock.bind(("0.0.0.0", 0))
        try:
            self.server = (socket.gethostbyname(host), int(port))
        except OSError as error:
            self.state = FAILED
            self.fail_reason = "cannot resolve %s: %s" % (host, error)
            return
        raw_name = self.name.encode("utf-8")[:16]
        blob = W.encode_loadout(self.weapon_ids, self.levels, self.operator, self.prestige)
        blob += EQUIP_BLOB.pack(*self.equipment)
        self._request = make_packet(PKT_CONNECT_REQ, 0, CONNECT_REQ.pack(C.PROTOCOL_VERSION, self.nonce, team & 0xFF,
                                                                         len(raw_name)) + raw_name + blob)
        self.state = CONNECTING
        self.fail_reason = ""
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, name="ClientReceive", daemon=True)
        self.thread.start()
        if not self.conditioner.passive and not self.conditioner.is_alive():
            self.conditioner.start()
        self._connect_started = time.perf_counter()
        self._last_request = 0.0
        self._last_send = time.perf_counter()

    def _receive_loop(self) -> None:
        sock = self.sock
        while self.running:
            self._keep_alive()
            try:
                data, addr = sock.recvfrom(C.RECV_BUFFER)
            except socket.timeout:
                continue
            except OSError:
                if not self.running:
                    break
                continue
            self.inbox.append((data, addr, time.perf_counter()))

    def _keep_alive(self) -> None:
        """
        Runs on the receive thread.  While the main thread is busy (baking an
        arena on its first load can take several seconds) no commands leave
        this client; an empty frame once per second stops the server from
        timing the connection out.
        """
        if self.state != CONNECTED or self.sock is None or self.server is None:
            return
        now = time.perf_counter()
        if now - self._last_send < C.KEEPALIVE_INTERVAL:
            return
        self._last_send = now
        payload = FRAME_HEADER.pack(0 if self.need_full else self.last_snapshot_tick, self.event_seq, now_ms() or 1,
                                    self.cmd_tick & 0xFFFFFFFF, int(min(self.ping_ms, 65535)), 0, 0, *self.equipment)
        try:
            self.sock.sendto(make_packet(PKT_CLIENT_FRAME, self.token, payload), self.server)
            self.stats["keepalives"] = self.stats.get("keepalives", 0) + 1
        except OSError:
            pass

    def _send(self, datagram: bytes) -> None:
        if self.sock is None or self.server is None:
            return
        self._last_send = time.perf_counter()
        try:
            self.conditioner.send(self.sock, datagram, self.server)
            self.stats["bytes_out"] += len(datagram)
        except OSError:
            pass

    def disconnect(self) -> None:
        if self.state == CONNECTED:
            for _ in range(2):
                self._send(make_packet(PKT_DISCONNECT, self.token, b"\x00"))
        self.close()

    def _abort(self, token: int, reason: str) -> None:
        """Give the slot back to the server and stop with an explanation for the player."""
        for _ in range(2):
            self._send(make_packet(PKT_DISCONNECT, token, b"\x00"))
        self.state = FAILED
        self.fail_reason = reason

    def close(self) -> None:
        self.running = False
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)
        self.thread = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self.state != FAILED:
            self.state = DISCONNECTED

    # ---- per frame ---------------------------------------------------------------------------
    def update(self) -> list:
        """Process inbound datagrams; returns the game events decoded this call."""
        now = time.perf_counter()
        if self.state == CONNECTING:
            if now - self._connect_started > C.CONNECT_TIMEOUT:
                self.state = FAILED
                self.fail_reason = "no response from server"
            elif now - self._last_request >= C.CONNECT_RETRY_INTERVAL:
                self._send(self._request)
                self._last_request = now
        inbox = self.inbox
        while inbox:
            data, addr, received = inbox.popleft()
            if addr != self.server:
                continue
            self.stats["bytes_in"] += len(data)
            try:
                self._handle_packet(data, received)
            except (struct.error, IndexError, ValueError, KeyError):
                self.stats["malformed"] += 1
        if self.state == CONNECTED and now - self.last_packet > C.CONNECTION_TIMEOUT:
            self.state = FAILED
            self.fail_reason = "connection to server lost"
        events, self.events = self.events, []
        return events

    def _handle_packet(self, data: bytes, received: float) -> None:
        magic, packet_type, token = HEADER.unpack_from(data, 0)
        if magic != C.PROTOCOL_MAGIC:
            return
        offset = HEADER.size
        if packet_type == PKT_CONNECT_ACK:
            if self.state == CONNECTING:
                self._on_ack(data, offset, received)
            return
        if packet_type == PKT_CONNECT_DENY:
            nonce, reason = CONNECT_DENY.unpack_from(data, offset)
            if self.state == CONNECTING and nonce == self.nonce:
                self.state = FAILED
                self.fail_reason = DENY_REASONS.get(reason, "connection refused")
            return
        if self.state != CONNECTED or token != self.token:
            return
        self.last_packet = received
        if packet_type == PKT_SNAPSHOT:
            self._on_snapshot(data, offset, received)
        elif packet_type == PKT_ROSTER:
            self._on_roster(data, offset)
        elif packet_type == PKT_DISCONNECT:
            self.state = FAILED
            self.fail_reason = "kicked: timed out" if data[offset:offset + 1] == b"\x01" else "server closed"

    def _on_ack(self, data: bytes, offset: int, received: float) -> None:
        (nonce, pid, tick, _rate, map_index, mode, team, token, max_players, upgrades, score_limit,
         time_limit, map_crc) = CONNECT_ACK.unpack_from(data, offset)
        if nonce != self.nonce:
            return
        if not local_map_matches(map_index, map_crc):
            self._abort(token, CONTENT_MISMATCH)
            return
        self.pid, self.team, self.token = pid, team, token
        self.map_index, self.mode, self.max_players = map_index, mode, max_players
        self.upgrades_enabled = bool(upgrades)
        self.score_limit, self.time_limit = score_limit, time_limit
        self.loaded = maps.load_map(map_index)
        levels = self.levels if self.upgrades_enabled else np.zeros_like(self.levels)
        operator = self.operator if self.upgrades_enabled else np.zeros_like(self.operator)
        self.predictor = Predictor(self.loaded.world, pid, self.weapon_ids, levels, operator, self.prestige)
        self.cmd_tick = tick
        self.state = CONNECTED
        self.last_packet = received
        self.map_changed = True
        self.need_full = True
        self.last_snapshot_tick = 0
        self.event_seq = 0
        self.interp.clear()
        self.sync = TimeSync()

    def _on_snapshot(self, data: bytes, offset: int, received: float) -> None:
        (tick, baseline_tick, ack_cmd, ack_shot, echo_ms, hold_ms, flags, n_reliable,
         n_transient) = SNAP_HEADER.unpack_from(data, offset)
        offset += SNAP_HEADER.size
        if tick <= self.last_snapshot_tick:
            self.stats["dropped_snapshots"] += 1
            return
        score0, score1, seconds, phase, winner = MATCH_STATE.unpack_from(data, offset)
        offset += MATCH_STATE.size
        owner_offset = -1
        if flags & SNAP_HAS_OWNER:
            owner_offset = offset
            offset += PlayerStore.OWNER_SIZE
        baseline = None
        if baseline_tick:
            baseline = self.frames[baseline_tick % C.SNAPSHOT_RING]
            if baseline is None or baseline.tick != baseline_tick:
                self.need_full = True                # cannot decode this delta: ask for a full snapshot
                self.stats["full_requests"] += 1
                return
        frame, offset = decode_entities(data, offset, baseline, self.interp.capacity)
        frame.tick = tick
        self.frames[tick % C.SNAPSHOT_RING] = frame
        self.last_snapshot_tick = tick
        self.need_full = False
        self.scores = [score0, score1]
        self.time_left, self.phase, self.winner = seconds, phase, winner
        self.stats["snapshots"] += 1
        if n_reliable:
            first_seq = _U32.unpack_from(data, offset)[0]
            offset += 4
            for index in range(n_reliable):
                event, offset = decode_event(data, offset)
                if first_seq + index == self.event_seq + 1:
                    self.event_seq += 1
                    self._deliver(event)
        for _ in range(n_transient):
            event, offset = decode_event(data, offset)
            self._deliver(event)
        self.sync.add(tick * C.TICK_MS, received * 1000.0)
        if echo_ms:
            rtt = ((int(received * 1000.0) & 0xFFFFFFFF) - echo_ms - hold_ms) & 0xFFFFFFFF
            if rtt < 5000:
                self.ping_ms += (rtt - self.ping_ms) * 0.1 if self.ping_ms else rtt
        if self.fire_records:
            self.fire_records = [r for r in self.fire_records if seq_newer(r[0], ack_shot)]
        self.interp.push(frame, tick * C.TICK_MS)
        if owner_offset >= 0 and self.predictor is not None:
            self.predictor.reconcile(ack_cmd, data, owner_offset)

    def _deliver(self, event: tuple) -> None:
        if event[0] == S.EV_MATCH:
            map_index, map_crc = event[3], event[5]
            event = event[:5]
            if map_index != self.map_index or map_crc != self.loaded.data.checksum:
                if not local_map_matches(map_index, map_crc):
                    self._abort(self.token, CONTENT_MISMATCH)
                    return
                self.map_index = map_index
                self.loaded = maps.load_map(map_index)
                self.predictor.set_world(self.loaded.world)
                self.interp.clear()
                self.map_changed = True
        self.events.append(event)
        self.stats["events"] += 1

    def _on_roster(self, data: bytes, offset: int) -> None:
        count = data[offset]
        offset += 1
        roster = {}
        for _ in range(count):
            pid, team, flags, kills, deaths, score, ping, length = ROSTER_ENTRY.unpack_from(data, offset)
            offset += ROSTER_ENTRY.size
            name = data[offset: offset + length].decode("utf-8", "replace")
            offset += length
            roster[pid] = RosterEntry(pid, team, flags, kills, deaths, score, ping, name)
        self.roster = roster
        if self.pid in roster:
            self.team = roster[self.pid].team

    # ---- simulation tick -------------------------------------------------------------------------
    def send_command(self, buttons: int, yaw: float, pitch: float):
        """
        Sample one input command: predict it locally right now, then upload it
        together with the previous ``CMD_REDUNDANCY`` commands and every
        unacknowledged FIRE record.  Returns ``(fired, lethal_used, tactical_used)``.
        """
        if self.state != CONNECTED or self.predictor is None:
            return False, False, False
        pred = self.predictor
        self.cmd_tick += 1
        tick = self.cmd_tick
        yaw = float(np.float32(M.wrap_angle(yaw)))          # predict with exactly the bits the server will receive
        pitch = float(np.float32(np.clip(pitch, -C.PITCH_LIMIT, C.PITCH_LIMIT)))
        buttons = int(buttons) & 0xFFFF & ~C.BTN_FIRED
        fired, lethal, tactical = pred.predict(tick, buttons, yaw, pitch)
        if fired:
            pred.c_buttons[tick % pred.capacity] |= C.BTN_FIRED
            self.shot_seq = (self.shot_seq + 1) & 0xFFFF
            self.fire_records.append((self.shot_seq, tick, int(max(self.render_ms, 0.0)) & 0xFFFFFFFF,
                                      int(pred.store.weapons.slot[0])))
            del self.fire_records[:-16]
        first = max(tick - C.CMD_REDUNDANCY, pred.acked_tick + 1, tick - pred.capacity + 1)
        commands = []
        for t in range(first, tick + 1):
            i = t % pred.capacity
            if pred.c_tick[i] != t:
                commands = []
                continue
            commands.append((int(pred.c_buttons[i]), float(pred.c_yaw[i]), float(pred.c_pitch[i]),
                             quantize_velocity(pred.c_vel[i])))
        payload = FRAME_HEADER.pack(0 if self.need_full else self.last_snapshot_tick, self.event_seq, now_ms() or 1,
                                    tick, int(min(self.ping_ms, 65535)), len(commands), len(self.fire_records),
                                    *self.equipment)
        payload += encode_commands(commands)
        for record in self.fire_records:
            payload += FIRE_RECORD.pack(*record)
        self._send(make_packet(PKT_CLIENT_FRAME, self.token, payload))
        return fired, lethal, tactical

    def sample_entities(self) -> EntityView:
        """Remote entities at ``server_now - 100 ms`` (also fixes the lag-compensation timestamp)."""
        self.render_ms = self.sync.server_ms(time.perf_counter() * 1000.0) - C.INTERP_DELAY_MS
        return self.interp.sample(self.render_ms, self.view)

    def estimated_server_tick(self) -> float:
        return self.sync.server_ms(time.perf_counter() * 1000.0) / C.TICK_MS


def host_local_game(config: ServerConfig) -> GameServer:
    """Start a listen server inside this process (solo play and LAN hosting)."""
    server = GameServer(config)
    server.start()
    return server
