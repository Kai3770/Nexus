"""
NEXUS-3D-EXTREME engine package.

Headless-safe modules (no window, no Ursina import):
    config, mathx, crypto, controller_3d (simulation half), weapons, maps,
    simulation, bots, network_lan, secure_save

Window-bound modules (import Ursina / Panda3D):
    renderer, audio, menus, game
"""
from . import config

__all__ = ["config"]
__version__ = config.GAME_VERSION
