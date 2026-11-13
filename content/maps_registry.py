"""
NEXUS-3D-EXTREME :: content/maps_registry.py
============================================
Every arena of the game as plain data.  Nothing in the engine knows a single
coordinate of a level: ``engine/maps.py`` reads this registry (and any
``content/maps/*.json`` file with the same structure) and derives the render
mesh, the collision world, spawn points and the bot navigation graph from it.

Adding a map
------------
Add one entry to ``MAPS`` (or drop a JSON file into ``content/maps/``).  A map
is a dictionary::

    "my_map": {
        "name": "MY MAP",
        "description": "One line shown in the match setup screen.",
        "environment": {...},          # optional, see ENVIRONMENT_DEFAULTS
        "mirror": True,                # stamp "half" twice, rotated 180 degrees, for the two teams
        "center": [ ...objects... ],   # placed once
        "half":   [ ...objects... ],   # team 0 side (x < 0); mirrored for team 1 when "mirror" is set
    }

Every object is one 8-tuple -- a transform plus what to put there::

    (x, y, z, scale_x, scale_y, scale_z, texture_type, spawn_team)

    x, z          centre of the object's footprint          y   its BOTTOM (objects stand on y)
    scale_*       full size in metres
    texture_type  what the object is (table below)
    spawn_team    -1 for scenery; 0 on a "spawn" marker (0 = the team that owns this half)

texture_type
------------
Surfaces (solid, textured box):
    concrete  floor  metal  rack  crate  hazard  grate  roof  brick  paint  glass  container
Prefabs (expand into several detailed parts, one collision box):
    server_rack   dark cabinet with status light strip        pillar     shaft with base and capital
    barrier       stepped concrete road barrier               railing    posts and rails
    container_unit shipping container with frame              lamp_warm / lamp_ice   pole, lamp head and light
    barrel / pylon  mesh props loaded from content/models/*.obj
Ramps (walkable wedge rising towards the named direction; scale_y is the rise):
    ramp_+x  ramp_-x  ramp_+z  ramp_-z
Decor without collision:
    glow_cyan  glow_warm  glow_red  glow_team      emissive strips
    paint_team                                     wall paint in the owning team's colour
Lights (scale_x is the radius in metres, the other scales are ignored):
    light_ice  light_warm  light_red  light_team
Markers:
    spawn        a spawn point for spawn_team (players face the arena centre)
"""

ENVIRONMENT_DEFAULTS = {
    "sky_top": (0.03, 0.05, 0.10),
    "sky_horizon": (0.20, 0.26, 0.36),
    "fog_color": (0.12, 0.15, 0.20),
    "fog_density": 0.012,
    "sun_dir": (0.35, -0.80, 0.48),      # direction the sunlight travels
    "sun_color": (1.00, 0.95, 0.86),
    "ambient": (0.30, 0.33, 0.40),
    "clouds": 0.45,                      # 0 clear .. 1 overcast
    "stars": 0.0,                        # 0 none .. 1 full night sky
    "skyline": 0.5,                      # height of the distant city silhouette (0 = none)
    "kill_y": -30.0,                     # falling below this height is fatal
}

MAP_ORDER = ("datacore", "skybridge", "crucible", "terminal", "atrium")

MAPS = {
    # ------------------------------------------------------------------ #
    #  DATACORE: three lanes through a server hall, raised centre console
    # ------------------------------------------------------------------ #
    "datacore": {
        "name": "DATACORE",
        "description": "Three lanes through a humming server hall. Hold the centre console.",
        "environment": {
            "sky_top": (0.02, 0.03, 0.07), "sky_horizon": (0.10, 0.16, 0.26), "fog_color": (0.07, 0.10, 0.15),
            "fog_density": 0.011, "sun_dir": (0.30, -0.85, 0.42), "sun_color": (0.85, 0.92, 1.00),
            "ambient": (0.30, 0.34, 0.42), "clouds": 0.25, "stars": 0.9, "skyline": 0.7,
        },
        "mirror": True,
        "center": [
            (0.0, -1.0, 0.0, 64.0, 1.0, 44.0, "floor", -1),
            (0.0, 0.0, 22.5, 66.0, 8.0, 1.0, "concrete", -1),
            (0.0, 0.0, -22.5, 66.0, 8.0, 1.0, "concrete", -1),
            (-32.5, 0.0, 0.0, 1.0, 8.0, 46.0, "concrete", -1),
            (32.5, 0.0, 0.0, 1.0, 8.0, 46.0, "concrete", -1),
            (0.0, 0.0, 0.0, 10.0, 1.2, 8.0, "metal", -1),              # centre console platform
            (0.0, 1.2, 0.0, 1.6, 1.8, 1.6, "server_rack", -1),
            (0.0, 7.2, 0.0, 0.6, 0.5, 44.0, "metal", -1),              # roof trusses
            (-16.0, 7.2, 0.0, 0.6, 0.5, 44.0, "metal", -1),
            (16.0, 7.2, 0.0, 0.6, 0.5, 44.0, "metal", -1),
            (0.0, 5.0, 0.0, 16.0, 0.0, 0.0, "light_ice", -1),
        ],
        "half": [
            (-7.0, 0.0, 0.0, 4.0, 1.2, 4.0, "ramp_+x", -1),             # up to the console
            (-19.0, 0.0, 8.0, 6.0, 2.8, 1.2, "server_rack", -1),        # lane dividers with doorways
            (-10.5, 0.0, 8.0, 5.0, 2.8, 1.2, "server_rack", -1),
            (-18.0, 0.0, -8.0, 8.0, 2.8, 1.2, "server_rack", -1),
            (-9.5, 0.0, -8.0, 3.0, 2.8, 1.2, "server_rack", -1),
            (-24.7, 0.0, 0.0, 0.6, 3.0, 12.0, "concrete", -1),          # spawn screen
            (-24.36, 1.4, 0.0, 0.08, 0.3, 12.0, "glow_team", -1),
            (-25.04, 1.4, 0.0, 0.08, 0.3, 12.0, "glow_team", -1),
            (-14.0, 0.0, 0.0, 1.4, 1.0, 1.4, "crate", -1),
            (-18.0, 0.0, 3.4, 1.2, 2.4, 2.4, "server_rack", -1),
            (-18.0, 0.0, -3.4, 1.2, 2.4, 2.4, "server_rack", -1),
            (-20.0, 0.0, 13.5, 1.4, 1.0, 1.4, "crate", -1),
            (-10.0, 0.0, 16.0, 2.4, 1.0, 1.2, "crate", -1),
            (-5.0, 0.0, 11.5, 1.2, 2.2, 1.2, "server_rack", -1),
            (-18.0, 0.0, -15.0, 1.4, 1.0, 1.4, "crate", -1),
            (-9.0, 0.0, -13.0, 1.4, 2.2, 2.6, "server_rack", -1),
            (-3.0, 0.0, -17.0, 2.4, 1.0, 1.2, "crate", -1),
            (-20.0, 0.0, 19.5, 8.0, 3.0, 3.0, "ramp_+x", -1),           # overlook catwalk
            (-9.0, 2.8, 19.5, 14.0, 0.2, 5.0, "grate", -1),
            (-9.0, 3.0, 17.1, 14.0, 0.9, 0.2, "railing", -1),
            (-15.6, 0.0, 17.4, 0.4, 2.8, 0.4, "metal", -1),
            (-9.0, 0.0, 17.4, 0.4, 2.8, 0.4, "metal", -1),
            (-2.4, 0.0, 17.4, 0.4, 2.8, 0.4, "metal", -1),
            (-31.3, 0.0, 16.0, 0.8, 0.9, 0.8, "barrel", -1),
            (-31.3, 0.0, -19.0, 0.8, 0.9, 0.8, "barrel", -1),
            (-29.5, 0.0, -9.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-29.5, 0.0, -4.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-29.5, 0.0, 0.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-29.5, 0.0, 4.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-29.5, 0.0, 9.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-27.0, 0.0, 13.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-27.0, 0.0, -13.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-27.0, 0.0, -16.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-27.0, 4.0, 0.0, 13.0, 0.0, 0.0, "light_team", -1),
            (-14.0, 3.5, 12.0, 11.0, 0.0, 0.0, "light_ice", -1),
            (-14.0, 3.5, -12.0, 11.0, 0.0, 0.0, "light_ice", -1),
        ],
    },

    # ------------------------------------------------------------------ #
    #  SKYBRIDGE: two rooftops, a bridge and lower service decks
    # ------------------------------------------------------------------ #
    "skybridge": {
        "name": "SKYBRIDGE",
        "description": "Rooftop duel. Cross the bridge or flank along the service decks.",
        "environment": {
            "sky_top": (0.05, 0.09, 0.22), "sky_horizon": (0.82, 0.48, 0.30), "fog_color": (0.40, 0.30, 0.28),
            "fog_density": 0.006, "sun_dir": (0.55, -0.45, 0.70), "sun_color": (1.00, 0.80, 0.60),
            "ambient": (0.36, 0.34, 0.40), "clouds": 0.6, "stars": 0.15, "skyline": 1.0, "kill_y": -9.0,
        },
        "mirror": True,
        "center": [
            (0.0, -1.0, 0.0, 12.0, 1.0, 14.0, "roof", -1),             # centre island
            (0.0, 0.0, 0.0, 2.0, 5.0, 2.0, "metal", -1),               # antenna mast
            (0.0, 3.8, 0.0, 2.1, 0.2, 2.1, "glow_red", -1),
            (0.0, 4.5, 0.0, 15.0, 0.0, 0.0, "light_warm", -1),
        ],
        "half": [
            (-23.0, -1.0, 0.0, 22.0, 1.0, 20.0, "roof", -1),            # rooftop
            (-33.0, -1.0, 12.5, 2.0, 1.0, 5.0, "roof", -1),             # ramp landings
            (-33.0, -1.0, -12.5, 2.0, 1.0, 5.0, "roof", -1),
            (-34.2, 0.0, 0.0, 0.4, 1.0, 30.0, "concrete", -1),          # parapets
            (-22.0, 0.0, 10.2, 20.0, 1.0, 0.4, "concrete", -1),
            (-22.0, 0.0, -10.2, 20.0, 1.0, 0.4, "concrete", -1),
            (-12.2, 0.0, 6.1, 0.4, 1.0, 7.8, "concrete", -1),
            (-12.2, 0.0, -6.1, 0.4, 1.0, 7.8, "concrete", -1),
            (-33.99, 0.45, 0.0, 0.06, 0.15, 30.0, "glow_team", -1),
            (-9.0, -0.4, 0.0, 6.0, 0.4, 4.0, "grate", -1),              # bridge to the island
            (-9.0, 0.0, 2.1, 6.0, 0.9, 0.2, "railing", -1),
            (-9.0, 0.0, -2.1, 6.0, 0.9, 0.2, "railing", -1),
            (-3.0, 0.0, 3.2, 3.0, 1.0, 0.4, "concrete", -1),            # island cover
            (-4.0, 0.0, -4.0, 1.4, 1.0, 1.4, "crate", -1),
            (-13.0, -3.0, 14.0, 26.0, 0.5, 6.0, "grate", -1),           # service decks 2.5 m below
            (-13.0, -3.0, -14.0, 26.0, 0.5, 6.0, "grate", -1),
            (-13.0, -2.5, 17.15, 26.0, 0.9, 0.3, "railing", -1),
            (-13.0, -2.5, -17.15, 26.0, 0.9, 0.3, "railing", -1),
            (-29.0, -2.5, 13.0, 6.0, 2.5, 4.0, "ramp_-x", -1),          # rooftop <-> deck
            (-29.0, -2.5, -13.0, 6.0, 2.5, 4.0, "ramp_-x", -1),
            (-14.0, -2.5, 14.0, 1.4, 1.0, 1.4, "crate", -1),
            (-8.0, -2.5, -14.0, 2.6, 1.0, 1.2, "crate", -1),
            (0.0, -2.5, 9.0, 4.0, 2.5, 4.0, "ramp_-z", -1),             # deck <-> island
            (-22.0, 0.0, 0.0, 4.0, 3.0, 4.0, "brick", -1),              # stairwell hut
            (-19.99, 1.55, 0.0, 0.06, 0.25, 2.0, "glow_team", -1),
            (-27.0, 0.0, 6.0, 2.4, 1.0, 1.4, "metal", -1),              # air handlers
            (-27.0, 0.0, -6.0, 2.4, 1.0, 1.4, "metal", -1),
            (-16.0, 0.0, 6.5, 1.4, 1.0, 1.4, "crate", -1),
            (-16.0, 0.0, -6.5, 1.4, 1.0, 1.4, "crate", -1),
            (-14.5, 0.0, 0.0, 0.6, 1.0, 2.4, "barrier", -1),
            (-33.2, 0.0, 9.0, 0.3, 4.2, 0.3, "lamp_warm", -1),
            (-33.2, 0.0, -9.0, 0.3, 4.2, 0.3, "lamp_warm", -1),
            (-31.5, 0.0, -7.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-31.5, 0.0, -4.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-31.5, 0.0, -1.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-31.5, 0.0, 1.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-31.5, 0.0, 4.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-31.5, 0.0, 7.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-29.0, 0.0, 8.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-29.0, 0.0, -8.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-28.0, 3.5, 0.0, 14.0, 0.0, 0.0, "light_team", -1),
            (-12.0, -0.5, 14.0, 10.0, 0.0, 0.0, "light_warm", -1),
            (-12.0, -0.5, -14.0, 10.0, 0.0, 0.0, "light_warm", -1),
        ],
    },

    # ------------------------------------------------------------------ #
    #  CRUCIBLE: compact arena around a raised core (free-for-all)
    # ------------------------------------------------------------------ #
    "crucible": {
        "name": "CRUCIBLE",
        "description": "A tight kill box around a raised core. Nowhere to rest.",
        "environment": {
            "sky_top": (0.08, 0.03, 0.03), "sky_horizon": (0.42, 0.18, 0.10), "fog_color": (0.18, 0.09, 0.07),
            "fog_density": 0.014, "sun_dir": (-0.40, -0.75, 0.52), "sun_color": (1.00, 0.72, 0.50),
            "ambient": (0.34, 0.28, 0.28), "clouds": 0.7, "stars": 0.0, "skyline": 0.4,
        },
        "mirror": True,
        "center": [
            (0.0, -1.0, 0.0, 36.0, 1.0, 36.0, "floor", -1),
            (0.0, 0.0, 18.5, 38.0, 7.0, 1.0, "brick", -1),
            (0.0, 0.0, -18.5, 38.0, 7.0, 1.0, "brick", -1),
            (-18.5, 0.0, 0.0, 1.0, 7.0, 38.0, "brick", -1),
            (18.5, 0.0, 0.0, 1.0, 7.0, 38.0, "brick", -1),
            (0.0, 0.0, 0.0, 6.0, 2.0, 6.0, "metal", -1),                # the core
            (0.0, 2.0, 0.0, 1.2, 2.2, 1.2, "server_rack", -1),
            (0.0, 5.5, 0.0, 15.0, 0.0, 0.0, "light_warm", -1),
        ],
        "half": [
            (-6.0, 0.0, 0.0, 6.0, 2.0, 3.0, "ramp_+x", -1),
            (-11.0, 0.0, 9.3, 4.0, 2.6, 0.6, "concrete", -1),           # corner bunkers
            (-12.7, 0.0, 11.3, 0.6, 2.6, 3.4, "concrete", -1),
            (-11.0, 0.0, -9.3, 4.0, 2.6, 0.6, "concrete", -1),
            (-12.7, 0.0, -11.3, 0.6, 2.6, 3.4, "concrete", -1),
            (-11.0, 1.2, 8.97, 4.0, 0.2, 0.06, "glow_team", -1),
            (-11.0, 1.2, -8.97, 4.0, 0.2, 0.06, "glow_team", -1),
            (-7.0, 0.0, 7.0, 1.4, 1.0, 1.4, "crate", -1),
            (-7.0, 0.0, -7.0, 1.4, 1.0, 1.4, "crate", -1),
            (0.0, 0.0, 10.5, 2.0, 4.0, 2.0, "pillar", -1),
            (-14.0, 0.0, 0.0, 0.6, 1.0, 4.0, "barrier", -1),
            (-4.5, 0.0, 14.5, 2.6, 1.0, 1.2, "crate", -1),
            (-16.0, 0.0, 0.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-16.0, 0.0, 15.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-16.0, 0.0, -15.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-10.5, 0.0, 16.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-10.5, 0.0, -16.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-16.0, 0.0, 7.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-12.0, 3.5, 11.0, 11.0, 0.0, 0.0, "light_team", -1),
            (-12.0, 3.5, -11.0, 11.0, 0.0, 0.0, "light_team", -1),
        ],
    },

    # ------------------------------------------------------------------ #
    #  TERMINAL: container yard with stacked steel and a sniper perch
    # ------------------------------------------------------------------ #
    "terminal": {
        "name": "TERMINAL",
        "description": "Container yard at dusk. Fight through the stacks or take the high boxes.",
        "environment": {
            "sky_top": (0.10, 0.16, 0.30), "sky_horizon": (0.74, 0.62, 0.52), "fog_color": (0.46, 0.44, 0.44),
            "fog_density": 0.007, "sun_dir": (-0.50, -0.62, -0.60), "sun_color": (1.00, 0.90, 0.74),
            "ambient": (0.40, 0.42, 0.46), "clouds": 0.5, "stars": 0.0, "skyline": 0.8,
        },
        "mirror": True,
        "center": [
            (0.0, -1.0, 0.0, 60.0, 1.0, 40.0, "roof", -1),             # asphalt yard
            (0.0, 0.0, 20.5, 62.0, 6.0, 1.0, "concrete", -1),
            (0.0, 0.0, -20.5, 62.0, 6.0, 1.0, "concrete", -1),
            (-30.5, 0.0, 0.0, 1.0, 6.0, 42.0, "concrete", -1),
            (30.5, 0.0, 0.0, 1.0, 6.0, 42.0, "concrete", -1),
            (0.0, 0.0, 6.5, 6.0, 2.6, 2.5, "container_unit", -1),      # centre pair
            (0.0, 0.0, -6.5, 6.0, 2.6, 2.5, "container_unit", -1),
            (0.0, 0.0, 0.0, 2.4, 1.0, 1.2, "crate", -1),
            (0.0, 6.0, 0.0, 18.0, 0.0, 0.0, "light_warm", -1),
        ],
        "half": [
            (-18.0, 0.0, 10.0, 6.0, 2.6, 2.5, "container_unit", -1),   # perch container
            (-24.0, 0.0, 10.0, 6.0, 2.6, 2.5, "ramp_+x", -1),          # ramp up to it
            (-18.0, 2.6, 9.0, 1.2, 1.0, 0.5, "crate", -1),
            (-18.0, 0.0, -10.0, 6.0, 2.6, 2.5, "container_unit", -1),
            (-10.0, 0.0, 1.5, 2.5, 2.6, 6.0, "container_unit", -1),    # mid blocker
            (-9.0, 0.0, 15.0, 2.5, 5.2, 6.0, "container_unit", -1),    # double stack
            (-6.0, 0.0, -15.0, 6.0, 2.6, 2.5, "container_unit", -1),
            (-14.0, 0.0, -3.5, 1.4, 1.0, 1.4, "crate", -1),
            (-21.0, 0.0, 0.0, 0.6, 1.0, 5.0, "barrier", -1),
            (-13.0, 0.0, 6.0, 0.8, 0.9, 0.8, "barrel", -1),
            (-13.9, 0.0, 6.4, 0.8, 0.9, 0.8, "barrel", -1),
            (-4.5, 0.0, -9.5, 0.8, 0.9, 0.8, "barrel", -1),
            (-23.5, 0.0, -16.0, 2.4, 1.0, 1.2, "crate", -1),
            (-25.0, 0.0, 17.5, 0.5, 0.7, 0.5, "pylon", -1),
            (-28.6, 0.0, 12.0, 0.3, 5.0, 0.3, "lamp_warm", -1),
            (-28.6, 0.0, -12.0, 0.3, 5.0, 0.3, "lamp_warm", -1),
            (-29.92, 2.2, 0.0, 0.08, 0.4, 16.0, "paint_team", -1),
            (-27.5, 0.0, -8.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-27.5, 0.0, -4.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-27.5, 0.0, 0.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-27.5, 0.0, 4.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-25.5, 0.0, -13.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-25.5, 0.0, 15.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-25.0, 0.0, 6.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-25.0, 0.0, -17.5, 1.0, 1.0, 1.0, "spawn", 0),
            (-26.0, 4.5, 0.0, 13.0, 0.0, 0.0, "light_team", -1),
        ],
    },

    # ------------------------------------------------------------------ #
    #  ATRIUM: two storey lobby, mezzanines joined by a bridge
    # ------------------------------------------------------------------ #
    "atrium": {
        "name": "ATRIUM",
        "description": "Corporate lobby on two floors. Own the mezzanine bridge.",
        "environment": {
            "sky_top": (0.20, 0.42, 0.78), "sky_horizon": (0.72, 0.84, 0.95), "fog_color": (0.62, 0.70, 0.78),
            "fog_density": 0.006, "sun_dir": (0.42, -0.78, -0.46), "sun_color": (1.00, 0.97, 0.90),
            "ambient": (0.46, 0.50, 0.56), "clouds": 0.35, "stars": 0.0, "skyline": 0.9,
        },
        "mirror": True,
        "center": [
            (0.0, -1.0, 0.0, 48.0, 1.0, 36.0, "floor", -1),
            (0.0, 0.0, 18.5, 50.0, 7.5, 1.0, "glass", -1),
            (0.0, 0.0, -18.5, 50.0, 7.5, 1.0, "glass", -1),
            (-24.5, 0.0, 0.0, 1.0, 7.5, 38.0, "concrete", -1),
            (24.5, 0.0, 0.0, 1.0, 7.5, 38.0, "concrete", -1),
            (0.0, 3.0, 0.0, 4.0, 0.2, 24.0, "grate", -1),               # mezzanine bridge
            (-2.1, 3.2, 0.0, 0.2, 0.9, 24.0, "railing", -1),
            (2.1, 3.2, 0.0, 0.2, 0.9, 24.0, "railing", -1),
            (0.0, 0.0, 0.0, 5.0, 1.1, 1.4, "barrier", -1),              # reception desk
            (0.0, 6.2, 0.0, 20.0, 0.0, 0.0, "light_ice", -1),
        ],
        "half": [
            (-12.0, 3.0, 15.0, 24.0, 0.2, 6.0, "floor", -1),            # north mezzanine (west half)
            (-12.0, 3.0, -15.0, 24.0, 0.2, 6.0, "floor", -1),           # south mezzanine (west half)
            (-16.0, 0.0, 10.5, 8.0, 3.2, 3.0, "ramp_+x", -1),           # stairs to the north mezzanine
            (-10.5, 3.0, 10.5, 3.0, 0.2, 3.0, "floor", -1),             # stair landing
            (-18.0, 3.2, 11.9, 12.0, 0.9, 0.2, "railing", -1),          # mezzanine rails
            (-5.5, 3.2, 11.9, 7.0, 0.9, 0.2, "railing", -1),
            (-13.0, 3.2, -11.9, 22.0, 0.9, 0.2, "railing", -1),
            (-18.0, 0.0, 12.5, 0.8, 3.0, 0.8, "pillar", -1),
            (-6.0, 0.0, 12.5, 0.8, 3.0, 0.8, "pillar", -1),
            (-18.0, 0.0, -12.5, 0.8, 3.0, 0.8, "pillar", -1),
            (-6.0, 0.0, -12.5, 0.8, 3.0, 0.8, "pillar", -1),
            (-10.0, 0.0, 3.0, 2.4, 1.0, 1.2, "brick", -1),              # planters
            (-14.0, 0.0, -5.0, 1.2, 1.0, 2.4, "brick", -1),
            (-6.0, 0.0, -8.0, 1.4, 1.0, 1.4, "crate", -1),
            (-17.0, 3.2, 15.5, 1.4, 1.0, 1.4, "crate", -1),
            (-8.0, 3.2, -15.5, 2.4, 1.0, 1.2, "crate", -1),
            (-23.92, 2.0, 0.0, 0.08, 0.5, 14.0, "paint_team", -1),
            (-23.0, 0.0, 7.0, 0.8, 0.9, 0.8, "barrel", -1),
            (-21.5, 0.0, -8.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-21.5, 0.0, -4.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-21.5, 0.0, 0.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-21.5, 0.0, 4.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-19.5, 0.0, -2.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-19.5, 0.0, 2.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-21.5, 3.2, -15.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-21.5, 3.2, 15.0, 1.0, 1.0, 1.0, "spawn", 0),
            (-20.0, 4.8, 0.0, 12.0, 0.0, 0.0, "light_team", -1),
            (-12.0, 2.2, 15.0, 9.0, 0.0, 0.0, "light_warm", -1),
            (-12.0, 2.2, -15.0, 9.0, 0.0, 0.0, "light_warm", -1),
        ],
    },
}
