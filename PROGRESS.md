# NEXUS-3D-EXTREME: Arbeitsstand

Stand nach dem Refactoring "AAA REFACTORING & OPTIMIZATION DIRECTIVE". Diese Datei ist der
gespeicherte Fortschritt; `README.md` beschreibt das Spiel selbst.

## Erledigt und geprüft

| Bereich | Stand |
| --- | --- |
| `engine/config.py` | Neu: Waffen-/Upgrade-Tabellen, TTK-Band, Audio-Decken, Grafik-Presets, Frame-Pacing |
| Waffen | 29 Waffen in 7 Klassen (17 im letzten Schritt neu), alle im Balance-Modell; Sound, Tracer und Ego-Modell entstehen aus den Werten bzw. aus `WEAPON_LOOKS`; Armory nach Klassen geblättert; Bots nutzen das ganze Arsenal |
| Waffenbalance | TTK 333 bis 483 ms (Körpertreffer, 100 HP), maximal aufgerüstet 300 bis 450 ms; `weapons.validate_balance()` leer |
| Audio | Master-Clamp 0..1, Kategorie-Decken (Waffe 0.25, Treffer 0.20, Explosion 0.30), Abfall 1/d² |
| Karten | `content/maps_registry.py` (5 Arenen als Tupel-Daten), `engine/maps.py` als Injektor, JSON-Drop-in, Laufzeit-Registrierung, CRC im Handshake |
| Renderer | `engine/renderer.py` neu: Material-Texturarrays, Normal-Mapping, Sonnenschatten (statisch + dynamisch), gebackene Lampen + AO, Himmelspanorama, 3D-Skyline, Bloom, gebündelte Pools und HUD |
| Frame-Pacing | Vsync standardmäßig aus, Limiter 144 fps, dynamische Auflösung (Ziel 60 fps), automatischer Preset-Rückfall, Cursor-Picking nur in Menüs |
| Stabilität | Rückstoß-/Landefeder im Kamera-Rig implizit integriert (vorher bei <= 10 fps divergent, Bild wurde einfarbig); Client-Keep-alive während des Erst-Backens |
| Tests | 99 bestanden (`python -m pytest tests -q` und `python tests/run_all.py`), davon 2 gerenderte Läufe unter xvfb |

## Messwerte (Container ohne GPU, Mesa llvmpipe, 1280x720, 8 Spieler, Listen-Server im selben Prozess)

| Build | fps | Bemerkung |
| --- | --- | --- |
| vorher (einfarbige Würfel) | 25 | feste Auflösung |
| jetzt, LOW | 36 bis 38 | dynamische Auflösung bei 50 % |
| jetzt, HIGH | 28 | bis der Preset-Rückfall greift |

CPU-Anteil (bei 320x180 gemessen): Client-Logik ca. 4.5 bis 5 ms pro Frame, Server-Tick ca. 1.8 ms.
Auf echter Grafikhardware wurde nicht gemessen.

## Bewusste Abweichungen von der Vorgabe

* Die Simulation bleibt bei festen 60-Hz-Ticks (Vorhersage und Rollback brauchen identische
  Schritte); frame-rate-unabhängig ist die Darstellung durch Interpolation mit dem echten dt.
* Das TTK-Band gilt für Körpertreffer. Kopftreffer behalten den Faktor 3.0 aus der ersten
  Spezifikation und sind schneller (Kopf-Eintrag von `ZONE_MULTIPLIER` auf etwa 1.3 setzen, falls
  auch sie im Band liegen sollen).
* Es gibt keine fremden 3D-Modelle oder Fotos im Projekt: Materialien und Himmel werden erzeugt,
  eigene Bilder / OBJ-Dateien lassen sich über `content/textures`, `content/sky`,
  `content/models` einhängen.

## Offen / nächste Schritte

1. Listen-Server in einen eigenen Prozess auslagern (`engine/network_lan.py`), damit er dem
   Render-Thread keine Interpreter-Zeit nimmt.
2. Kartendefinition beim Verbinden vom Server übertragen, statt Clients ohne die Karte abzuweisen.
3. Erstes Backen einer Arena (bis ca. 6 s auf ULTRA) in einen Hintergrund-Thread verlegen.
4. Auf echter Hardware messen und die Presets danach justieren.

Nichts davon ist committet oder gepusht; der Klon von `Kai3770/Nexus` ist nur lesend eingebunden.
