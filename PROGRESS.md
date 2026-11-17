# NEXUS-3D-EXTREME: Arbeitsstand

Stand nach dem Refactoring "AAA REFACTORING & OPTIMIZATION DIRECTIVE". Diese Datei ist der
gespeicherte Fortschritt; `README.md` beschreibt das Spiel selbst.

## Erledigt und geprüft

| Bereich | Stand |
| --- | --- |
| `engine/config.py` | Neu: Waffen-/Upgrade-Tabellen, TTK-Band, Audio-Decken, Grafik-Presets, Frame-Pacing |
| Waffen | 29 Waffen in 7 Klassen (17 im letzten Schritt neu), alle im Balance-Modell; Sound, Tracer und Ego-Modell entstehen aus den Werten bzw. aus `WEAPON_LOOKS`; Armory nach Klassen geblättert; Bots nutzen das ganze Arsenal |
| Ausrüstung | Letal [G]: PULSE GRENADE, RAZOR HATCHET (Tomahawk), CLING CHARGE (Haftladung), THERMITE FLASK (Brand); Taktisch [F]: NANO STIM, SHOCK GRENADE (Betäubung), SMOKE CANISTER; Wahl im Menü EQUIPMENT, im Match über Pause (gilt ab nächstem Respawn); `equipment.validate_equipment()` leer |
| Waffenbalance | TTK 333 bis 483 ms (Körpertreffer, 100 HP), maximal aufgerüstet 300 bis 450 ms; `weapons.validate_balance()` leer |
| Audio | Master-Clamp 0..1, Kategorie-Decken (Waffe 0.25, Treffer 0.20, Explosion 0.30), Abfall 1/d² |
| Karten | `content/maps_registry.py` (5 Arenen als Tupel-Daten), `engine/maps.py` als Injektor, JSON-Drop-in, Laufzeit-Registrierung, CRC im Handshake |
| Renderer | `engine/renderer.py` neu: Material-Texturarrays, Normal-Mapping, Sonnenschatten (statisch + dynamisch), gebackene Lampen + AO, Himmelspanorama, 3D-Skyline, Bloom, gebündelte Pools und HUD |
| Frame-Pacing | Vsync standardmäßig aus, Limiter 144 fps, dynamische Auflösung (Ziel 60 fps), automatischer Preset-Rückfall, Cursor-Picking nur in Menüs |
| Stabilität | Rückstoß-/Landefeder im Kamera-Rig implizit integriert (vorher bei <= 10 fps divergent, Bild wurde einfarbig); Client-Keep-alive während des Erst-Backens |
| Spielmodi | TEAM DEATHMATCH, FREE FOR ALL und neu ESCALATION: Waffenleiter über alle 29 Waffen (`config.ESCALATION_LADDER`), jeder Waffen-Kill gibt die nächste Waffe, ein Kill mit der letzten gewinnt, Beil-Kills stehlen eine Stufe, Selbst-Kills kosten eine; `weapons.validate_escalation()` leer |
| Scorestreaks | PULSE SCAN (3 Kills, Gegner auf der Minimap), AEGIS SHIELD (5 Kills, 50er Überschild), ORBITAL LANCE (7 Kills, Schlag mit 2 s Vorwarnung); einmal pro Leben verdient, bleiben über den Tod hinaus, Einsatz mit X; nicht in ESCALATION; Bots nutzen sie; `streaks.validate_streaks()` leer |
| Minimap und Death-Recap | Minimap aus der Arena-Geometrie gerastert (eigene Position, Trupp, Schuss-Pings, Scan, Lance-Gefahrenzone); Recap beim Tod: Waffe, Entfernung, Trefferbilanz, Restleben des Gegners |
| UPLINK | Zonen-Modus mit drei Zonen pro Arena, automatisch und symmetrisch aus den Kartendaten abgeleitet; Erobern (6/4/3 s), Neutralisieren, Umkämpft, Punkte pro gehaltener Zone, Grenze 150; Bots spielen auf Ziele und verteilen sich spiegelbildlich; HUD-Zonenleiste, Markierungen mit Entfernung, Erobern-Balken, Minimap |
| Match-Report und Medaillen | 12 Medaillen mit Popups (+50 XP), Report mit MVP, Bilanz, Trefferquote, Credits/XP und Medaillen neben dem Scoreboard |
| Aufträge | 3 tägliche und 1 wöchentlicher Auftrag aus 10 Arten nach UTC-Datum, Fortschritt über die Profil-Statistik, Auszahlung 2.500 / 15.000 Credits, im signierten Spielstand; Menü CONTRACTS |
| Netcode | Owner-State überträgt das komplette Loadout (Waffen + Upgrade-Stufen) und den Streak-Zustand, Snapshots den Schild und den Zonenzustand; Treffer-Ereignis mit Kennung "Kugel", Kill-Ereignis mit Kennung "erster Kill", Roster mit Spielphase und sofort beim Phasenwechsel; Snapshots nennen Arena und Prüfsumme; Owner-State in voller Genauigkeit (float64); Protokoll-Version 16 |
| Code-Review | Rollenbasierte Prüfung (Sicherheit/Persistenz durch einen eigenen Prüfer, dazu Fuzzing aller Paketpfade, Dauertest aller Modi mit strenger Numerik, Wiederholung zeitkritischer Tests); behoben: Server-Absturz durch NaN-Blickwinkel, Tick-Manipulation (Dauerfeuer/Abklingzeiten), unbegrenzte FIRE-Einträge und Inbox, Übernahme laufender Sitzungen, Markup in Namen, Zeitlimit 0, Wettlauf beim Speichern, Backup bei fehlender Datei, Lesefehler als Manipulation, NaN im Spielstand, UPLINK-Zonen ohne exakten Nullpunkt; Host-Schalter UPGRADES |
| Code-Review Regeln und Client | Zwei weitere Prüfer (Spielregeln, Client/UI), 28 Funde nachgestellt und behoben: Granaten/Feuer/Lanzen und Sturz-Anspruch gehen mit einem Spieler, der das Match verlässt (sonst bekam ein Nachrücker im selben Slot die Kills); Karte ohne Spawns für einen Trupp friert nicht mehr ein; zwischen zwei Matches zählt nichts (Stürze, Rückstoß); Sturz zählt nur für feindlichen Schaden; Kills nach dem eigenen Tod verlängern keine Serie; Kill-Bonus nie negativ; Gleichstand in FFA/ESCALATION ist ein Unentschieden; ESCALATION ohne Operator-Upgrades (Server und Vorhersage); Beil-Kill auf der letzten Stufe gewinnt nicht; Ausrüstung und Belohnungen beenden den Spawnschutz; Lanze trifft das anvisierte Ziel; 32 Feuer-/Rauchzonen; Trefferquote nur aus Kugeln; FIRST STRIKE vom Server; MVP und Report aus dem Endstand; Ausrüstungs-Kills (Beil, Feuer) zählen für DEMOLITION und Aufträge; HUD nicht über Menüs und ohne Reste aus dem letzten Match; Klick auf RESUME schießt nicht; kein doppelter Respawn-Ton; ESCALATION-Intro bleibt stehen; Pause zwischen Matches mit Ergebnis im Report, Countdown und Belohnungen unter den Tafeln; 4:3 und 5:4 (Menüs skaliert, Killfeed unter dem Kopf, Hinweis umbrochen), Fenstergröße fest; nur eine Meldung gleichzeitig; Markup und kaputte Dateien bei eigenen Karten. Haftladung: nur volle Blast Plating plus AEGIS überlebt (Doku korrigiert) |
| Code-Review Netcode | Netcode-Prüfer, 5 Funde nachgestellt und behoben: nach langem Hänger des Clients blieb der Ereigniskanal für immer stehen (keine Kills, Spawns, Kartenwechsel mehr; jetzt überspringt der Client die verlorenen Ereignisse, Snapshots nennen die Arena); Lanzen-Zielerfassung ohne Lag-Kompensation; Owner-State nur float32 (Replay nicht bitgenau); Neue Spieler sahen bestehendes Feuer, Rauch und Lanzen nicht; Blickzeit der Schüsse lief nach 49,7 Tagen Serverlaufzeit über. Eigene Gegenprüfung: Owner-State vollständig, 12 Live-Sitzungen mit 40 ms, Jitter und 2 % Verlust ohne eine Korrektur, Fuzzer ohne Fund |
| Tests | 192 bestanden (`python -m pytest tests -q` und `python tests/run_all.py`), davon 2 gerenderte Läufe unter xvfb; jede Review-Regression schlägt auf dem alten Code fehl |

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

Die priorisierte Planung (Sprints 2 bis 4 und Ideenspeicher) steht in `ROADMAP.md`. Technisch offen:

1. Listen-Server in einen eigenen Prozess auslagern (`engine/network_lan.py`), damit er dem
   Render-Thread keine Interpreter-Zeit nimmt.
2. Kartendefinition beim Verbinden vom Server übertragen, statt Clients ohne die Karte abzuweisen.
3. Erstes Backen einer Arena (bis ca. 6 s auf ULTRA) in einen Hintergrund-Thread verlegen.
4. Auf echter Hardware messen und die Presets danach justieren.

Alle Stände sind auf dem Branch `claude/aaa-refactor` von `Kai3770/Nexus` committet und gepusht.
