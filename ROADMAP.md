# NEXUS-3D-EXTREME – Roadmap

Stand: 17.11.2026, nach Sprint 3 und dem Code-Review. Geplant aus einer Code-Analyse (Knowledge-Graph mit graphify:
1472 Knoten, 3349 Kanten, 81 Module-Gruppen, keine Import-Zyklen) und Bot-Messungen.

## Bestandsaufnahme

**Stark:** deterministischer 60-Hz-Netcode mit Vorhersage, Rollback und Lag-Kompensation; 29 Waffen
in einem geprüften Balance-Modell (TTK 300–500 ms); 7 Ausrüstungsteile; 5 Arenen als Daten;
prozedurale Grafik und Sounds; Wirtschaft mit Sponsoren und Prestige; Bots in drei Stufen; über
120 automatische Tests.

**Lücken aus Spielersicht:**
- Kein Killcam und kein Zuschauermodus; nur LAN (Scorestreaks, Minimap und Death-Recap kamen in Sprint 2,
  UPLINK, Match-Report, Medaillen und Aufträge in Sprint 3).
- Keine frei belegbaren Tasten, kein Controller.
- Nur LAN, kein Spiel übers Internet.

**Technische Schulden (aus dem Graphen):**
- Zentrale Knoten: `NexusGame` (43 Verbindungen), `ProfileManager` (41), `GameWorld` (35).
- Sehr große Dateien: `engine/renderer.py` (2600 Zeilen), `engine/network_lan.py` (1900 Zeilen).
- Am lockersten zusammenhängend: Bewegung/Vorhersage und die Menü-Screens.
- Offene Punkte aus PROGRESS.md: Listen-Server im eigenen Prozess, Kartenübertragung beim
  Verbinden, Backen im Hintergrund, Messung auf echter Grafikhardware.

## Wie priorisiert wird

Nutzen für Spieler gegen Aufwand (S = bis ein Tag, M = wenige Tage, L = eine Woche oder mehr).
Alles, was Spielregeln ändert, läuft entweder durch die gemeinsame Simulation
(`simulate_commands`) oder als Server-Zustand über den Owner-State, damit Vorhersage und Server
nie auseinanderlaufen. Balance-Prüfungen (`validate_balance`, `validate_equipment`,
`validate_escalation`) bleiben leer. Nur eigene Namen und Inhalte.

## Sprint 1 – erledigt

| Thema | Ergebnis |
| --- | --- |
| Neuer Modus ESCALATION | Waffenleiter über alle 29 Waffen; jeder Waffen-Kill gibt die nächste Waffe, ein Kill mit der letzten gewinnt; Beil-Kills stehlen eine Stufe, Selbst-Kills kosten eine; Banner, Feed-Zeilen, Sounds, Scoreboard-Spalte, Beschreibung im Setup, `--mode esc` |
| Netcode | Der Owner-State überträgt jetzt das komplette Loadout (Waffen + Upgrade-Stufen); vom Server vergebene Waffen kommen ohne Extra-Nachricht in der Vorhersage an |
| Fehler behoben | Doppel-Kill mit einer Schrotsalve wurde der zweiten Waffe zugerechnet; Waffenwechsel zwischen zwei gleichen Waffen hätte Nachladen umgangen; Steuerungstext im Einstellungsmenü veraltet |
| Tests | 12 neue Tests (Regeln, Codec, Vorhersage im Live-Netz, Bot-Matches) |

## Sprint 2 – „Serien, die sich lohnen“ – erledigt

| Thema | Ergebnis |
| --- | --- |
| Scorestreaks | PULSE SCAN (3 Kills), AEGIS SHIELD (5), ORBITAL LANCE (7); einmal pro Leben verdient, warten über den Tod hinaus, Einsatz mit X; Server-Regeln, Anzeige über Owner-State und Snapshots; Bots nutzen sie |
| Minimap | Aus der Arena-Geometrie gerastert; eigene Position, Trupp, Schuss-Pings, Scan, Lance-Gefahrenzone |
| Death-Recap | Waffe, Entfernung, Trefferbilanz in beide Richtungen, Restleben des Gegners, Schaden von anderen |
| Tests | 13 neue Tests (140 insgesamt), gerenderte Szenen für Markierung, Einschlag, Schild und Recap geprüft |

## Sprint 3 – „Matchgefühl“ – erledigt

| Thema | Ergebnis |
| --- | --- |
| Modus UPLINK | Drei Zonen pro Arena, automatisch und fair aus den Kartendaten abgeleitet; Erobern, Neutralisieren, Umkämpft, Punkte pro gehaltener Zone; Bots spielen auf Ziele (Bot-Matches enden nach 2,5 bis 4 Minuten, ausgeglichen über 12 Testläufe) |
| Match-Report und Medaillen | 12 Medaillen mit Popups, Report mit MVP und persönlicher Bilanz neben dem Scoreboard |
| Aufträge | 3 tägliche und 1 wöchentlicher Auftrag nach UTC-Datum, Auszahlung sofort, im signierten Spielstand |
| Tests | 18 neue Tests (158 insgesamt), gerenderte Szenen für Erobern, Medaillen, Report und Menüs geprüft |

## Code-Review nach Sprint 3 – erledigt

| Thema | Ergebnis |
| --- | --- |
| Sicherheit und Speichern | Netz-Eingaben gehärtet (NaN, Tick-Sprünge, Fluten, Sitzungsübernahme, Markup), Spielstand robust gegen Wettläufe, Lesefehler und NaN |
| Spielregeln | 14 Funde behoben (u. a. Nachrücker erbt Granaten, Einfrieren ohne Spawns, Punkte in der Pause, Gleichstände, ESCALATION-Kit, Spawnschutz, Lanzenziel) |
| Client | 14 Funde behoben (Endstand für MVP/Report, Trefferquote, HUD über Menüs, 4:3, Pause zwischen Matches, Eingabe nach RESUME) |
| Tests | 185 insgesamt; jede Regression schlägt auf dem alten Code fehl; Szenen bei 16:9 und 4:3 gerendert und geprüft |

## Sprint 4 – Technik und Hosting

| Thema | Aufwand |
| --- | --- |
| Listen-Server in eigenen Prozess auslagern (Host-Framerate) | M–L |
| Kartendefinition beim Verbinden übertragen statt Clients abzuweisen | M |
| Erstes Backen einer Arena im Hintergrund-Thread | S–M |
| Tests automatisch bei jedem Push (GitHub Actions, headless) | S |
| `renderer.py` in ein Paket aufteilen (HUD, Post-FX, Effekt-Pools, Welt) und `NexusGame` in Teilsysteme (Eingabe, Effekte, Session) | M |

## Später / Ideenspeicher

- Killcam aus dem Snapshot-Verlauf (L), Zuschauermodus (M)
- Freie Tastenbelegung und Controller (M)
- Spieler kollidieren miteinander; echte Rampen statt Treppenstufen (je M, Netcode)
- Rückstoß serverseitig prüfen (M, Anti-Cheat)
- Waffen-Meisterschaft mit Tarnmustern aus den prozeduralen Texturen (M)
- Aufsätze (Visiere, Griffe, Magazine) – nur mit Erweiterung des Balance-Modells (L)
- Karteneditor im Spiel (L), Spiel übers Internet mit Relay (L)
- Windows-Build per PyInstaller (M), Messung auf echter Grafikhardware
