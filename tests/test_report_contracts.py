"""Sprint 3: medals, the match report and daily / weekly contracts."""
import tempfile

import helpers  # noqa: F401

from engine import config as C
from engine import secure_save as SS
from engine.report import MEDAL_KEYS, MEDAL_NAMES, MatchReport

ME, OTHER, THIRD = 0, 1, 2


def test_the_medal_table_is_complete():
    assert len(MEDAL_KEYS) == len(set(MEDAL_KEYS)) == len(C.MEDAL_TABLE)
    assert all(MEDAL_NAMES[key] for key in MEDAL_KEYS)


def test_multi_eliminations_first_strike_and_streaks():
    report = MatchReport()
    assert report.kill(ME, OTHER, ME, 10.0) == ["first_strike"]
    assert report.kill(ME, THIRD, ME, 12.0) == ["double"]
    assert report.kill(ME, OTHER, ME, 15.5) == ["triple"]
    assert report.kill(ME, THIRD, ME, 30.0) == []                          # too late for a chain
    assert report.kill(ME, OTHER, ME, 40.0) == ["iron_will"]
    assert report.kill(OTHER, THIRD, ME, 41.0) == []                       # somebody else's elimination
    report.kill(OTHER, ME, ME, 42.0)                                        # we die: the streak resets
    assert report.streak == 0 and report.deaths == 1 and report.best_streak == 5
    assert report.kill(ME, OTHER, ME, 50.0) == ["payback"]                  # the one who got us
    for t in range(9):
        report.kill(ME, THIRD, ME, 100.0 + 10.0 * t)
    assert report.medals.count("overdrive") == 1


def test_first_strike_only_for_the_very_first_elimination():
    report = MatchReport()
    report.kill(OTHER, THIRD, ME, 1.0)
    assert report.kill(ME, OTHER, ME, 2.0) == []
    report = MatchReport()
    report.kill(ME, ME, ME, 1.0, suicide=True)                              # a fall is no elimination
    assert report.kill(ME, OTHER, ME, 2.0) == ["first_strike"]


def test_equipment_rewards_headshots_and_captures():
    report = MatchReport()
    for t in range(3):
        report.kill(ME, OTHER, ME, 10.0 * t, explosive=True)
    assert "demolition" in report.medals and report.equipment_kills == 3
    assert report.kill(ME, THIRD, ME, 100.0, explosive=True, reward=True) == ["starfall"]
    assert report.equipment_kills == 3 and report.reward_kills == 1
    for _ in range(4):
        assert report.hit(30.0, headshot=True) == []
    assert report.hit(30.0, headshot=True) == ["headhunter"]
    assert report.capture() == [] and report.capture() == [] and report.capture() == ["uplinker"]


def test_finish_awards_sharpshooter_and_mvp_and_the_summary_reads_well():
    report = MatchReport()
    for _ in range(40):
        report.shot()
    for _ in range(24):
        report.hit(25.0)
    report.kill(ME, OTHER, ME, 1.0)
    report.kill(ME, THIRD, ME, 2.0)
    report.reward(1500, 700)
    assert report.finish(ME, ME) == ["sharpshooter", "mvp"]
    assert MatchReport().finish(OTHER, ME) == []
    text = report.summary("OPERATOR", 2350, uplinks=True)
    lines = text.split("\n")
    assert lines[0] == "MVP   OPERATOR   2.4K"
    assert "ELIMINATIONS 2   DEATHS 0   K/D 2.0" in lines
    assert "ACCURACY 60%   (24 of 40)" in lines
    assert "DAMAGE 600   CAPTURES 0" in lines and "+1.5K CREDITS   +700 XP" in lines
    assert "FIRST STRIKE   DOUBLE DOWN" in lines and "SHARPSHOOTER   MVP" in lines
    report.reset()
    assert report.kills == report.shots == 0 and report.medals == [] and "NO MEDALS" in report.summary()


def manager(day=20000, hour=10):
    pm = SS.ProfileManager(tempfile.mkdtemp())
    pm.clock = lambda: day * 86400.0 + hour * 3600.0
    assert pm.load() == "new"
    return pm


def test_contract_rotation_follows_the_calendar():
    assert SS.contract_week(3) == 0 and SS.contract_week(4) == 1          # 1970-01-05 was a Monday
    first = SS.daily_contracts(20000)
    assert first == SS.daily_contracts(20000) and len(set(first)) == C.DAILY_CONTRACTS
    assert any(SS.daily_contracts(20000 + d) != first for d in range(1, 5))
    pm = manager()
    rows = pm.contract_rows()
    assert [row["scope"] for row in rows] == ["DAILY"] * C.DAILY_CONTRACTS + ["WEEKLY"]
    assert all(row["progress"] == 0 and not row["done"] for row in rows)
    daily, weekly = pm.contract_resets()
    assert 0.0 < daily <= 86400.0 and 0.0 < weekly <= 7 * 86400.0


def test_progress_pays_out_once_and_renews_with_the_day():
    pm = manager()
    key = pm.profile.contracts["daily"][0]["key"]
    stat = SS.CONTRACT_ROWS[key]["stat"]
    target = SS.contract_target(key, False)
    credits = pm.profile.credits
    pm.record(stat, target - 1)
    assert pm.pop_notices() == [] and not pm.profile.contracts["daily"][0]["done"]
    pm.record(stat, 5)
    notices = pm.pop_notices()
    assert any(n[0] == "CONTRACT COMPLETE" and n[2] == C.CONTRACT_REWARDS[0] for n in notices)
    assert pm.profile.contracts["daily"][0]["done"] and pm.profile.stats["contracts"] >= 1
    assert pm.profile.credits >= credits + C.CONTRACT_REWARDS[0]
    pm.record(stat, 1000)                                                    # the same contract never pays twice
    assert not any(n == ("CONTRACT COMPLETE", SS.contract_text(key, False), C.CONTRACT_REWARDS[0])
                   for n in pm.pop_notices())
    pm.record("best_streak", 99, maximum=True)                               # records never feed contracts
    pm.clock = lambda: 20001 * 86400.0 + 600.0                              # the next day
    pm.refresh_contracts()
    assert pm.profile.contracts["day"] == 20001 and not any(e["done"] for e in pm.profile.contracts["daily"])


def test_contracts_survive_the_signed_save_and_are_sanitised():
    pm = manager()
    pm.profile.contracts["daily"][0]["progress"] = 1.0e9                     # out of range -> clamped and done
    pm.profile.contracts["daily"].append({"key": "nonsense", "progress": 1, "done": False})
    loaded = SS.Profile.from_dict(pm.profile.to_dict())
    contracts = loaded.contracts
    assert contracts["day"] == 20000 and len(contracts["daily"]) == C.DAILY_CONTRACTS
    first = contracts["daily"][0]
    assert first["done"] and first["progress"] == SS.contract_target(first["key"], False)
    assert contracts["weekly"]["key"] in SS.CONTRACT_ROWS
    old = pm.profile.to_dict()
    del old["contracts"]
    assert SS.Profile.from_dict(old).contracts == SS.empty_contracts()      # saves from before contracts load fine
