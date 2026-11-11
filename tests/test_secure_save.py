"""Module D: economy math, sponsor thread and tamper-evident persistence."""
import os
import tempfile
import time

import numpy as np

from helpers import raises

from engine import secure_save as SS
from engine import weapons as W


def test_cost_formula_and_bulk_closed_forms():
    assert float(SS.exp_cost(150.0, 1.32, 0)) == 150.0
    assert np.isclose(float(SS.exp_cost(150.0, 1.32, 7)), 150.0 * 1.32 ** 7)       # Cost = Base * Mult ^ Level
    levels = np.arange(6)
    assert np.allclose(SS.exp_cost(50.0, 1.15, levels), 50.0 * 1.15 ** levels)
    loop = sum(50.0 * 1.15 ** k for k in range(7, 32))
    assert np.isclose(float(SS.bulk_cost(50.0, 1.15, 7, 25)), loop)
    for budget in (0.0, 49.0, 50.0, 1.0e4, 1.0e9):
        n = int(SS.max_affordable(50.0, 1.15, 3, budget))
        assert float(SS.bulk_cost(50.0, 1.15, 3, n)) <= budget + 1e-6 < float(SS.bulk_cost(50.0, 1.15, 3, n + 1))
    assert float(SS.xp_required(1)) == SS.XP_BASE and SS.xp_required(10) > SS.xp_required(9)


def test_upgrade_cost_tables():
    costs = SS.weapon_upgrade_costs(0, np.zeros(8, int))
    assert np.allclose(costs, W.UPGRADE_BASE_COST)
    assert np.all(SS.weapon_upgrade_costs(6, np.zeros(8, int)) > costs)                # pricier weapon tier
    assert np.all(np.isinf(SS.weapon_upgrade_costs(0, W.UPGRADE_MAX)))
    one_more = SS.weapon_upgrade_costs(0, np.ones(8, int))
    assert np.allclose(one_more, np.rint(W.UPGRADE_BASE_COST * W.UPGRADE_COST_MULT))
    assert np.all(np.isinf(SS.operator_upgrade_costs(W.OPERATOR_MAX)))


def test_sponsor_income_milestones_and_prestige():
    assert SS.sponsor_income([0] * 8) == 0.0
    assert SS.sponsor_income([9, 0, 0, 0, 0, 0, 0, 0]) == 4.5
    assert SS.sponsor_income([10, 0, 0, 0, 0, 0, 0, 0]) == 10.0                         # milestone doubles the contract
    assert np.isclose(SS.sponsor_income([10, 0, 0, 0, 0, 0, 0, 0], 2), 10.0 * SS.PRESTIGE_INCOME_MULT ** 2)
    need0, level0 = SS.prestige_requirement(0)
    need1, level1 = SS.prestige_requirement(1)
    assert need1 == need0 * SS.PRESTIGE_REQUIREMENT_MULT and level1 > level0


def test_number_formatting():
    assert [SS.format_number(v) for v in (0, 12.5, 999, 1000, 1234567, 2.5e9)] == ["0", "12.5", "999", "1.00K", "1.23M", "2.50B"]
    assert SS.format_number(float("inf")) == "MAX" and SS.format_number(-1500) == "-1.50K"
    assert SS.format_duration(3725) == "1h 02m" and SS.format_duration(75) == "1m 15s"


def fresh_manager():
    manager = SS.ProfileManager(tempfile.mkdtemp())
    assert manager.load() == "new"
    return manager


def test_purchases_experience_and_prestige():
    pm = fresh_manager()
    assert not pm.buy_sponsor(0, 1)                                                    # broke
    pm.add_credits(10000.0)
    assert pm.buy_sponsor(0, 10) and pm.profile.sponsors[0] == 10 and pm.income_per_second() == 10.0
    before = pm.profile.credits
    cost = float(pm.weapon_costs("vanguard")[W.UPG_DAMAGE])
    assert pm.buy_weapon_upgrade("vanguard", W.UPG_DAMAGE) and np.isclose(before - pm.profile.credits, cost)
    assert not pm.buy_weapon_upgrade("breacher", W.UPG_DAMAGE)                         # locked weapon
    assert not pm.can_unlock("breacher")                                               # level too low
    gained = pm.add_xp(float(SS.xp_required(1) + SS.xp_required(2)) + 5.0)
    assert gained == 2 and pm.profile.level == 3 and np.isclose(pm.profile.xp, 5.0)
    assert pm.unlock_weapon("breacher") and pm.equip(0, "breacher") and pm.profile.loadout[0] == "breacher"
    assert pm.equip(1, "breacher") and pm.profile.loadout == ["sidekick", "breacher"]  # slots swap, never duplicate
    assert pm.equip(0, "vanguard") and pm.profile.loadout == ["vanguard", "breacher"]
    assert not pm.equip(0, "aurora")                                                   # still locked
    ids, levels, operator, prestige = pm.loadout_arrays()
    assert ids.tolist() == [W.WEAPON_INDEX["vanguard"], W.WEAPON_INDEX["breacher"]] and levels[0, W.UPG_DAMAGE] == 1
    assert len(pm.loadout_blob()) == W.LOADOUT_BLOB_SIZE
    assert pm.buy_operator_upgrade(W.OP_BOUNTY) and np.isclose(pm.combat_multiplier(), 1.1)
    assert np.isclose(pm.add_credits(100.0, combat=True), 110.0)
    assert not pm.can_prestige()
    pm.add_credits(SS.prestige_requirement(0)[0])
    pm.profile.level = SS.prestige_requirement(0)[1]
    assert pm.do_prestige() and pm.profile.prestige == 1 and pm.profile.credits == 0.0 and pm.profile.sponsors[0] == 0
    assert np.isclose(pm.combat_multiplier(), 1.25)


def test_corporate_sponsor_thread_accrues_in_real_time():
    pm = fresh_manager()
    pm.add_credits(1.0e6)
    pm.buy_sponsor(2, 10)
    rate = pm.income_per_second()
    assert rate == 18.0 * 10 * 2
    start_credits = pm.profile.credits
    thread = SS.CorporateSponsorThread(pm, interval=0.02, autosave_interval=0.1)
    started = time.perf_counter()
    thread.start()
    time.sleep(0.5)
    thread.stop(save=False)
    elapsed = time.perf_counter() - started
    assert thread.wakeups >= 10
    assert abs(thread.total_accrued - rate * elapsed) < rate * 0.08                    # tracks wall-clock time
    assert np.isclose(pm.profile.credits - start_credits, thread.total_accrued)
    assert pm.path.exists()                                                            # autosave fired


def test_save_file_layout_and_round_trip():
    pm = fresh_manager()
    pm.add_credits(4321.0)
    pm.set_name("Night Shift!")
    assert pm.save()
    lines = pm.path.read_text().strip().splitlines()
    assert lines[0] == SS.SAVE_HEADER and lines[-1].startswith(SS.SIGNATURE_PREFIX) and len(lines[-1]) == len(SS.SIGNATURE_PREFIX) + 64
    assert "4321" not in pm.path.read_text() and "credits" not in pm.path.read_text()  # body is sealed
    data_string = SS.canonical_json(pm.profile.to_dict())
    assert lines[-1][len(SS.SIGNATURE_PREFIX):] == SS.sign_data_string(data_string)   # sha256(data + salt)
    again = SS.ProfileManager(pm.directory)
    assert again.load() == "ok" and again.profile.credits >= 4321.0 and again.profile.name == "Night Shift"


def test_tampering_is_detected_and_backup_restored():
    pm = fresh_manager()
    pm.add_credits(777.0)
    pm.save()
    pm.add_credits(1.0)
    pm.save()                                                                          # second save creates the backup
    text = pm.path.read_text()
    lines = text.strip().splitlines()
    body = list(lines[1])
    body[7] = "A" if body[7] != "A" else "B"
    for forged in ("\n".join([lines[0], "".join(body)] + lines[2:]) + "\n",            # edited body
                   "\n".join(lines[:-1] + [SS.SIGNATURE_PREFIX + "0" * 64]) + "\n",    # edited signature
                   text.replace(SS.SAVE_HEADER, "NEXUS3D-PROFILE v9")):                # edited header
        with raises(SS.SaveTamperError):
            SS.decode_save(forged)
    pm.path.write_text("\n".join([lines[0], "".join(body)] + lines[2:]) + "\n")
    restored = SS.ProfileManager(pm.directory)
    assert restored.load() == "restored_backup" and restored.profile.integrity_strikes == 1
    assert 777.0 <= restored.profile.credits < 778.5
    assert any("tampered" in name for name in os.listdir(pm.directory))                # evidence is quarantined
    restored.save()
    restored.path.write_text("garbage")
    restored.backup_path.write_text("more garbage")
    reset = SS.ProfileManager(pm.directory)
    assert reset.load() == "tampered_reset" and reset.profile.credits == 0.0 and reset.profile.integrity_strikes == 1


def test_hand_crafted_save_without_the_salt_is_rejected():
    pm = fresh_manager()
    pm.save()
    rich = SS.Profile()
    rich.credits = 1.0e12
    data_string = SS.canonical_json(rich.to_dict())
    import base64
    import hashlib
    from engine import crypto
    sealed = base64.b64encode(crypto.seal(SS.SAVE_KEY, data_string.encode())).decode()
    unsalted = hashlib.sha256(data_string.encode()).hexdigest()
    with raises(SS.SaveTamperError):
        SS.decode_save("%s\n%s\n%s%s\n" % (SS.SAVE_HEADER, sealed, SS.SIGNATURE_PREFIX, unsalted))
    assert SS.decode_save(SS.encode_save(rich)).credits == 1.0e12                      # the genuine encoder is accepted


def test_offline_earnings_are_capped_and_loaded_values_are_clamped():
    pm = fresh_manager()
    pm.add_credits(1.0e5)
    pm.buy_sponsor(0, 10)
    pm.save()
    pm.profile.saved_at -= 3600.0
    text = SS.encode_save(pm.profile)
    pm.path.write_text(text)
    later = SS.ProfileManager(pm.directory)
    assert later.load() == "ok"
    assert abs(later.offline_credits - 10.0 * 3600.0 * SS.OFFLINE_EFFICIENCY) < 10.0 * 5
    pm.profile.saved_at -= 30 * 24 * 3600.0
    pm.path.write_text(SS.encode_save(pm.profile))
    capped = SS.ProfileManager(pm.directory)
    capped.load()
    assert np.isclose(capped.offline_credits, 10.0 * SS.OFFLINE_CAP_SECONDS * SS.OFFLINE_EFFICIENCY)
    raw = pm.profile.to_dict()
    raw.update(level=9999, prestige=-5, operator=[99, 99, 99, 99], sponsors=[10 ** 9] * 8)
    raw["weapons"]["vanguard"]["levels"] = [500] * 8
    clean = SS.Profile.from_dict(raw)
    assert clean.level == SS.MAX_LEVEL and clean.prestige == 0 and clean.operator == [int(v) for v in W.OPERATOR_MAX]
    assert clean.weapons["vanguard"]["levels"] == [int(v) for v in W.UPGRADE_MAX] and clean.sponsors[0] == SS.SPONSOR_MAX_LEVEL
