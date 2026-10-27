"""data/econ_archive.py: fetch, merge, guard and cadence, against recorded
fixtures. No test here touches the network: every run() takes an injected fake
session, and a sentinel session fails the test if a real call is attempted.

Fixtures in tests/fixtures/econ/ are verbatim cuts of real responses downloaded
2026-10-23 (the older GetDeploying copy) and 2026-10-27 (everything else).
"""
from __future__ import annotations

import csv
import inspect
import io
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

import build_static
import data_guard
from data import econ_archive as E

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures" / "econ"
NOW = datetime(2026, 10, 27, 14, 0, tzinfo=timezone.utc)


def _fx(name: str) -> str:
    return (FX / name).read_text(encoding="utf-8")


def _rows(name: str) -> tuple[list[str], list[list[str]]]:
    return E._parse_csv(_fx(name))


class _Resp:
    """Mimics requests.Response: `.content` is bytes, and `.text` decodes them
    with `.encoding`, which requests sets to ISO-8859-1 for a text/* response
    with no charset (what epoch.ai sends, verified 2026-10-27)."""

    def __init__(self, body, status: int = 200, encoding: str = "utf-8"):
        self.body, self.status_code, self.encoding = body, status, encoding

    @property
    def content(self) -> bytes:
        if isinstance(self.body, bytes):
            return self.body
        s = self.body if isinstance(self.body, str) else json.dumps(self.body)
        return s.encode("utf-8")

    @property
    def text(self):
        return self.content.decode(self.encoding or "utf-8", errors="replace")

    def json(self):
        return json.loads(self.body) if isinstance(self.body, str) else self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Serves the recorded fixtures by URL; records every call."""

    def __init__(self, overrides: dict | None = None, fail_views: set | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.overrides = overrides or {}
        self.fail_views = fail_views or set()

    def get(self, url, params=None, headers=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params))
        # epoch.ai: Content-Type text/csv with no charset -> requests' Latin-1
        enc = "ISO-8859-1" if url.startswith(E.EPOCH_BASE) else "utf-8"
        for key, body in self.overrides.items():
            if url.endswith(key):
                if isinstance(body, Exception):
                    raise body
                return _Resp(body, encoding=enc)
        if url == E.GD_URL:
            return _Resp(_fx("getdeploying_weekly_2026-10-27.csv"))
        if url == E.SOFR_SEARCH_URL:
            return _Resp(_fx("sofr_search_small.json"))
        if url == E.SOFR_LAST_URL:
            return _Resp(_fx("sofr_last3.json"))
        if url.endswith("/views/options"):
            return _Resp(_fx("ix_options.json"))
        if url.endswith("/availability"):
            return _Resp(_fx("ix_availability.json"))
        if url.endswith("/views/inference"):
            key = (params.get("model"), params.get("sequence"))
            if key in self.fail_views:
                raise requests.ConnectionError(f"forced failure {key}")
            if key == ("DeepSeek-V4-Pro", "8k/1k"):
                return _Resp(_fx("ix_inference_dsv4_8k1k.json"))
            if key == ("Kimi-K3", "agentic-traces"):
                return _Resp(_fx("ix_inference_kimik3_agentic.json"))
            field = "p90_intvty" if key[1] == "agentic-traces" else "median_intvty"
            return _Resp({"xAxis": {"field": field}, "series": []})
        epoch = {
            "ml_hardware.csv": "epoch_ml_hardware_head.csv",
            "ai_companies_revenue_reports.csv": "epoch_revenue_head.csv",
            "ai_companies_compute_spend.csv": "epoch_compute_head.csv",
            "ai_companies_funding_rounds.csv": "epoch_funding_head.csv",
            "ai_companies_usage_reports.csv": "epoch_usage_head.csv",
        }
        for base, fx in epoch.items():
            if url == E.EPOCH_BASE + base:
                return _Resp(_fx(fx), encoding=enc)
        raise AssertionError(f"unexpected URL in test: {url}")


class NoNetwork:
    def get(self, *a, **k):
        raise AssertionError("a real network call was attempted")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Archives in tmp, fixture-sized floors (the real floors assume full files)."""
    monkeypatch.setattr(E, "ECON_DIR", tmp_path / "econ")
    monkeypatch.setattr(E, "EPOCH_DIR", tmp_path / "epoch")
    monkeypatch.setattr(E, "GD_MIN_UPSTREAM_ROWS", 1)
    low = {k: {**v, "min_rows": 1} for k, v in data_guard.ARCHIVE_CSVS.items()}
    monkeypatch.setattr(data_guard, "ARCHIVE_CSVS", low)
    return tmp_path


def _state(tmp: Path) -> dict:
    return json.loads((tmp / "econ" / "fetch_state.json").read_text())


# ── GetDeploying ──────────────────────────────────────────────────────────────

def test_getdeploying_merge_keeps_weeks_that_left_the_window():
    """2025-10-20 left upstream's rolling window between 10-23 and 10-27; the
    archive keeps it, takes the revised 2026-10-19 values, adds 2026-10-26."""
    _, old = _rows("getdeploying_weekly_2026-10-23.csv")
    _, new = _rows("getdeploying_weekly_2026-10-27.csv")
    assert not [r for r in new if r[1] == "2025-10-20"], "fixture premise"
    merged = E.merge_getdeploying(old, new)
    assert len(merged) >= len(old) and len(merged) >= len(new)
    keys = {E._gd_key(r) for r in merged}
    assert {E._gd_key(r) for r in old} <= keys
    assert {E._gd_key(r) for r in new} <= keys
    assert [r for r in merged if r[1] == "2025-10-20"] == sorted(
        [r for r in old if r[1] == "2025-10-20"], key=E._gd_sort)
    # upstream wins on the revised in-progress week
    by_key = {E._gd_key(r): r for r in merged}
    for r in new:
        assert by_key[E._gd_key(r)] == r
    assert any(r[1] == "2026-10-26" for r in merged)


def test_seed_never_overrides_archive_rows():
    _, old = _rows("getdeploying_weekly_2026-10-23.csv")
    _, new = _rows("getdeploying_weekly_2026-10-27.csv")
    archive = E.merge_getdeploying([], new)
    seeded = E.merge_getdeploying(archive, old, upstream_wins=False)
    by_key = {E._gd_key(r): r for r in seeded}
    for r in new:
        assert by_key[E._gd_key(r)] == r, "the seed overrode a newer archive row"
    assert any(r[1] == "2025-10-20" for r in seeded), "the seed recovered nothing"


def test_getdeploying_cells_are_byte_verbatim_and_sorted(sandbox):
    out = E.run(only=["getdeploying"], now=NOW, session=FakeSession(), sleep_s=0)
    assert out["getdeploying"]["status"] == "ok", out
    text = (sandbox / "econ" / "gpu_prices_weekly.csv").read_text()
    assert "\r" not in text and text.endswith("\n")
    upstream_lines = set(_fx("getdeploying_weekly_2026-10-27.csv").splitlines())
    assert set(text.splitlines()) == upstream_lines, "cells were reformatted"
    assert "1.6972" in text                           # verbatim, not 1.6972000001
    header, rows = E._parse_csv(text)
    assert header == E.GD_HEADER
    assert rows == sorted(rows, key=E._gd_sort)


def test_rewrite_with_nothing_new_is_byte_identical(sandbox):
    E.run(only=["getdeploying"], now=NOW, session=FakeSession(), sleep_s=0)
    path = sandbox / "econ" / "gpu_prices_weekly.csv"
    before, mtime = path.read_bytes(), path.stat().st_mtime_ns
    time.sleep(0.01)
    out = E.run(only=["getdeploying"], now=NOW, force=True, session=FakeSession(), sleep_s=0)
    assert out["getdeploying"]["wrote"] == []
    assert path.read_bytes() == before and path.stat().st_mtime_ns == mtime
    assert E._write_if_changed(path, before.decode()) is False


@pytest.mark.parametrize("mutate, reason", [
    (lambda t: t.replace("median_price", "median_usd", 1), "header"),
    (lambda t: t.replace("2026-10-26", "2026-09-28").replace("2026-10-19", "2026-09-21")
                .replace("2026-10-12", "2026-09-14"), "days old"),
    (lambda t: t.replace(",3.92,", ",0,", 1), "median_price"),
    (lambda t: t.replace(",3.92,", ",392,", 1), "median_price"),
])
def test_getdeploying_guards_refuse(sandbox, mutate, reason):
    good = FakeSession()
    E.run(only=["getdeploying"], now=NOW, session=good, sleep_s=0)
    path = sandbox / "econ" / "gpu_prices_weekly.csv"
    before = path.read_bytes()
    bad = FakeSession({E.GD_URL: mutate(_fx("getdeploying_weekly_2026-10-27.csv"))})
    out = E.run(only=["getdeploying"], now=NOW, force=True, session=bad, sleep_s=0)
    assert out["getdeploying"]["status"] == "failed"
    assert reason in out["getdeploying"]["error"]
    assert path.read_bytes() == before, "a refused fetch changed the archive"
    st = _state(sandbox)["getdeploying"]
    assert st["ok"] is False and st["failing_since"]


def test_getdeploying_refuses_an_upstream_that_shrank(sandbox):
    E.run(only=["getdeploying"], now=NOW, session=FakeSession(), sleep_s=0)
    state_path = sandbox / "econ" / "fetch_state.json"
    st = json.loads(state_path.read_text())
    st["getdeploying"]["upstream_rows"] = 1000          # last fetch had 1000
    state_path.write_text(json.dumps(st))
    out = E.run(only=["getdeploying"], now=NOW, force=True, session=FakeSession(), sleep_s=0)
    assert out["getdeploying"]["status"] == "failed"
    assert ">20%" in out["getdeploying"]["error"]


# ── SOFR ──────────────────────────────────────────────────────────────────────

def test_sofr_parse_merge_and_revision(sandbox):
    rows = E.parse_sofr(json.loads(_fx("sofr_last3.json")))
    assert [r[:2] for r in rows] == [["2026-10-26", "3.93"], ["2026-10-23", "3.92"],
                                     ["2026-10-22", "3.90"]]
    assert rows[0][2] == "3086"
    merged = E.merge_sofr([["2026-10-22", "3.95", "2900", ""]], rows)
    assert merged[0] == ["2026-10-22", "3.90", "2945", ""], "a revision did not replace"
    assert [r[0] for r in merged] == sorted(r[0] for r in merged)

    # First run backfills from SOFR's first day; the next uses last/10.
    s = FakeSession()
    first = datetime(2026, 10, 12, 14, tzinfo=timezone.utc)
    assert E.run(only=["sofr"], now=first, session=s, sleep_s=0)["sofr"]["status"] == "ok"
    assert s.calls == [(E.SOFR_SEARCH_URL, {"startDate": "2018-04-02", "endDate": "2026-10-12"})]
    # 15 days later the archive is stale: search from a week before its newest
    # row (served here with the real 10-26 response), never a gap.
    s2 = FakeSession({E.SOFR_SEARCH_URL: _fx("sofr_last3.json")})
    out = E.run(only=["sofr"], now=NOW, force=True, session=s2, sleep_s=0)
    assert out["sofr"]["status"] == "ok", out
    assert s2.calls == [(E.SOFR_SEARCH_URL, {"startDate": "2026-10-02", "endDate": "2026-10-27"})]
    s3 = FakeSession()
    out = E.run(only=["sofr"], now=NOW + timedelta(days=1), force=True, session=s3, sleep_s=0)
    assert out["sofr"]["status"] == "ok", out
    assert s3.calls == [(E.SOFR_LAST_URL, {})]
    header, body = E._parse_csv((sandbox / "econ" / "sofr.csv").read_text())
    assert header == E.SOFR_HEADER
    assert body[-1] == ["2026-10-26", "3.93", "3086", ""]
    assert len(body) == len({r[0] for r in body})


def test_sofr_request_after_a_long_absence_searches_from_a_week_before():
    url, params = E.sofr_request([["2026-09-01", "4.30", "2000", ""]],
                                 datetime(2026, 10, 27).date())
    assert url == E.SOFR_SEARCH_URL and params["startDate"] == "2026-08-25"


@pytest.mark.parametrize("patch", [{"percentRate": 393}, {"type": "EFFR"}, {"effectiveDate": "26/10/2026"}])
def test_sofr_units_guard(patch):
    payload = json.loads(_fx("sofr_last3.json"))
    payload["refRates"][0].update(patch)
    with pytest.raises(E.SourceError):
        E.parse_sofr(payload)


# ── InferenceX ────────────────────────────────────────────────────────────────

def _options() -> dict:
    return json.loads(_fx("ix_options.json"))


def test_ix_rates_change_log():
    opts = _options()
    first = E.parse_ix_options(opts, "2026-10-27")
    assert len(first) == len(opts["hardware"]) == 13
    gb300 = [r for r in first if r[1] == "gb300"][0]
    assert gb300[6:] == ["2.31", "5"]
    log = E.merge_ix_rates([], first)
    assert E.merge_ix_rates(log, E.parse_ix_options(opts, "2026-11-02")) == log
    opts["hardware"] = [dict(h) for h in opts["hardware"]]
    for h in opts["hardware"]:
        if h["key"] == "h100":
            h["costPerHour"] = {"h": 1.10, "r": 2}
    opts["hardware"] = [h for h in opts["hardware"] if h["key"] != "vr200"]
    log2 = E.merge_ix_rates(log, E.parse_ix_options(opts, "2026-11-09"))
    assert len(log2) == len(log) + 1
    h100 = [r for r in log2 if r[1] == "h100"]
    assert [r[0] for r in h100] == ["2026-10-27", "2026-11-09"] and h100[-1][6] == "1.1"
    assert [r for r in log2 if r[1] == "vr200"], "vanished hardware lost its rows"


def test_ix_options_guard_refuses_too_few_priced_chips():
    opts = _options()
    opts["hardware"] = opts["hardware"][:5]
    with pytest.raises(E.SourceError):
        E.parse_ix_options(opts, "2026-10-27")


def test_ix_request_plan_from_availability():
    plan = E.plan_ix_requests(_options(), json.loads(_fx("ix_availability.json")))
    assert ("DeepSeek-V4-Pro", "8k/1k") in plan
    assert ("DeepSeek-V4-Pro", "agentic-traces") in plan
    assert ("Kimi-K3", "agentic-traces") in plan
    assert ("Kimi-K2.5", "8k/1k") in plan                # dbKey kimik2.6 -> display name
    assert all(seq in ("8k/1k", "agentic-traces") for _, seq in plan), "deprecated sequence planned"
    assert not [p for p in plan if p[0] == "DeepSeek-R1-0528"], "1k/1k-only model planned"
    params = E.ix_view_params(_options(), "DeepSeek-V4-Pro", "8k/1k")
    assert params["gpus"].split(",") == [h["key"] for h in _options()["hardware"]]
    assert params["precisions"].split(",") == _options()["precisions"]
    assert params["xmode"] == "interactivity" and params["format"] == "json"
    assert params["optimal"] == "false" and params["best"] == "false"


def test_ix_points_parse():
    rows, dropped = E.parse_ix_view(json.loads(_fx("ix_inference_dsv4_8k1k.json")),
                                    "DeepSeek-V4-Pro", "8k/1k", "2026-10-27",
                                    {"gb300": "GB300 NVL72"})
    assert dropped == 0 and rows
    col = {c: i for i, c in enumerate(E.IX_POINTS_HEADER)}
    assert all(len(r) == len(E.IX_POINTS_HEADER) for r in rows)
    assert {r[col["x_field"]] for r in rows} == {"median_intvty"}
    assert {r[col["hw_key"]] for r in rows} == {"gb300"}
    assert {r[col["gpu"]] for r in rows} == {"GB300 NVL72"}
    payload = json.loads(_fx("ix_inference_dsv4_8k1k.json"))
    pts = {str(p["id"]): (s, p) for s in payload["series"] for p in s["points"]}
    for r in rows:
        s, p = pts[r[col["point_id"]]]
        assert r[col["gpu_count_upstream"]] == str(p["tp"])
        assert r[col["gpu_count"]] == str(p["tp"])         # disagg: P-GPU + D-GPU, kept
        assert r[col["framework"]] == s["framework"] and r[col["spec_method"]] == s["specMethod"]
        topo = dict(t.split("=", 1) for t in p["topologyKey"].split("|") if "=" in t)
        assert r[col["tensor_parallel"]] == ("" if topo["TP"] == "?" else topo["TP"])
        assert r[col["expert_parallel"]] == ("" if topo["EP"] == "?" else topo["EP"])
        assert float(r[col["tput_tok_s_chip"]]) == pytest.approx(p["metrics"]["tpPerGpu"], rel=1e-5)
        assert r[col["run_id"]] == str(p["runId"]) and r[col["first_seen"]] == "2026-10-27"
    agentic, _ = E.parse_ix_view(json.loads(_fx("ix_inference_kimik3_agentic.json")),
                                 "Kimi-K3", "agentic-traces", "2026-10-27")
    assert {r[col["x_field"]] for r in agentic} == {"p90_intvty"}
    assert E.run_url(26856602694) == "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/26856602694"
    assert E.run_url("javascript:alert(1)") == ""
    topo = E._topology("PD|GPU=8|TP=?|EP=8|P-TP=4")
    assert topo["TP"] == "" and topo["EP"] == "8" and topo["P-TP"] == "4"


def test_ix_points_drop_invalid():
    payload = json.loads(_fx("ix_inference_dsv4_8k1k.json"))
    payload["series"][0]["points"][0]["x"] = 0
    payload["series"][0]["points"][1]["metrics"]["tpPerGpu"] = None
    rows, dropped = E.parse_ix_view(payload, "DeepSeek-V4-Pro", "8k/1k", "2026-10-27")
    assert dropped == 2


def test_ix_merge_dedupes_by_point_id_and_keeps_first_seen():
    payload = json.loads(_fx("ix_inference_dsv4_8k1k.json"))
    old, _ = E.parse_ix_view(payload, "DeepSeek-V4-Pro", "8k/1k", "2026-10-20")
    new, _ = E.parse_ix_view(payload, "DeepSeek-V4-Pro", "8k/1k", "2026-10-27")
    merged = E.merge_ix_points(old, new + new)
    assert len(merged) == len(old)
    assert {r[E._IX_FIRST_SEEN] for r in merged} == {"2026-10-20"}
    gone = E.merge_ix_points(merged, new[:1])
    assert len(gone) == len(merged), "points that left the view were deleted"


def test_ix_cost_units_match_the_api():
    """$/M tok = (USD/chip-hr) x 1e6 / (3600 x tok/s/chip), against the API's own
    costhOutput for the same point (GB300 NVL72, DeepSeek-V4-Pro 8k/1k)."""
    view = json.loads(_fx("ix_inference_dsv4_8k1k_costhOutput_gb300.json"))
    pt = view["series"][0]["points"][0]
    h = [r for r in E.parse_ix_options(_options(), "2026-10-27") if r[1] == "gb300"][0][6]
    usd_per_m = float(h) * 1e6 / (3600 * pt["metrics"]["outputTputPerGpu"])
    assert usd_per_m == pytest.approx(pt["metrics"]["costhOutput"], abs=1e-9)
    assert pt["metrics"]["costhOutput"] == pytest.approx(0.06624178692477896, abs=1e-12)


def test_ix_partial_failure(sandbox):
    # 1 of 5 planned views failing: the rest merge, ok.
    s = FakeSession(fail_views={("Kimi-K3", "agentic-traces")})
    out = E.run(only=["inferencex"], now=NOW, session=s, sleep_s=0)
    assert out["inferencex"]["status"] == "ok", out
    pts = (sandbox / "econ" / "inferencex_points.csv").read_text()
    assert "DeepSeek-V4-Pro" in pts and "Kimi-K3" not in pts
    # 3 of 5 failing: not ok, but what did arrive is still merged.
    s = FakeSession(fail_views={("Kimi-K3", "agentic-traces"), ("DeepSeek-V4-Pro", "agentic-traces"),
                                ("Kimi-K2.5", "8k/1k")})
    out = E.run(only=["inferencex"], now=NOW, force=True, session=s, sleep_s=0)
    assert out["inferencex"]["status"] == "failed"
    assert "views succeeded" in out["inferencex"]["error"]
    # An exhausted budget stops issuing view calls and records 'budget'.
    s = FakeSession()
    o = E.run_inferencex(s, NOW, {}, deadline=time.monotonic() - 1, sleep_s=0)
    assert not o.ok and o.error.startswith("budget")
    assert not [c for c in s.calls if c[0].endswith("/views/inference")]


def test_ix_archive_is_append_only_across_runs(sandbox):
    E.run(only=["inferencex"], now=NOW, session=FakeSession(), sleep_s=0)
    path = sandbox / "econ" / "inferencex_points.csv"
    n = len(E._parse_csv(path.read_text())[1])
    empty = {"xAxis": {"field": "median_intvty"}, "series": []}
    later = NOW + timedelta(days=7)
    s = FakeSession({"/views/inference": empty})
    E.run(only=["inferencex"], now=later, session=s, sleep_s=0)
    assert len(E._parse_csv(path.read_text())[1]) == n


# ── Epoch mirrors ─────────────────────────────────────────────────────────────

def test_epoch_mirror_sanitised_and_stable(sandbox):
    out = E.run(only=["epoch_hardware", "epoch_companies"], now=NOW, session=FakeSession(), sleep_s=0)
    assert out["epoch_hardware"]["status"] == "ok", out
    assert out["epoch_companies"]["status"] == "ok", out
    hw = (sandbox / "epoch" / "ml_hardware.csv").read_text()
    raw = _fx("epoch_ml_hardware_head.csv")
    assert "\n" in next(r for r in csv.reader(io.StringIO(raw)) if "H100 SXM5" in r[0])[3], "fixture premise"
    header, rows = E._parse_csv(hw)
    assert len(rows) == 5 and hw.count("\n") == 6, "an embedded newline survived"
    fund = (sandbox / "epoch" / "ai_companies_funding_rounds.csv").read_text()
    assert "\t" not in fund
    # numbers untouched
    _, raw_rows = E._parse_csv(raw)
    tdp = header.index("TDP (W)")
    assert [r[tdp] for r in rows] == [r[tdp] for r in raw_rows]
    before = {p.name: p.read_bytes() for p in (sandbox / "epoch").iterdir()}
    out = E.run(only=["epoch_hardware", "epoch_companies"], now=NOW, force=True,
                session=FakeSession(), sleep_s=0)
    assert out["epoch_hardware"]["wrote"] == [] and out["epoch_companies"]["wrote"] == []
    assert {p.name: p.read_bytes() for p in (sandbox / "epoch").iterdir()} == before


def test_epoch_refuses_missing_columns_and_big_drops(sandbox, monkeypatch):
    E.run(only=["epoch_hardware"], now=NOW, session=FakeSession(), sleep_s=0)
    path = sandbox / "epoch" / "ml_hardware.csv"
    before = path.read_bytes()
    renamed = _fx("epoch_ml_hardware_head.csv").replace("TDP (W)", "Power (W)", 1)
    out = E.run(only=["epoch_hardware"], now=NOW, force=True,
                session=FakeSession({"ml_hardware.csv": renamed}), sleep_s=0)
    assert out["epoch_hardware"]["status"] == "failed" and "missing columns" in out["epoch_hardware"]["error"]
    header, rows = _rows("epoch_ml_hardware_head.csv")
    shrunk = E._csv_text(header, rows[:3])                    # 5 -> 3 rows = 40% drop
    out = E.run(only=["epoch_hardware"], now=NOW, force=True,
                session=FakeSession({"ml_hardware.csv": shrunk}), sleep_s=0)
    assert out["epoch_hardware"]["status"] == "failed" and "drop" in out["epoch_hardware"]["error"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("basename, column, value", [
    ("ai_companies_revenue_reports.csv", "Annualized revenue (USD)", "4.5e13"),   # 1000x
    ("ai_companies_revenue_reports.csv", "Annualized revenue (USD)", "4500"),     # in $M?
    ("ai_companies_usage_reports.csv", "Daily tokens", "1000000.0"),
    ("ai_companies_usage_reports.csv", "Date", "2010-01-01"),
    ("ai_companies_revenue_reports.csv", "Date", "2027-12-31"),
    ("ai_companies_revenue_reports.csv", "Date", "last spring"),
])
def test_epoch_unit_ranges(basename, column, value):
    fx = {"ai_companies_revenue_reports.csv": "epoch_revenue_head.csv",
          "ai_companies_usage_reports.csv": "epoch_usage_head.csv"}[basename]
    header, rows = _rows(fx)
    assert E.validate_epoch(basename, header, rows, NOW.date()) == [], "fixture must pass as recorded"
    i = header.index(column)
    rows[0][i] = value
    assert E.validate_epoch(basename, header, rows, NOW.date())


def test_epoch_columns_match_case_insensitively():
    header, rows = _rows("epoch_compute_head.csv")
    assert "Source Type" in header                         # Epoch's own inconsistency
    rev_h, rev_rows = _rows("epoch_revenue_head.csv")
    rev_h = [("Source Type" if h == "Source type" else h) for h in rev_h]
    assert E.validate_epoch("ai_companies_revenue_reports.csv", rev_h, rev_rows, NOW.date()) == []


# ── cadence, isolation, escalation ───────────────────────────────────────────

def test_cadence_and_retry():
    t = NOW
    assert E.due({}, "getdeploying", t)
    ok = {"getdeploying": {"checked_at": E._iso(t - timedelta(hours=5)), "ok": True}}
    assert not E.due(ok, "getdeploying", t)
    assert E.due(ok, "getdeploying", t, force=True)
    ok["getdeploying"]["checked_at"] = E._iso(t - timedelta(hours=144))
    assert E.due(ok, "getdeploying", t)
    bad = {"getdeploying": {"checked_at": E._iso(t - timedelta(hours=5)), "ok": False}}
    assert not E.due(bad, "getdeploying", t)
    bad["getdeploying"]["checked_at"] = E._iso(t - timedelta(hours=6))
    assert E.due(bad, "getdeploying", t)
    assert E.due({"sofr": {"checked_at": E._iso(t - timedelta(hours=20)), "ok": True}}, "sofr", t)
    st = {"sofr": {"failing_since": E._iso(t - timedelta(hours=23)), "error": "x"}}
    assert E.escalate(st, t) == []
    st["sofr"]["failing_since"] = E._iso(t - timedelta(hours=24))
    assert E.escalate(st, t) and "sofr" in E.escalate(st, t)[0]
    assert E.escalate({"sofr": {"failing_since": None}}, t) == []


def test_failing_since_is_kept_until_a_success(sandbox):
    s = FakeSession({E.SOFR_SEARCH_URL: requests.ConnectionError("down")})
    E.run(only=["sofr"], now=NOW, session=s, sleep_s=0)
    first = _state(sandbox)["sofr"]["failing_since"]
    E.run(only=["sofr"], now=NOW + timedelta(hours=4), session=s, sleep_s=0)
    assert _state(sandbox)["sofr"]["failing_since"] == first
    E.run(only=["sofr"], now=datetime(2026, 10, 12, 18, tzinfo=timezone.utc), force=True,
          session=FakeSession(), sleep_s=0)
    st = _state(sandbox)["sofr"]
    assert st["ok"] is True and st["failing_since"] is None


def test_one_source_failing_never_stops_the_others(sandbox):
    s = FakeSession({E.GD_URL: requests.ConnectionError("forced")})
    out = E.run(now=NOW, session=s, sleep_s=0)
    assert out["getdeploying"]["status"] == "failed"
    for name in ("epoch_hardware", "epoch_companies", "inferencex"):
        assert out[name]["status"] == "ok", (name, out[name])
    assert (sandbox / "econ" / "inferencex_points.csv").exists()
    assert (sandbox / "epoch" / "ml_hardware.csv").exists()
    st = _state(sandbox)
    assert st["getdeploying"]["ok"] is False and st["getdeploying"]["failing_since"]
    assert st["inferencex"]["ok"] is True


def test_nothing_due_means_no_network_and_no_state_write(sandbox):
    E.run(only=["getdeploying"], now=NOW, session=FakeSession(), sleep_s=0)
    state = sandbox / "econ" / "fetch_state.json"
    before = state.read_bytes()
    out = E.run(only=["getdeploying"], now=NOW + timedelta(hours=1), session=NoNetwork(), sleep_s=0)
    assert out == {"getdeploying": {"status": "skipped"}}
    assert state.read_bytes() == before


def test_no_network_on_import_and_in_suite(sandbox, monkeypatch):
    import importlib

    def boom(*a, **k):
        raise AssertionError("network on import")
    monkeypatch.setattr(requests, "get", boom)
    importlib.reload(E)
    # reload rebound the module globals; point them back at the sandbox
    monkeypatch.setattr(E, "ECON_DIR", sandbox / "econ")
    monkeypatch.setattr(E, "EPOCH_DIR", sandbox / "epoch")
    out = E.run(now=NOW, session=NoNetwork(), sleep_s=0)
    assert all(r["status"] == "failed" and "network call" in r["error"] for r in out.values())
    assert "session" in inspect.signature(E.run).parameters


def test_sources_are_licensed_and_allowed():
    assert set(E.SOURCES) == set(E.RUNNERS)
    for name, src in E.SOURCES.items():
        for attr in ("url", "terms_url", "licence", "credit", "verified"):
            assert getattr(src, attr), f"{name}.{attr} is empty"
        assert src.cadence_h > src.retry_h > 0
        assert src.url.startswith("https://") and src.terms_url.startswith("https://")
        for host in E.BANNED_HOSTS:
            assert host not in src.url.lower() and host not in src.terms_url.lower()
    text = (ROOT / "data" / "econ_archive.py").read_text().lower()
    urls = [u for u in __import__("re").findall(r"https?://[^\s\"')]+", text)]
    for u in urls:
        for host in E.BANNED_HOSTS:
            assert host not in u, f"banned source referenced: {u}"
    assert "cc by 4.0" in E.SOURCES["getdeploying"].licence.lower()
    assert "cc by 4.0" in E.SOURCES["epoch_companies"].licence.lower()
    assert "inferencex" in E.SOURCES["inferencex"].credit.lower()


def test_archive_writer_uses_the_csv_sanitiser():
    import static_helpers
    assert E._FORMULA_LEAD == static_helpers._FORMULA_LEAD
    rows = E.sanitise_rows([['=HYPERLINK("https://x","y")', "-0.4", "+1e3", "a\nb", "\tTab", "@sum"]])
    assert rows == [["'=HYPERLINK(\"https://x\",\"y\")", "-0.4", "+1e3", "a b", " Tab", "'@sum"]]


def test_econ_module_is_not_shipped_to_the_browser():
    """Nothing in the browser reads these archives raw: they would break the
    ~100 KB pydata budget (the GetDeploying archive alone is ~1 MB)."""
    src = inspect.getsource(build_static.build_pybundle)
    assert '"data/econ_archive.py"' not in src
    shipped = {Path(p).name for p in build_static.DATA_CSVS}
    for name in ("gpu_prices_weekly.csv", "inferencex_points.csv", "sofr.csv"):
        assert name not in shipped


def test_committed_archives_have_the_documented_schema():
    """E, F and G code against these headers; changing one is a contract change."""
    expect = {
        E.GPU_PRICES: E.GD_HEADER,
        E.SOFR_CSV: E.SOFR_HEADER,
        E.IX_RATES: E.IX_RATES_HEADER,
        E.IX_POINTS: E.IX_POINTS_HEADER,
    }
    assert E.GD_HEADER == ["gpu_slug", "date", "billing_type", "reservation_months", "min_price",
                           "max_price", "median_price", "provider_median_price",
                           "provider_count", "offering_count"]
    assert E.SOFR_HEADER == ["effective_date", "percent_rate", "volume_bn", "revision_indicator"]
    for rel, header in expect.items():
        p = ROOT / rel
        if not p.exists():
            pytest.skip(f"{rel} not archived yet")
        with p.open(newline="", encoding="utf-8") as fh:
            assert next(csv.reader(fh)) == header, rel
    for base, rel in E.EPOCH_FILES.items():
        p = ROOT / rel
        if p.exists():
            with p.open(newline="", encoding="utf-8") as fh:
                got = [h.lower() for h in next(csv.reader(fh))]
            for col in E.EPOCH_REQUIRED[base]:
                assert col.lower() in got, (rel, col)
    assert set(data_guard.ARCHIVE_CSVS) == set(expect) | set(E.EPOCH_FILES.values())


# ── review fixes (2026-10-27) ────────────────────────────────────────────────

def test_epoch_mirror_is_decoded_as_utf8_not_latin1(sandbox):
    """epoch.ai sends text/csv with no charset, so requests' .text is Latin-1
    and every en dash / curly quote became mojibake in the committed mirrors.
    FakeSession serves Epoch exactly that way."""
    assert "OpenAI–Nvidia" in _fx("epoch_funding_head.csv"), "fixture premise"
    out = E.run(only=["epoch_hardware", "epoch_companies"], now=NOW, session=FakeSession(), sleep_s=0)
    assert out["epoch_companies"]["status"] == "ok", out
    fund = (sandbox / "epoch" / "ai_companies_funding_rounds.csv").read_text(encoding="utf-8")
    rev = (sandbox / "epoch" / "ai_companies_revenue_reports.csv").read_text(encoding="utf-8")
    assert "OpenAI–Nvidia" in fund
    assert "DeepSeek’s" in rev
    for p in (sandbox / "epoch").iterdir():
        assert "â\u0080" not in p.read_text(encoding="utf-8"), f"mojibake in {p.name}"


def test_non_utf8_body_is_refused_not_guessed():
    with pytest.raises(E.SourceError, match="not UTF-8"):
        E._utf8(_Resp("café,1\n".encode("latin-1"), encoding="ISO-8859-1"))
    assert E._utf8(_Resp("﻿a,b\n1,2\n")) == "a,b\n1,2\n"          # BOM stripped


def test_committed_epoch_mirrors_have_no_mojibake():
    for rel in E.EPOCH_FILES.values():
        p = ROOT / rel
        if p.exists():
            text = p.read_text(encoding="utf-8")
            for bad in ("â\u0080", "Â "):
                assert bad not in text, f"{rel} holds Latin-1-decoded UTF-8 ({bad!r})"


def test_allow_shrink_accepts_a_real_getdeploying_prune(sandbox):
    E.run(only=["getdeploying"], now=NOW, session=FakeSession(), sleep_s=0)
    state_path = sandbox / "econ" / "fetch_state.json"
    st = json.loads(state_path.read_text())
    real = st["getdeploying"]["upstream_rows"]
    st["getdeploying"]["upstream_rows"] = real * 2          # upstream then halved its window
    state_path.write_text(json.dumps(st))
    later = NOW + timedelta(hours=1)
    out = E.run(only=["getdeploying"], now=later, force=True, session=FakeSession(), sleep_s=0)
    assert out["getdeploying"]["status"] == "failed" and ">20%" in out["getdeploying"]["error"]
    # A failed source is not due again for retry_h; --allow-shrink retries it
    # now, accepts the candidate and resets the baseline.
    assert not E.due(_state(sandbox), "getdeploying", later + timedelta(minutes=5))
    out = E.run(only=["getdeploying"], now=later + timedelta(minutes=5), session=FakeSession(),
                sleep_s=0, allow_shrink=True)
    assert out["getdeploying"]["status"] == "ok", out
    st = _state(sandbox)["getdeploying"]
    assert st["upstream_rows"] == real and st["failing_since"] is None
    # ...and the next ordinary run passes on its own.
    out = E.run(only=["getdeploying"], now=later, force=True, session=FakeSession(), sleep_s=0)
    assert out["getdeploying"]["status"] == "ok", out


def test_allow_shrink_accepts_an_epoch_prune_but_keeps_floors(sandbox):
    E.run(only=["epoch_hardware"], now=NOW, session=FakeSession(), sleep_s=0)
    header, rows = _rows("epoch_ml_hardware_head.csv")
    shrunk = E._csv_text(header, rows[:3])                     # 40% drop
    s = FakeSession({"ml_hardware.csv": shrunk})
    out = E.run(only=["epoch_hardware"], now=NOW, force=True, session=s, sleep_s=0)
    assert out["epoch_hardware"]["status"] == "failed"
    out = E.run(only=["epoch_hardware"], now=NOW, force=True, session=s, sleep_s=0,
                allow_shrink=True)
    assert out["epoch_hardware"]["status"] == "ok", out
    assert len(E._parse_csv((sandbox / "epoch" / "ml_hardware.csv").read_text())[1]) == 3


def test_allow_shrink_waives_only_the_relative_rules():
    h = ["Hardware name", "TDP (W)"]
    before = E._csv_text(h, [[f"c{i}", "700"] for i in range(200)])
    after = E._csv_text(h, [[f"c{i}", "70"] for i in range(155)])     # -22%, median x0.1
    path = "data/raw/epoch/ml_hardware.csv"
    found = data_guard.archive_violations(path, before, after)
    assert any("drop" in p for p in found) and any("median" in p for p in found)
    assert data_guard.archive_violations(path, before, after, allow_shrink=True) == []
    tiny = E._csv_text(h, [[f"c{i}", "700"] for i in range(10)])
    assert data_guard.archive_violations(path, before, tiny, allow_shrink=True), "floor waived"
    sofr_h = ["effective_date", "percent_rate", "volume_bn", "revision_indicator"]
    rows = [[f"2020-01-{i:02d}", "1.50", "1000", ""] for i in range(1, 29)] * 50
    lost = data_guard.archive_violations("data/carried/econ/sofr.csv", E._csv_text(sofr_h, rows),
                                         E._csv_text(sofr_h, rows[:-1]), allow_shrink=True)
    assert any("never lose a row" in p for p in lost), "append-only waived"


def test_allow_shrink_flag_reaches_the_module(monkeypatch):
    seen = {}
    monkeypatch.setattr(E, "run", lambda **kw: seen.update(kw) or {})
    monkeypatch.setattr(E, "load_state", lambda *a: {})
    assert E.main(["--allow-shrink"]) == 0 and seen["allow_shrink"] is True
    assert E.main([]) == 0 and seen["allow_shrink"] is False


def test_ix_minority_view_failure_warns_retries_and_escalates(sandbox, capsys, monkeypatch):
    dead = ("Kimi-K3", "agentic-traces")
    out = E.run(only=["inferencex"], now=NOW, session=FakeSession(fail_views={dead}), sleep_s=0)
    r = out["inferencex"]
    assert r["status"] == "ok" and "1/" in r["warn"] and "Kimi-K3" in r["warn"]
    st = _state(sandbox)["inferencex"]
    assert st["view_failing_since"] == {"Kimi-K3 | agentic-traces": E._iso(NOW)}
    # retried on the short interval, not after six days
    assert E.due(_state(sandbox), "inferencex", NOW + timedelta(hours=6))
    assert E.escalate(_state(sandbox), NOW + timedelta(hours=23)) == []
    t2 = NOW + timedelta(hours=24)
    E.run(only=["inferencex"], now=t2, session=FakeSession(fail_views={dead}), sleep_s=0)
    st = _state(sandbox)["inferencex"]
    assert st["view_failing_since"]["Kimi-K3 | agentic-traces"] == E._iso(NOW), "clock reset"
    esc = E.escalate(_state(sandbox), t2)
    assert esc and "Kimi-K3 | agentic-traces" in esc[0]
    # recovery clears the clock and the short retry
    E.run(only=["inferencex"], now=t2 + timedelta(hours=6), session=FakeSession(), sleep_s=0)
    st = _state(sandbox)["inferencex"]
    assert "view_failing_since" not in st and E.escalate(_state(sandbox), t2 + timedelta(hours=7)) == []
    assert not E.due(_state(sandbox), "inferencex", t2 + timedelta(hours=13))
    # main() prints the warning for GitHub
    monkeypatch.setattr(E, "run", lambda **kw: {"inferencex": {
        "status": "ok", "rows": 1, "wrote": [], "notes": [], "error": "", "warn": "1/5 views failed: x"}})
    monkeypatch.setattr(E, "load_state", lambda *a: {})
    assert E.main([]) == 0
    assert "::warning title=Economics archive::inferencex: 1/5 views failed" in capsys.readouterr().out


def test_ix_view_clock_survives_a_budget_stop_and_a_crash():
    st = {"inferencex": {"view_failing_since": {"A | 8k/1k": "2026-10-26T00:00:00Z",
                                                "B | 8k/1k": "2026-10-26T00:00:00Z"}}}
    E._record(st, "inferencex", NOW, E.Outcome(False, error="budget", failed_views=[],
                                               pending_views=["A | 8k/1k"]))
    assert st["inferencex"]["view_failing_since"] == {"A | 8k/1k": "2026-10-26T00:00:00Z"}
    E._record(st, "inferencex", NOW, E.Outcome(False, error="ConnectionError"))
    assert st["inferencex"]["view_failing_since"] == {"A | 8k/1k": "2026-10-26T00:00:00Z"}


def test_ix_disagg_throughput_is_per_gpu_of_the_whole_deployment():
    """8k/1k disaggregated points: upstream's output is per DECODE GPU and its
    input per PREFILL GPU. Point 438832 (GB300, P-GPU=60, D-GPU=12)."""
    payload = json.loads(_fx("ix_inference_dsv4_8k1k.json"))
    up = {str(p["id"]): p for s in payload["series"] for p in s["points"]}
    rows, _ = E.parse_ix_view(payload, "DeepSeek-V4-Pro", "8k/1k", "2026-10-27")
    col = E._IX_COL
    r = {x[col["point_id"]]: x for x in rows}["438832"]
    assert (r[col["gpu_count"]], r[col["prefill_gpus"]], r[col["decode_gpus"]]) == ("72", "60", "12")
    m = up["438832"]["metrics"]
    assert m["outputTputPerGpu"] * 12 == pytest.approx(m["tpPerGpu"] * 72 / 9, rel=1e-3)  # premise
    tput, out, inp = (float(r[col[c]]) for c in
                      ("tput_tok_s_chip", "output_tput_tok_s_chip", "input_tput_tok_s_chip"))
    assert out == pytest.approx(tput / 9, rel=1e-3) and inp == pytest.approx(tput * 8 / 9, rel=1e-3)
    # upstream's value is recoverable from the stored split
    assert out * 72 / 12 == pytest.approx(m["outputTputPerGpu"], rel=1e-5)
    # every row now satisfies output + input = total, whatever its deployment
    for x in rows:
        assert float(x[col["output_tput_tok_s_chip"]]) + float(x[col["input_tput_tok_s_chip"]]) \
            == pytest.approx(float(x[col["tput_tok_s_chip"]]), rel=E._IX_BASIS_TOL)
    # point 424199 (56 GPUs, 8 decode): all-chip output pricing, not decode-only
    cost_view = json.loads(_fx("ix_inference_dsv4_8k1k_costhOutput_gb300.json"))
    r = E.parse_ix_view(cost_view, "DeepSeek-V4-Pro", "8k/1k", "2026-10-27")[0][0]
    assert r[col["point_id"]] == "424199" and r[col["decode_gpus"]] == "8"
    assert 2.31e6 / (3600 * float(r[col["output_tput_tok_s_chip"]])) == pytest.approx(0.4637, abs=1e-3)


def test_ix_unrecognised_throughput_basis_is_blank_not_guessed():
    assert E.ix_rebase_throughput(100.0, 10.0, 90.0, 8, None, None) == (10.0, 90.0)
    assert E.ix_rebase_throughput(100.0, 300.0, 50.0, 8, None, None) == (None, None)
    o, i = E.ix_rebase_throughput(100.0, 150.0, 100.0, 8, 4, 4)   # 150*4 + 100*4 = 1000 != 800
    assert (o, i) == (None, None)


def test_ix_single_node_gpu_count_is_physical():
    """'Single|GPU=64|TP=8|EP=8' runs on one 8-GPU node: tpPerGpu matches the
    TP=8/EP=1 run in the same sweep (real points 440984 / 440998)."""
    rows, _ = E.parse_ix_view(json.loads(_fx("ix_inference_dsv4_8k1k_b200_single.json")),
                              "DeepSeek-V4-Pro", "8k/1k", "2026-10-27")
    col = E._IX_COL
    by = {r[col["point_id"]]: r for r in rows}
    assert (by["440984"][col["gpu_count"]], by["440984"][col["gpu_count_upstream"]]) == ("8", "64")
    assert (by["440998"][col["gpu_count"]], by["440998"][col["gpu_count_upstream"]]) == ("8", "8")
    a, b = (float(by[k][col["tput_tok_s_chip"]]) for k in ("440984", "440998"))
    assert a == pytest.approx(b, rel=0.01), "per-GPU throughput says both run on 8 GPUs"
    assert by["440984"][col["prefill_gpus"]] == "" and by["440984"][col["decode_gpus"]] == ""
    assert E.physical_gpus(True, 16, 4, 4) == 4 and E.physical_gpus(True, 4, 2, 2) == 2
    assert E.physical_gpus(True, 8, 8, 1) == 8
    assert E.physical_gpus(False, 32, 8, 32) == 32          # multi-node EP: a real count
    assert E.physical_gpus(True, 4, 8, 1) == 4              # TPU: TP counts cores


def test_ix_old_schema_archives_conform_and_keep_first_seen(sandbox):
    """Archives written before the schema change still merge: rows are re-laid
    out by column name, gpu_count is recomputed, first_seen survives."""
    rows = []
    for fx in ("ix_inference_dsv4_8k1k_b200_single.json", "ix_inference_dsv4_8k1k.json"):
        rows += E.parse_ix_view(json.loads(_fx(fx)), "DeepSeek-V4-Pro", "8k/1k", "2026-10-20")[0]
    old_h = E.IX_POINTS_HEADER[:22]
    c = E._IX_COL
    # the old layout: 22 columns, gpu_count = upstream's label
    old = [r[:12] + [r[c["gpu_count_upstream"]]] + r[13:22] for r in rows]
    assert {r[0]: r[12] for r in old if r[0] in ("440984", "440998")} == {"440984": "64",
                                                                            "440998": "8"}
    conformed = E.conform_ix_points(old_h, old)
    by = {r[0]: r for r in conformed}
    assert all(len(r) == len(E.IX_POINTS_HEADER) for r in conformed)
    assert (by["440984"][c["gpu_count"]], by["440984"][c["gpu_count_upstream"]]) == ("8", "64")
    (sandbox / "econ").mkdir(parents=True, exist_ok=True)
    (sandbox / "econ" / "inferencex_points.csv").write_text(E._csv_text(old_h, old))
    old_rates = ["valid_from"] + E.IX_RATES_HEADER[1:]
    (sandbox / "econ" / "inferencex_rates.csv").write_text(
        E._csv_text(old_rates, E.parse_ix_options(_options(), "2026-10-27")))
    out = E.run(only=["inferencex"], now=NOW, session=FakeSession(), sleep_s=0)
    assert out["inferencex"]["status"] == "ok", out
    header, body = E._parse_csv((sandbox / "econ" / "inferencex_points.csv").read_text())
    assert header == E.IX_POINTS_HEADER and all(len(r) == len(header) for r in body)
    kept = {r[0]: r for r in body}
    assert kept["440984"][c["first_seen"]] == "2026-10-20", "points left the view are kept as-is"
    rh, rrows = E._parse_csv((sandbox / "econ" / "inferencex_rates.csv").read_text())
    assert rh == E.IX_RATES_HEADER and rh[0] == "observed_on"
    assert len(rrows) == 13, "a renamed column re-logged the marks"
