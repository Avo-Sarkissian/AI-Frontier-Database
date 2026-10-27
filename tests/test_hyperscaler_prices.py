"""data/econ_archive.py's hyperscaler source: Azure Retail Prices API and AWS's
public on-demand price map -> data/raw/econ/hyperscaler_gpu_prices.csv.

Fixtures are trimmed recordings made 2026-10-27: tests/fixtures/econ/
azure_retail_gpu_page{1,2}.json (60 real items for the four mapped SKUs in
eastus / eastus2 / westus2 / brazilsouth, Spot, Low Priority and Windows
meters included, split into two pages joined by a NextPageLink) and
aws_ondemand_gpu.json (the real manifest, the four mapped p-family types plus
p4d and m5.large). No test touches the network.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

import data_guard
from data import econ_archive as E

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures" / "econ"
NOW = datetime(2026, 10, 27, 14, 0, tzinfo=timezone.utc)
# Azure's real NextPageLink form carries the port (seen live 2026-10-27).
PAGE2 = "https://prices.azure.com:443/api/retail/prices?$filter=fixture&$skip=1000"


def _load(name: str):
    return json.loads((FX / name).read_text(encoding="utf-8"))


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def json(self):
        return self.body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class Session:
    def __init__(self, aws=None, azure_fail=False):
        self.calls = []
        self.aws = aws if aws is not None else _load("aws_ondemand_gpu.json")
        self.azure_fail = azure_fail

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if url == E.AZURE_PRICES_URL:
            if self.azure_fail:
                return _Resp({}, 503)
            assert "Standard_ND96isr_H100_v5" in params["$filter"]
            return _Resp(_load("azure_retail_gpu_page1.json"))
        if url == PAGE2:
            return _Resp(_load("azure_retail_gpu_page2.json"))
        if url == E.AWS_ONDEMAND_URL:
            return _Resp(self.aws)
        raise AssertionError(f"unexpected URL {url}")


class NoNetwork:
    def get(self, *a, **k):
        raise AssertionError("a real network call was attempted")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "ECON_DIR", tmp_path / "econ")
    monkeypatch.setattr(E, "EPOCH_DIR", tmp_path / "epoch")
    monkeypatch.setattr(E, "RAW_ECON_DIR", tmp_path / "raw_econ")
    return tmp_path


def _out(tmp: Path) -> str:
    return (tmp / "raw_econ" / "hyperscaler_gpu_prices.csv").read_text()


def _rows(text: str) -> dict:
    header, rows = E._parse_csv(text)
    assert header == E.HYPERSCALER_HEADER
    return {r[1]: dict(zip(header, r)) for r in rows}


def test_azure_parsing_filters_spot_windows_lowpriority_and_follows_pages():
    s = Session()
    items = E._azure_items(s)
    assert [c[0] for c in s.calls] == [E.AZURE_PRICES_URL, PAGE2]
    assert any("Spot" in i["meterName"] for i in items)
    assert any("Windows" in i["productName"] for i in items)
    assert any("Low Priority" in i["meterName"] for i in items)
    rows, missing = E.parse_azure(items, NOW.date())
    assert missing == []
    got = {r[1]: r for r in rows}
    h100 = got["Standard_ND96isr_H100_v5"]
    assert (h100[2], h100[5], h100[6]) == ("eastus", "98.32", "12.29")
    gb200 = got["Standard_ND128isr_NDR_GB200_v6"]
    assert (gb200[4], gb200[5], gb200[6]) == ("4", "108.16", "27.04")
    # no eastus listing: the next preferred region, recorded
    assert got["Standard_ND96isr_H200_v5"][2] == "eastus2"
    assert got["Standard_ND96isr_H200_v5"][6] == "10.6"
    assert got["Standard_ND96isr_MI300X_v5"][6] == "6"


def test_a_spot_or_windows_meter_is_never_the_price():
    item = {"armSkuName": "Standard_ND96isr_H100_v5", "armRegionName": "eastus",
            "type": "Consumption", "currencyCode": "USD", "unitOfMeasure": "1 Hour",
            "retailPrice": 18.17, "effectiveStartDate": "2026-01-01T00:00:00Z",
            "meterName": "ND96isrH100v5 Spot", "skuName": "ND96isrH100v5 Spot",
            "productName": "Virtual Machines NDsr H100 v5 Series"}
    win = {**item, "meterName": "ND96isrH100v5", "skuName": "ND96isrH100v5",
           "productName": "Virtual Machines NDsr H100 v5 Series Windows", "retailPrice": 102.7}
    rows, missing = E.parse_azure([item, win], NOW.date())
    assert "Standard_ND96isr_H100_v5" in missing and not [r for r in rows if "H100" in r[1]]


def test_aws_parsing():
    rows, missing = E.parse_aws(_load("aws_ondemand_gpu.json"))
    assert missing == []
    got = {r[1]: r for r in rows}
    assert set(got) == {"p5.48xlarge", "p5en.48xlarge", "p6-b200.48xlarge", "p6-b300.48xlarge"}
    assert got["p5.48xlarge"][6] == "6.88"
    assert got["p6-b300.48xlarge"][6] == "17.802"
    assert {r[7] for r in rows} == {"2026-10-23"}         # manifest hawkFilePublicationDate
    assert all(r[2] == "us-east-1" for r in rows)


def test_every_sku_cites_its_gpu_count():
    for spec in E.HYPERSCALER_SKUS:
        assert spec["doc_url"].startswith("https://") and spec["gpus"] in (4, 8)


def test_run_writes_a_byte_stable_snapshot_and_keeps_observed_on(sandbox):
    out = E.run(only=["hyperscaler"], now=NOW, session=Session(), sleep_s=0)
    assert out["hyperscaler"]["status"] == "ok", out
    first = _out(sandbox)
    rows = _rows(first)
    assert len(rows) == 8 and {r["observed_on"] for r in rows.values()} == {"2026-10-27"}
    later = NOW + timedelta(days=7)
    E.run(only=["hyperscaler"], now=later, session=Session(), sleep_s=0, force=True)
    assert _out(sandbox) == first, "an unchanged upstream rewrote the file"
    aws = _load("aws_ondemand_gpu.json")
    for rec in aws["regions"][E.AWS_REGION_NAME].values():
        if rec["Instance Type"] == "p5.48xlarge":
            rec["price"] = "49.5360000000"
    E.run(only=["hyperscaler"], now=later, session=Session(aws=aws), sleep_s=0, force=True)
    after = _rows(_out(sandbox))
    assert after["p5.48xlarge"]["observed_on"] == later.date().isoformat()
    assert after["p5.48xlarge"]["price_per_gpu_hr"] == "6.192"
    assert after["p5en.48xlarge"]["observed_on"] == "2026-10-27"


def test_a_failure_keeps_the_old_file_and_is_recorded(sandbox):
    E.run(only=["hyperscaler"], now=NOW, session=Session(), sleep_s=0)
    before = _out(sandbox)
    out = E.run(only=["hyperscaler"], now=NOW + timedelta(days=7), sleep_s=0,
                session=Session(azure_fail=True), force=True)
    assert out["hyperscaler"]["status"] == "failed" and "503" in out["hyperscaler"]["error"]
    assert _out(sandbox) == before
    st = json.loads((sandbox / "econ" / "fetch_state.json").read_text())["hyperscaler"]
    assert st["ok"] is False and st["failing_since"]


def test_skips_when_fresh(sandbox):
    E.run(only=["hyperscaler"], now=NOW, session=Session(), sleep_s=0)
    out = E.run(only=["hyperscaler"], now=NOW + timedelta(days=5), session=NoNetwork(),
                sleep_s=0)
    assert out == {"hyperscaler": {"status": "skipped"}}


def test_units_guard_refuses_a_cents_price():
    rows, _ = E.parse_aws(_load("aws_ondemand_gpu.json"))
    rows[0][6] = "0.0688"
    assert any("outside" in p for p in E.validate_hyperscaler(rows))
    assert any("floor" in p for p in E.validate_hyperscaler(rows[:3]))


def test_publish_guard_floor_and_a_5x_jump(sandbox):
    E.run(only=["hyperscaler"], now=NOW, session=Session(), sleep_s=0)
    text = _out(sandbox)
    assert data_guard.archive_violations(E.HYPERSCALER, text, text) == []
    header, rows = E._parse_csv(text)
    jumped = [r[:6] + [str(float(r[6]) * 5)] + r[7:] for r in rows]
    bad = data_guard.archive_violations(E.HYPERSCALER, text, E._csv_text(header, jumped))
    assert any("median 'price_per_gpu_hr' moved" in b for b in bad)
    short = data_guard.archive_violations(E.HYPERSCALER, None, E._csv_text(header, rows[:5]))
    assert any("floor 6" in b for b in short)


def test_committed_snapshot_matches_the_schema():
    p = ROOT / E.HYPERSCALER
    if not p.exists():
        pytest.skip("not fetched yet")
    rows = _rows(p.read_text(encoding="utf-8"))
    assert len(rows) >= E.HYPERSCALER_MIN_ROWS
    lo, hi = E.HYPERSCALER_GPU_HR_RANGE
    assert all(lo <= float(r["price_per_gpu_hr"]) <= hi for r in rows.values())


def test_azure_next_link_accepts_only_the_retail_prices_api():
    ok = E.azure_next_link_ok
    assert ok(PAGE2)
    assert ok("https://prices.azure.com/api/retail/prices?$skip=1000")
    for bad in ("http://prices.azure.com/api/retail/prices?$skip=1000",
                "https://prices.azure.com:8443/api/retail/prices?$skip=1000",
                "https://prices.azure.com.evil.example/api/retail/prices",
                "https://evil.example/api/retail/prices?x=https://prices.azure.com",
                "https://user@prices.azure.com/api/retail/prices",
                "https://prices.azure.com/api/other",
                "https://prices.azure.com:notaport/api/retail/prices", None, ""):
        assert not ok(bad), bad
