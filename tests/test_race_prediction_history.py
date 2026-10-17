"""Regression coverage for historical prediction extraction and repeat imports."""

import json
from datetime import date
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from garmin_health_data.constants import GARMIN_DATA_REGISTRY, APIMethodTimeParam
from garmin_health_data.extractor import GarminExtractor
from garmin_health_data.garmin_client.api import get_race_prediction_history
from garmin_health_data.models import RacePredictions, User
from garmin_health_data.processor import GarminProcessor
from garmin_health_data.processor_helpers import FileSet


def test_history_chunks_without_gaps_and_uses_daily_endpoint():
    client = MagicMock(display_name="test-user")
    client._connectapi.side_effect = [
        [{"calendarDate": "2024-01-01", "time5K": 1500}],
        [{"calendarDate": "2025-01-01", "time5K": 1400}],
        [],
    ]
    result = get_race_prediction_history(client, "2024-01-01", "2026-01-02")
    assert len(result) == 2
    calls = client._connectapi.call_args_list
    assert [c.kwargs["params"] for c in calls] == [
        {"fromCalendarDate": "2024-01-01", "toCalendarDate": "2024-12-31"},
        {"fromCalendarDate": "2025-01-01", "toCalendarDate": "2026-01-01"},
        {"fromCalendarDate": "2026-01-02", "toCalendarDate": "2026-01-02"},
    ]
    assert all("/daily/" in c.args[0] for c in calls)


@pytest.mark.parametrize("response", [None, {}, [None], [{"calendarDate": "bad"}]])
def test_malformed_history_fails_instead_of_appearing_complete(response):
    client = MagicMock()
    client._connectapi.return_value = response
    with pytest.raises((ValueError, TypeError)):
        get_race_prediction_history(client, "2026-01-01", "2026-01-02")


def test_history_keeps_requested_dates_and_latest_duplicate():
    client = MagicMock()
    client._connectapi.return_value = [
        {"calendarDate": "2026-01-02", "time5K": 1500},
        {"calendarDate": "2026-01-01", "time5K": 1600},
        {"calendarDate": "2026-01-02", "time5K": 1400},
        {"calendarDate": "2026-01-03", "time5K": 1300},
    ]
    rows = get_race_prediction_history(client, "2026-01-01", "2026-01-02")
    assert [r["time5K"] for r in rows] == [1600, 1400]
    with pytest.raises(ValueError):
        get_race_prediction_history(client, "2026-01-02", "2026-01-01")


def test_extractor_routes_history_to_dated_legacy_compatible_files(tmp_path):
    extractor = GarminExtractor(
        date(2026, 1, 1), date(2026, 1, 3), tmp_path, ["RACE_PREDICTIONS"]
    )
    extractor.user_id = "123"
    extractor.garmin_client = MagicMock()
    rows = [
        {"calendarDate": "2026-01-01", "time5K": 1500},
        {"calendarDate": "2026-01-03", "time5K": 1400},
    ]
    extractor.garmin_client.get_race_prediction_history.return_value = rows
    kind = GARMIN_DATA_REGISTRY.get_by_name("RACE_PREDICTIONS")
    assert kind.api_method_time_param == APIMethodTimeParam.RANGE
    files = extractor._extract_data_by_type(kind, date(2026, 1, 1), date(2026, 1, 3))
    extractor.garmin_client.get_race_prediction_history.assert_called_once_with(
        "2026-01-01", "2026-01-03"
    )
    assert len(files) == 2
    assert [json.loads(f.read_text()) for f in files] == rows
    assert all(row["calendarDate"] in f.name for row, f in zip(rows, files))


def test_reimports_refresh_values_preserve_creation_and_keep_newest_latest(
    db_session, tmp_path
):
    db_session.add_all(
        [User(user_id=1, full_name="Test"), User(user_id=2, full_name="Other")]
    )
    db_session.flush()
    other = RacePredictions(user_id=2, date=date(2026, 1, 1), time_5k=1800, latest=True)
    db_session.add(other)
    db_session.flush()
    proc = GarminProcessor(FileSet(file_paths=[], files={}), db_session)
    proc.user_id = 1
    path = tmp_path / "prediction.json"

    def process(day, value):
        path.write_text(json.dumps({"calendarDate": day, "time5K": value}))
        proc._process_race_predictions(path, db_session)
        db_session.flush()
        db_session.expire_all()

    process("2026-01-03", 1500)
    first = db_session.get(RacePredictions, (1, date(2026, 1, 3))).create_ts
    process("2026-01-01", 1600)
    process("2026-01-03", 1400)
    process("2026-01-01", 1550)
    rows = db_session.scalars(
        select(RacePredictions)
        .where(RacePredictions.user_id == 1)
        .order_by(RacePredictions.date)
    ).all()
    assert [(r.time_5k, r.latest) for r in rows] == [(1550, False), (1400, True)]
    assert rows[1].create_ts == first
    assert db_session.get(RacePredictions, (2, date(2026, 1, 1))).latest
