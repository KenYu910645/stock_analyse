from downloader import events


def test_normalize_roc_date_handles_slash_and_compact():
    assert events.normalize_roc_date("100/01/03") == "2011-01-03"
    assert events.normalize_roc_date("1150610") == "2026-06-10"


def test_parse_list_rows_extracts_event_and_detail_keys():
    html = """
    <table>
      <tr><td>公司代號</td><td>公司名稱</td><td>發言日期</td><td>發言時間</td><td>主旨</td><td></td></tr>
      <tr>
        <td>2330</td><td>台積電</td><td>115/06/10</td><td>13:52:56</td>
        <td>台積公司2026年5月營收報告</td>
        <td><input onclick="document.t05st01_fm.action='ajax_t05st01';document.t05st01_fm.seq_no.value='1';document.t05st01_fm.spoke_time.value='135256';document.t05st01_fm.spoke_date.value='20260610';document.t05st01_fm.co_id.value='2330';document.t05st01_fm.TYPEK.value='sii';openWindow(this.form ,'');"></td>
      </tr>
      <tr>
        <td>2317</td><td>鴻海</td><td>115/06/10</td><td>14:00:00</td>
        <td>其他公司公告</td><td></td>
      </tr>
    </table>
    """

    rows = events.parse_list_rows(html, {"2330"}, "2026-06-20T12:00:00")

    assert rows == [
        {
            "Date": "2026-06-10",
            "Time": "13:52:56",
            "Code": "2330",
            "Name": "台積電",
            "Subject": "台積公司2026年5月營收報告",
            "FactDate": "",
            "Clause": "",
            "Description": "",
            "Spokesperson": "",
            "SpokespersonTitle": "",
            "SpokespersonPhone": "",
            "Source": "MOPS",
            "SourcePath": "/mops/web/ajax_t05st01",
            "SourceMarket": "sii",
            "DetailSeqNo": "1",
            "DetailSpokeDate": "20260610",
            "DetailSpokeTime": "135256",
            "FetchedAt": "2026-06-20T12:00:00",
        }
    ]


def test_parse_list_rows_keeps_detail_keys_aligned_after_filtered_rows():
    html = """
    <table>
      <tr>
        <td>2317</td><td>鴻海</td><td>115/06/10</td><td>14:00:00</td>
        <td>其他公司公告</td>
        <td><input onclick="document.t05st01_fm.seq_no.value='8';document.t05st01_fm.spoke_time.value='140000';document.t05st01_fm.spoke_date.value='20260610';document.t05st01_fm.co_id.value='2317';document.t05st01_fm.TYPEK.value='sii';"></td>
      </tr>
      <tr>
        <td>2330</td><td>台積電</td><td>115/06/10</td><td>13:52:56</td>
        <td>台積公司2026年5月營收報告</td>
        <td><input onclick="document.t05st01_fm.seq_no.value='1';document.t05st01_fm.spoke_time.value='135256';document.t05st01_fm.spoke_date.value='20260610';document.t05st01_fm.co_id.value='2330';document.t05st01_fm.TYPEK.value='sii';"></td>
      </tr>
    </table>
    """

    rows = events.parse_list_rows(html, {"2330"}, "2026-06-20T12:00:00")

    assert len(rows) == 1
    assert rows[0]["DetailSeqNo"] == "1"
    assert rows[0]["DetailSpokeTime"] == "135256"


def test_parse_detail_row_extracts_description_fields():
    detail_html = """
    <table class="hasBorder">
      <tr><td class="tblHead">發言人</td><td>黃仁昭</td>
          <td class="tblHead">發言人職稱</td><td>資深副總經理暨財務長</td>
          <td class="tblHead">發言人電話</td><td>03-563-6688</td></tr>
      <tr><td class="tblHead">符合條款</td><td>第 20 款</td>
          <td class="tblHead">事實發生日</td><td colspan="3">115/06/10</td></tr>
      <tr><td class="tblHead">說明</td><td colspan="5"><pre>1.事實發生日:115/06/10
2.公司名稱:台積電</pre></td></tr>
    </table>
    """

    detail = events.parse_detail_row(detail_html)

    assert detail["Spokesperson"] == "黃仁昭"
    assert detail["SpokespersonTitle"] == "資深副總經理暨財務長"
    assert detail["SpokespersonPhone"] == "03-563-6688"
    assert detail["Clause"] == "第 20 款"
    assert detail["FactDate"] == "2026-06-10"
    assert "公司名稱" in detail["Description"]


def test_dedupe_rows_prefers_row_with_description():
    base = {
        "Date": "2026-06-10",
        "Time": "13:52:56",
        "Code": "2330",
        "DetailSeqNo": "1",
        "Description": "",
    }
    rich = dict(base)
    rich["Description"] = "full"

    assert events.dedupe_rows([base, rich]) == [rich]


def test_row_needs_detail_and_merge_detail():
    row = {column: "" for column in events.OUTPUT_COLUMNS}
    row.update(
        {
            "Date": "2026-06-10",
            "Time": "13:52:56",
            "Code": "2330",
            "DetailSeqNo": "1",
        }
    )

    assert events.row_needs_detail(row)
    changed = events.merge_detail(
        row,
        {
            "FactDate": "2026-06-10",
            "Clause": "20",
            "Description": "full detail",
        },
        "2026-06-20T20:00:00",
    )

    assert changed
    assert not events.row_needs_detail(row)
    assert row["Description"] == "full detail"
    assert row["FetchedAt"] == "2026-06-20T20:00:00"


def test_enrich_existing_details_updates_csv(monkeypatch, tmp_path):
    output_dir = tmp_path / "events"
    log_dir = tmp_path / "logs"
    output_dir.mkdir()
    path = output_dir / "2330_TSMC.csv"
    row = {column: "" for column in events.OUTPUT_COLUMNS}
    row.update(
        {
            "Date": "2026-06-10",
            "Time": "13:52:56",
            "Code": "2330",
            "Name": "TSMC",
            "Subject": "material event",
            "Source": "MOPS",
            "SourcePath": "/mops/web/ajax_t05st01",
            "SourceMarket": "sii",
            "DetailSeqNo": "1",
            "DetailSpokeDate": "20260610",
            "DetailSpokeTime": "135256",
            "FetchedAt": "2026-06-20T12:00:00",
        }
    )
    events.write_rows(path, [row])

    def fake_fetch_detail(session, source_row, retries=2, timeout=20):
        assert source_row["Code"] == "2330"
        return {
            "FactDate": "2026-06-10",
            "Clause": "20",
            "Description": "full detail",
            "Spokesperson": "Alice",
            "SpokespersonTitle": "CFO",
            "SpokespersonPhone": "02-1234-5678",
        }

    monkeypatch.setattr(events, "fetch_detail", fake_fetch_detail)
    monkeypatch.setattr(events.time, "sleep", lambda seconds: None)

    report = events.enrich_existing_details(
        output_dir=output_dir,
        log_dir=log_dir,
        instruments=[events.Instrument(code="2330", name="TSMC")],
        start_date=events.parse_iso_date("2026-06-01"),
        end_date=events.parse_iso_date("2026-06-20"),
        detail_sleep_min=0,
        detail_sleep_max=0,
        max_detail_rows=None,
        detail_save_every=100,
        detail_retries=2,
        detail_timeout=20,
        max_consecutive_detail_failures=20,
        detail_failure_log=log_dir / "events_detail_failures.csv",
        retry_known_detail_failures=False,
    )

    rows = events.read_existing_rows(path)
    assert report.exists()
    assert rows[0]["FactDate"] == "2026-06-10"
    assert rows[0]["Clause"] == "20"
    assert rows[0]["Description"] == "full detail"
    assert rows[0]["Spokesperson"] == "Alice"


def test_enrich_existing_details_skips_known_failures(monkeypatch, tmp_path):
    output_dir = tmp_path / "events"
    log_dir = tmp_path / "logs"
    output_dir.mkdir()
    path = output_dir / "2330_TSMC.csv"
    row = {column: "" for column in events.OUTPUT_COLUMNS}
    row.update(
        {
            "Date": "2026-06-10",
            "Time": "13:52:56",
            "Code": "2330",
            "Name": "TSMC",
            "Subject": "material event",
            "Source": "MOPS",
            "SourcePath": "/mops/web/ajax_t05st01",
            "SourceMarket": "sii",
            "DetailSeqNo": "1",
            "DetailSpokeDate": "20260610",
            "DetailSpokeTime": "135256",
            "FetchedAt": "2026-06-20T12:00:00",
        }
    )
    events.write_rows(path, [row])
    failure_log = log_dir / "events_detail_failures.csv"
    events.append_detail_failure(failure_log, row, "throttled", "2026-06-20T13:00:00")

    def fail_fetch_detail(session, source_row, retries=2, timeout=20):
        raise AssertionError("known failures should not be retried")

    monkeypatch.setattr(events, "fetch_detail", fail_fetch_detail)

    report = events.enrich_existing_details(
        output_dir=output_dir,
        log_dir=log_dir,
        instruments=[events.Instrument(code="2330", name="TSMC")],
        start_date=events.parse_iso_date("2026-06-01"),
        end_date=events.parse_iso_date("2026-06-20"),
        detail_sleep_min=0,
        detail_sleep_max=0,
        max_detail_rows=None,
        detail_save_every=100,
        detail_retries=2,
        detail_timeout=20,
        max_consecutive_detail_failures=20,
        detail_failure_log=failure_log,
        retry_known_detail_failures=False,
    )

    assert "Rows skipped from known failure log: 1" in report.read_text(encoding="utf-8")
    assert events.read_existing_rows(path)[0]["Description"] == ""


def test_enrich_existing_details_skips_malformed_dates(monkeypatch, tmp_path):
    output_dir = tmp_path / "events"
    log_dir = tmp_path / "logs"
    output_dir.mkdir()
    path = output_dir / "2330_TSMC.csv"
    malformed = {column: "" for column in events.OUTPUT_COLUMNS}
    malformed.update(
        {
            "Date": "",
            "Time": "13:52:56",
            "Code": "2330",
            "Name": "TSMC",
            "Subject": "material event",
            "Source": "MOPS",
            "SourcePath": "/mops/web/ajax_t05st01",
            "SourceMarket": "sii",
            "DetailSeqNo": "1",
            "DetailSpokeDate": "20260610",
            "DetailSpokeTime": "135256",
            "FetchedAt": "2026-06-20T12:00:00",
        }
    )
    blank_row = {column: "" for column in events.OUTPUT_COLUMNS}
    events.write_rows(path, [malformed, blank_row])

    def fail_fetch_detail(session, source_row, retries=2, timeout=20):
        raise AssertionError("malformed date rows should not be requested")

    monkeypatch.setattr(events, "fetch_detail", fail_fetch_detail)

    report = events.enrich_existing_details(
        output_dir=output_dir,
        log_dir=log_dir,
        instruments=[events.Instrument(code="2330", name="TSMC")],
        start_date=events.parse_iso_date("2026-06-01"),
        end_date=events.parse_iso_date("2026-06-20"),
        detail_sleep_min=0,
        detail_sleep_max=0,
        max_detail_rows=None,
        detail_save_every=100,
        detail_retries=2,
        detail_timeout=20,
        max_consecutive_detail_failures=20,
        detail_failure_log=log_dir / "events_detail_failures.csv",
        retry_known_detail_failures=False,
    )

    text = report.read_text(encoding="utf-8")
    assert "Rows skipped from malformed dates: 1" in text
    rows = events.read_existing_rows(path)
    assert rows[0]["Description"] == ""


def test_enrich_existing_details_sleeps_after_failure(monkeypatch, tmp_path):
    output_dir = tmp_path / "events"
    log_dir = tmp_path / "logs"
    output_dir.mkdir()
    path = output_dir / "2330_TSMC.csv"
    row = {column: "" for column in events.OUTPUT_COLUMNS}
    row.update(
        {
            "Date": "2026-06-10",
            "Time": "13:52:56",
            "Code": "2330",
            "Name": "TSMC",
            "Subject": "material event",
            "Source": "MOPS",
            "SourcePath": "/mops/web/ajax_t05st01",
            "SourceMarket": "sii",
            "DetailSeqNo": "1",
            "DetailSpokeDate": "20260610",
            "DetailSpokeTime": "135256",
            "FetchedAt": "2026-06-20T12:00:00",
        }
    )
    events.write_rows(path, [row])

    def fail_fetch_detail(session, source_row, retries=2, timeout=20):
        raise ValueError("throttled")

    sleeps = []
    monkeypatch.setattr(events, "fetch_detail", fail_fetch_detail)
    monkeypatch.setattr(events.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(events.random, "uniform", lambda start, end: 3)

    events.enrich_existing_details(
        output_dir=output_dir,
        log_dir=log_dir,
        instruments=[events.Instrument(code="2330", name="TSMC")],
        start_date=events.parse_iso_date("2026-06-01"),
        end_date=events.parse_iso_date("2026-06-20"),
        detail_sleep_min=2,
        detail_sleep_max=4,
        max_detail_rows=1,
        detail_save_every=100,
        detail_retries=2,
        detail_timeout=20,
        max_consecutive_detail_failures=20,
        detail_failure_log=log_dir / "events_detail_failures.csv",
        retry_known_detail_failures=False,
    )

    assert sleeps == [3]
