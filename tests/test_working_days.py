from datetime import date

from app.tools.working_days import choose_calendar, count_working_days, working_dates


def test_weekdays_only():
    # Mon 2026-09-14 .. Fri 2026-09-18
    assert count_working_days(date(2026, 9, 14), date(2026, 9, 18), []) == 5


def test_weekend_only_range_is_zero():
    assert count_working_days(date(2026, 9, 19), date(2026, 9, 20), []) == 0


def test_range_spanning_weekend():
    # Thu..Tue spans Sat/Sun: Thu, Fri, Mon, Tue
    assert count_working_days(date(2026, 9, 17), date(2026, 9, 22), []) == 4


def test_holiday_on_weekday_is_excluded():
    assert count_working_days(date(2026, 9, 14), date(2026, 9, 18), [date(2026, 9, 16)]) == 4


def test_holiday_on_weekend_changes_nothing():
    assert count_working_days(date(2026, 9, 14), date(2026, 9, 20), [date(2026, 9, 19)]) == 5


def test_range_spanning_weekend_and_holiday():
    days = working_dates(date(2026, 9, 18), date(2026, 9, 22), [date(2026, 9, 21)])
    assert days == [date(2026, 9, 18), date(2026, 9, 22)]


def test_single_day_and_holiday_only():
    assert count_working_days(date(2026, 9, 21), date(2026, 9, 21), [date(2026, 9, 21)]) == 0
    assert count_working_days(date(2026, 9, 22), date(2026, 9, 22), []) == 1


def test_calendar_choice_uses_region_and_year(db, corpus):
    with db.read() as conn:
        india = choose_calendar(conn, corpus.docs.values(), "employee", "IN", 2026, date(2026, 9, 15))
        germany = choose_calendar(conn, corpus.docs.values(), "employee", "DE", 2026, date(2026, 9, 15))
        us = choose_calendar(conn, corpus.docs.values(), "employee", "US", 2026, date(2026, 9, 15))
    assert india.doc_ids == ("CAL-IN-2026",)  # the 2025 calendar has no 2026 dates
    assert date(2026, 9, 21) in india.holidays
    assert date(2026, 9, 23) not in india.holidays  # restricted/optional holidays are not closures
    assert germany.doc_ids == ("CAL-DE-2026",)
    assert date(2026, 9, 21) not in germany.holidays
    assert us.doc_ids == () and not us.holidays  # no calendar for this region


def test_calendar_not_yet_effective_is_not_used(db, corpus):
    with db.read() as conn:
        choice = choose_calendar(conn, corpus.docs.values(), "employee", "IN", 2026, date(2025, 12, 31))
    assert choice.doc_ids == ()
