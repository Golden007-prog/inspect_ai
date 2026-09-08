"""Results-level: zero scored samples attempt-calls the metric (#5150)."""

import math

from inspect_ai._eval.task.results import ScorerInfo, scorers_from_metric_dict
from inspect_ai._util.registry import registry_unqualified_name
from inspect_ai.scorer import Score, accuracy, frequency, grouped, mean
from inspect_ai.scorer._metric import SampleScore
from inspect_ai.scorer._metrics.std import ci, ci_wilson


def _unscored(n: int = 3):
    return [
        SampleScore(score=Score(value=float("nan")), sample_metadata={"group": "A"})
        for _ in range(n)
    ]


def test_all_unscored_grouped_reports_shaped_aggregate_key():
    metric = grouped(mean(), group_key="group")
    results = scorers_from_metric_dict(
        scorer_name="test_scorer",
        scorer_info=ScorerInfo(name="test_scorer", metrics={"score": [metric]}),
        sample_scores=_unscored(),
        metrics={"score": [metric]},
    )
    metrics = results[0].metrics
    # The degenerate shape survives: aggregate key present, not a bare flat NaN row.
    all_metrics = [m for m in metrics.values() if m.name == "all"]
    assert len(all_metrics) == 1
    assert math.isnan(all_metrics[0].value)


def test_all_unscored_scalar_metric_still_flat_nan():
    metric = accuracy()
    results = scorers_from_metric_dict(
        scorer_name="test_scorer",
        scorer_info=ScorerInfo(name="test_scorer", metrics={"score": [metric]}),
        sample_scores=_unscored(),
        metrics={"score": [metric]},
    )
    metrics = results[0].metrics
    assert len(metrics) == 1
    only = next(iter(metrics.values()))
    assert math.isnan(only.value)


def test_all_unscored_list_path_grouped_reports_shaped_key():
    # Same fence on the list path: an all-unscored run must not collapse to a
    # single flat NaN row named after the metric.
    from inspect_ai._eval.task.results import scorer_for_metrics

    metric = grouped(mean(), group_key="group")
    results = scorer_for_metrics(
        scorer_name="test_scorer",
        scorer_info=ScorerInfo(name="test_scorer", metrics=None),
        sample_scores=_unscored(),
        metrics=[metric],
    )
    names = {m.name for r in results for m in r.metrics.values()}
    assert "all" in names
    for r in results:
        for m in r.metrics.values():
            assert math.isnan(m.value)


def test_all_unscored_list_path_scalar_metric_still_flat_nan():
    from inspect_ai._eval.task.results import scorer_for_metrics

    results = scorer_for_metrics(
        scorer_name="test_scorer",
        scorer_info=ScorerInfo(name="test_scorer", metrics=None),
        sample_scores=_unscored(),
        metrics=[accuracy()],
    )
    assert len(results) == 1
    assert len(results[0].metrics) == 1
    assert math.isnan(next(iter(results[0].metrics.values())).value)


def test_all_unscored_interval_metrics_report_nan_bounds():
    # #5150: a shaped metric's degenerate value reaches the log unchanged, so
    # an all-unscored run must not report `lower`/`upper` of 0.0 — that reads
    # as a measured interval pinned at zero.
    from inspect_ai._eval.task.results import scorer_for_metrics

    for interval_metric in (ci(), ci_wilson()):
        results = scorer_for_metrics(
            scorer_name="test_scorer",
            scorer_info=ScorerInfo(name="test_scorer", metrics=None),
            sample_scores=_unscored(),
            metrics=[interval_metric],
        )
        # assert the shape survived rather than the metric raising and
        # falling back to a flat NaN row, which math.isnan alone cannot tell
        # apart: the keys and the metric group must both be intact
        reported = {
            m.name: (m.value, m.group) for r in results for m in r.metrics.values()
        }
        assert set(reported) == {"lower", "upper"}
        assert {g for _, g in reported.values()} == {
            registry_unqualified_name(interval_metric)
        }
        assert all(math.isnan(v) for v, _ in reported.values())


def test_all_unscored_frequency_reports_declared_categories_as_nan():
    from inspect_ai._eval.task.results import scorer_for_metrics

    results = scorer_for_metrics(
        scorer_name="test_scorer",
        scorer_info=ScorerInfo(name="test_scorer", metrics=None),
        sample_scores=_unscored(),
        metrics=[frequency(categories=["yes", "no"])],
    )
    reported = {m.name: (m.value, m.group) for r in results for m in r.metrics.values()}
    assert set(reported) == {"yes", "no"}
    assert {g for _, g in reported.values()} == {"frequency"}
    assert all(math.isnan(v) for v, _ in reported.values())


def test_all_unscored_frequency_counts_stay_zero():
    # counts are defined at zero observations; only the proportion is not
    from inspect_ai._eval.task.results import scorer_for_metrics

    results = scorer_for_metrics(
        scorer_name="test_scorer",
        scorer_info=ScorerInfo(name="test_scorer", metrics=None),
        sample_scores=_unscored(),
        metrics=[frequency(categories=["yes", "no"], normalize=False)],
    )
    reported = {m.name: m.value for r in results for m in r.metrics.values()}
    assert reported == {"yes": 0.0, "no": 0.0}


def test_all_unscored_frequency_without_categories_is_flat_nan():
    # no declared categories means no shape to report, so the empty mapping
    # falls back to the single flat NaN row
    from inspect_ai._eval.task.results import scorer_for_metrics

    results = scorer_for_metrics(
        scorer_name="test_scorer",
        scorer_info=ScorerInfo(name="test_scorer", metrics=None),
        sample_scores=_unscored(),
        metrics=[frequency()],
    )
    assert len(results) == 1
    assert len(results[0].metrics) == 1
    assert math.isnan(next(iter(results[0].metrics.values())).value)


def test_all_unscored_shaped_metrics_on_the_dict_metric_path():
    # the list path is covered above; `metrics={"*": [frequency()]}` is the form
    # frequency()'s own docstring recommends and it goes through a different
    # branch, with its own key construction
    metrics = {"*": [ci(), ci_wilson(), frequency(categories=["yes", "no"])]}
    results = scorers_from_metric_dict(
        scorer_name="test_scorer",
        scorer_info=ScorerInfo(name="test_scorer", metrics=metrics),
        sample_scores=_unscored(),
        metrics=metrics,
    )
    # ci and ci_wilson both contribute `lower`/`upper`, so key on (group, name)
    # -- keying on the name alone silently collapses one metric into the other
    reported = {(m.group, m.name): m.value for r in results for m in r.metrics.values()}
    assert {
        ("ci", "lower"),
        ("ci", "upper"),
        ("ci_wilson", "lower"),
        ("ci_wilson", "upper"),
        ("frequency", "yes"),
        ("frequency", "no"),
    } <= set(reported)
    assert all(math.isnan(v) for v in reported.values())


def test_all_unscored_shaped_metrics_do_not_warn(caplog):
    # a metric that RAISES on empty input also ends up as NaN, via the
    # empty_metric_value fallback plus a warning. math.isnan cannot tell the two
    # apart, so assert the intended path by its silence.
    from inspect_ai._eval.task.results import scorer_for_metrics

    caplog.clear()
    scorer_for_metrics(
        scorer_name="test_scorer",
        scorer_info=ScorerInfo(name="test_scorer", metrics=None),
        sample_scores=_unscored(),
        metrics=[ci(), ci_wilson(), frequency(categories=["yes", "no"])],
    )
    assert not [r for r in caplog.records if "empty" in r.getMessage().lower()]
