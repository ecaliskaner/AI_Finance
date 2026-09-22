from __future__ import annotations

import numpy as np
import pytest

from ai_finance.research.purged_cv import purged_splits


class TestLayout:
    def test_basic_shape(self):
        splits = purged_splits(1000, train_size=400, test_size=100, horizon=1)

        assert len(splits) == 6
        first = splits[0]
        assert first.test_positions[0] == 400
        assert first.test_positions[-1] == 499

    def test_steps_forward_by_one_test_window(self):
        splits = purged_splits(1000, train_size=400, test_size=100, horizon=1)
        assert [s.test_positions[0] for s in splits] == [400, 500, 600, 700, 800, 900]

    def test_rolling_training_window_moves(self):
        splits = purged_splits(1000, train_size=400, test_size=100, horizon=1)
        assert splits[0].train_positions[0] == 0
        assert splits[1].train_positions[0] == 100

    def test_anchored_training_window_keeps_its_start(self):
        splits = purged_splits(1000, train_size=400, test_size=100, horizon=1, anchored=True)
        assert all(s.train_positions[0] == 0 for s in splits)
        assert splits[1].n_train > splits[0].n_train

    def test_empty_when_too_short(self):
        assert purged_splits(100, train_size=400, test_size=100, horizon=1) == []

    def test_max_splits_caps_the_count(self):
        splits = purged_splits(5000, train_size=400, test_size=100, horizon=1, max_splits=3)
        assert len(splits) == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"train_size": 0, "test_size": 10, "horizon": 1},
            {"train_size": 10, "test_size": 0, "horizon": 1},
            {"train_size": 10, "test_size": 10, "horizon": 0},
            {"train_size": 10, "test_size": 10, "horizon": 1, "embargo": -1},
        ],
    )
    def test_rejects_nonsense(self, kwargs):
        with pytest.raises(ValueError):
            purged_splits(1000, **kwargs)


class TestPurging:
    """The reason this module exists rather than a plain train/test split."""

    def test_no_training_label_reaches_into_the_test_window(self):
        horizon = 12
        for split in purged_splits(2000, train_size=500, test_size=200, horizon=horizon):
            last_train = int(split.train_positions[-1])
            first_test = int(split.test_positions[0])
            # The label of the last training row closes at last_train + horizon.
            assert last_train + horizon <= first_test

    def test_embargo_widens_the_gap_further(self):
        horizon, embargo = 6, 20
        for split in purged_splits(
            2000, train_size=500, test_size=200, horizon=horizon, embargo=embargo
        ):
            last_train = int(split.train_positions[-1])
            first_test = int(split.test_positions[0])
            assert last_train + horizon + embargo <= first_test

    def test_purge_count_is_reported(self):
        split = purged_splits(1000, train_size=400, test_size=100, horizon=10, embargo=5)[0]
        assert split.purged == 15
        assert split.n_train == 400 - 15

    def test_a_longer_horizon_purges_more(self):
        short = purged_splits(1000, train_size=400, test_size=100, horizon=1)[0]
        long = purged_splits(1000, train_size=400, test_size=100, horizon=50)[0]
        assert long.n_train == short.n_train - 49

    def test_training_and_testing_never_overlap(self):
        for split in purged_splits(2000, train_size=500, test_size=200, horizon=10, embargo=5):
            assert not set(split.train_positions.tolist()) & set(split.test_positions.tolist())

    def test_positions_are_contiguous_and_ordered(self):
        for split in purged_splits(2000, train_size=500, test_size=200, horizon=10):
            assert np.all(np.diff(split.train_positions) == 1)
            assert np.all(np.diff(split.test_positions) == 1)

    def test_a_fold_with_nothing_left_to_train_on_is_skipped(self):
        """Silently training on three rows is worse than having one fewer fold."""
        splits = purged_splits(1000, train_size=50, test_size=100, horizon=60)
        assert splits == []

    def test_purging_costs_data_and_that_is_the_point(self):
        without = purged_splits(1000, train_size=400, test_size=100, horizon=1)[0]
        with_purge = purged_splits(1000, train_size=400, test_size=100, horizon=48, embargo=12)[0]

        assert with_purge.n_train < without.n_train
        assert with_purge.train_end_before_purge == without.train_end_before_purge
