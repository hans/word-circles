import math

from word_circles.roseg import grid, on_grid_edge, paired_no_threshold, select_config, select_no_threshold


def row(param_id, prominence, theta, window, R, n_empty=0):
    return {"param_id": param_id, "prominence": prominence, "threshold_max_angle": theta,
            "smooth_window_size": window, "R": R, "n_empty": n_empty}


ROWS = [
    row(0, 0.0, math.pi / 2, 1, 0.70),
    row(1, 0.0, math.pi, 1, 0.80),               # best overall, but no threshold: not selectable
    row(2, 0.175, math.pi / 2, 1, 0.75, n_empty=3),  # best thresholded, but leaves utterances empty
    row(3, 0.175, math.pi / 2, 2, 0.72),
    row(4, 0.175, math.pi, 2, 0.60),
    row(5, 0.349, math.pi / 4, 1, 0.72),         # ties param 3 on R, later in loop order
    row(6, 0.349, math.pi, 1, 0.65),
]


def test_selection_takes_best_thresholded_config_without_empty_utterances_first_in_loop_order():
    assert select_config(ROWS)["param_id"] == 3


def test_no_threshold_ablation_is_the_selected_configs_theta_pi_partner():
    assert paired_no_threshold(ROWS, select_config(ROWS))["param_id"] == 4


def test_retuned_no_threshold_row_searches_only_the_theta_pi_slice():
    assert select_no_threshold(ROWS)["param_id"] == 1


def test_authors_grid_is_their_300_configs_in_their_loop_order():
    configs = grid()
    assert len(configs) == 300
    assert [c["param_id"] for c in configs] == list(range(300))
    assert configs[0] == {"param_id": 0, "prominence": 0.0, "threshold_max_angle": math.pi / 4, "smooth_window_size": 1}
    assert configs[1]["smooth_window_size"] == 2 and configs[3]["threshold_max_angle"] > math.pi / 4
    assert configs[-1]["prominence"] == 0.785 and configs[-1]["threshold_max_angle"] == math.pi
    # LEACE's shipped config p=0.175 / w=3 / theta=pi/4 is on the grid.
    assert any(c["prominence"] == 0.175 and c["smooth_window_size"] == 3
               and c["threshold_max_angle"] == math.pi / 4 for c in configs)


def test_selection_on_lowest_theta_or_extreme_prominence_is_a_grid_edge():
    configs = grid()
    assert on_grid_edge({"prominence": 0.175, "threshold_max_angle": math.pi / 4}, configs)
    assert on_grid_edge({"prominence": 0.785, "threshold_max_angle": math.pi / 2}, configs)
    assert not on_grid_edge({"prominence": 0.349, "threshold_max_angle": math.pi / 3}, configs)
