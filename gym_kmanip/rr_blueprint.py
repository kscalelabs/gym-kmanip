from typing import List

import rerun as rr
import rerun.blueprint as rrb


def make_blueprint(
    q_keys: List,
    obs_list: List[str],
    act_list: List[str],
):
    time_series_views: List[rrb.SpaceView] = []
    for name in q_keys:
        time_series_views.append(
            rrb.TimeSeriesView(origin="/state/q", name=name),
        )
    if "grip_r" in obs_list:
        time_series_views.append(
            rrb.TimeSeriesView(origin="/action", name="grip_r"),
        )
    if "grip_l" in obs_list:
        time_series_views.append(
            rrb.TimeSeriesView(origin="/action", name="grip_l"),
        )
    camera_views: List[rrb.SpaceView] = []
    for name in obs_list:
        if "cam" in name:
            camera_views.append(
                rrb.Spatial2DView(origin="/world", name=name),
            )
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(
                    origin="/world",
                    name="scene",
                ),
                rrb.Horizontal(*camera_views),
            ),
            rrb.Vertical(*time_series_views),
        ),
        collapse_panels=True,
    )
    return blueprint
