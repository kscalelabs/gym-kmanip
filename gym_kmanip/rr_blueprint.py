from typing import List

import rerun.blueprint as rrb

import gym_kmanip as k


def make_blueprint(
    q_keys: List,
    obs_list: List[str],
    act_list: List[str],
):
    time_series_views: List[rrb.SpaceView] = []
    if "q_pos" in obs_list:
        time_series_views.append(
            rrb.TimeSeriesView(origin="/state/q_pos", name="q_pos"),
        )
    if "q_vel" in obs_list:
        time_series_views.append(
            rrb.TimeSeriesView(origin="/state/q_vel", name="q_vel"),
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
    for obs_name in obs_list:
        if "camera" in obs_name:
            cam: k.Cam = k.CAMERAS[obs_name.split("/")[-1]]
            camera_views.append(
                rrb.Spatial2DView(origin=f"/camera/{cam.name}", name=cam.name),
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
