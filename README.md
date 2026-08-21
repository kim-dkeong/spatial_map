# spatial_map

```python
from spatial_map import (
    parse_behavior_json, load_voltage_recording, load_suite2p,
    sync_json_to_vrec, find_cue_position, find_choice_position,
    build_frame_table, spatial_map, zscore, smooth, plot_raster_psth,
)

# 1) parsing behavior (VR log file .json)
frames, trials = parse_behavior_json("behavior.json")

# 2) load data (2p imaging)
vrec = load_voltage_recording("vrec.mat")
s2p = load_suite2p("suite2p.mat")

# 3) sync json to voltage recording in 2p
a, b, resid = sync_json_to_vrec(trials, vrec["trial"])

# 4) cue / choice location,build frame table
cue_pos = find_cue_position(frames, trials)
F = build_frame_table(frames, trials, cue_pos, vrec["frame_times"], a, b,
                       n_img_frames=s2p["sig"].shape[1])

# 5) z-score -> spatial map 
Z = zscore(s2p["sig"])
valid = F["inside"].to_numpy()
maps, occ, edges, centers, ids = spatial_map(Z, F, valid)
maps = smooth(maps)

# 6) raster-PSTH
cue_of = trials.set_index("iTrial").cue.reindex(ids).to_numpy()
fig, ax_raster, ax_psth = plot_raster_psth(maps, centers, cue_of, cell=4)
```

## Composition

- `spatial_map/io.py` — json, voltage recording(.mat), suite2p(.mat) loader
- `spatial_map/sync.py` — json ↔ voltage recording sync (`sync_json_to_vrec`)
- `spatial_map/events.py` — extract cue/choice per trials
- `spatial_map/frames.py` — build frame table
- `spatial_map/maps.py` — spatial map, z-score, smoothing
- `spatial_map/viz.py` — raster-PSTH visualization

