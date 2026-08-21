# spatial_map

VR 미로 행동 로그(json)와 imaging 데이터(voltage recording, suite2p)를 시간축으로
정렬하고, dF/F 를 공간(z) bin 으로 묶어 spatial map / raster-PSTH를 만드는 도구
모음입니다. 원래 `spatial_map.ipynb` 노트북에 있던 함수들을 재사용 가능한 파이썬
모듈로 정리했습니다.

## 설치

로컬에서 개발 중인 상태로 설치 (코드를 수정하면 바로 반영됨):

```bash
git clone https://github.com/<사용자명>/spatial_map.git
cd spatial_map
pip install -e .
```

또는 clone 없이 GitHub에서 바로 설치:

```bash
pip install git+https://github.com/<사용자명>/spatial_map.git
```

## 사용법

```python
from spatial_map import (
    parse_behavior_json, load_voltage_recording, load_suite2p,
    sync_json_to_vrec, find_cue_position, find_choice_position,
    build_frame_table, spatial_map, zscore, smooth, plot_raster_psth,
)

# 1) 행동 로그 파싱
frames, trials = parse_behavior_json("behavior.json")

# 2) imaging 관련 데이터 로드
vrec = load_voltage_recording("vrec.mat")
s2p = load_suite2p("suite2p.mat")

# 3) json 시계 <-> voltage recording 시계 동기화
a, b, resid = sync_json_to_vrec(trials, vrec["trial"])

# 4) cue / choice 위치, frame 테이블 구성
cue_pos = find_cue_position(frames, trials)
F = build_frame_table(frames, trials, cue_pos, vrec["frame_times"], a, b,
                       n_img_frames=s2p["sig"].shape[1])

# 5) z-score 후 spatial map 생성
Z = zscore(s2p["sig"])
valid = F["inside"].to_numpy()
maps, occ, edges, centers, ids = spatial_map(Z, F, valid)
maps = smooth(maps)

# 6) 특정 세포의 raster-PSTH 그리기
cue_of = trials.set_index("iTrial").cue.reindex(ids).to_numpy()
fig, ax_raster, ax_psth = plot_raster_psth(maps, centers, cue_of, cell=4)
```

## 모듈 구성

- `spatial_map/io.py` — 행동 json, voltage recording(.mat), suite2p(.mat) 로더
- `spatial_map/sync.py` — json ↔ voltage recording 시계 동기화 (`sync_json_to_vrec`)
- `spatial_map/events.py` — trial 별 cue/choice 좌표 추출
- `spatial_map/frames.py` — imaging frame 단위 테이블 구성 (`build_frame_table`)
- `spatial_map/maps.py` — spatial map 생성, z-score, smoothing
- `spatial_map/viz.py` — raster-PSTH 시각화

## 참고

- `ephys.ophys` (`Ophys`, `VRec`, `Suite2p`) 는 pip으로 배포되지 않는 개인용 로컬
  모듈입니다. `spatial_map/__init__.py`에서 있으면 불러오고 없으면 조용히
  넘어가도록 처리해두었으니, 다른 환경에 설치할 때 없어도 나머지 함수는 정상
  동작합니다.
