# From YouTube to Motion Classifier: a practical MoCap tutorial

> Extract human motion from any video and classify gestures, with no marker suit and no special hardware.

<!-- MoCap image coming soon -->

Four notebooks take you from a YouTube search query to a trained gesture classifier.
The example uses tennis strokes but you can adapt it to any sport or movement in a few minutes.

```
YouTube videos  →  monocular MoCap  →  pose features  →  classifier
   (notebook 01)      (02 / 03)            (03)              (04)
```

---

## Notebooks

### 01 — Download videos (`01_download_videos.ipynb`)

Uses yt-dlp to search YouTube and download videos automatically. You just give it a list of search queries, it handles the rest: filtering by duration, deduplication, downloading at 480p, saving a `metadata.json`.

Change `SEARCH_QUERIES` at the top to match your movement of interest.

### 02 — Visualisation (`02_mediapipe_pose.ipynb`)

Just exploration: skeleton overlay on video frames, 3D joint trajectories, stride cycle detection from ankle height. Good to run first to check your videos are working and the pose extraction looks reasonable.

### 03 — Pose → SMPL (`03_pose_to_smpl.ipynb`)

This is version 1 of the MoCap step — no GPU needed.

MediaPipe gives you 33 body landmarks per frame (positions in 3D). The notebook converts those into SMPL body parameters (joint rotations) by computing the angle between each bone's observed direction and its T-pose reference. It's a rough approximation but it works fine for classification.

Output is saved as `hmr4d_results.pt`, the same format as GVHMR, so you can swap one for the other without touching the rest of the pipeline.

| | MediaPipe (v1) | GVHMR (v2) |
|---|---|---|
| GPU | no | yes (8+ GB) |
| Install | `pip install mediapipe` | conda env |
| Accuracy | approximate | much better |
| Speed | real-time | ~2 fps |

### 04 — Classification (`04_classification.ipynb`)

Loads the SMPL files from notebook 03 and trains a Conv1D to distinguish gesture classes.
Input is the `body_pose` sequence `(T, 69)` directly — no hand-crafted features.
Outputs a learning curve and a confusion matrix.

---

## Install

```bash
pip install yt-dlp mediapipe opencv-python-headless matplotlib scipy torch scikit-learn
```

Then run the notebooks in order. Everything works on CPU, Google Colab is fine.

---

## Changing the movement class

The main things to edit:

- `SEARCH_QUERIES` in notebook 01 (or `CLASSES` in notebook 04) — your YouTube search terms
- `MAX_DURATION_SEC` — skip long videos
- `SEQ_LEN` in notebook 04 — how many frames the LSTM sees
- `SMPL_BONES` in notebooks 02/04 — which joints to track (useful if some are occluded a lot)

---

## Version 2 — GVHMR

If you have a GPU, [GVHMR](https://github.com/zju3dv/GVHMR) (SIGGRAPH Asia 2024) gives much better pose estimates than the geometric approximation in notebook 02. It outputs the same `hmr4d_results.pt` format so you can plug it straight into notebook 04.
