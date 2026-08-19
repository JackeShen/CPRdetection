# Pose UI

A PyQt5 desktop UI for YOLO pose estimation. It uses the **local** `ultralytics`
package (this repo is installed into the `oldshen` env as an editable package) and
the local model `pose/yolo26m-pose.pt` by default.

> Note: this UI calls `ultralytics` directly inside a worker thread. The sibling
> script `pose/pose_estimate.py` is a **separate CLI** and is NOT invoked by the UI.

## Run

From the project root. For a GUI app on Windows it is most reliable to activate
the environment first, then launch:

```powershell
conda activate oldshen
python .\pose\ui\pose_ui.py
```

(Using `conda run -n oldshen python .\pose\ui\pose_ui.py` also works, but some
 Windows setups pass the GUI session less reliably through `conda run`.)

- Device is **Auto** by default (uses the GPU when available, otherwise CPU).
  You can force CPU / GPU 0 from the `Device` dropdown.
- The default model path is `pose\yolo26m-pose.pt` (ships with this folder).
  Change it via the `Model` field / Browse button if needed.

For realtime preview, use the `Labels` option:

- `No TXT`: only preview/save rendered image or video.
- `Save TXT`: save YOLO pose labels to `labels`.
- `Save label conf`: append detection confidence to each label row.

For videos, labels are saved one file per frame, for example:

```text
outputs\ui_predict\labels\video_name_000123.txt
```
