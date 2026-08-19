# Pose Estimation

This folder contains a standalone pose-estimation script using the local model:

```powershell
python .\pose_estimate.py --source C:\path\to\image.jpg
python .\pose_estimate.py --source C:\path\to\video.mp4 --stream
python .\pose_estimate.py --source 0 --show --stream
```

Default model:

```text
.\yolo26m-pose.pt
```

Annotated images/videos are saved to:

```text
.\outputs\predict
```

Common options:

```powershell
python .\pose_estimate.py --source input.mp4 --conf 0.35 --imgsz 960 --device 0 --stream
python .\pose_estimate.py --source input.jpg --save-txt --save-conf
```
