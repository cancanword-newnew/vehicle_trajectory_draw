# 方案A本地实现：ByteTrack + TrackEval + supervision

本项目聚焦一条精简主链路：

1. 轨迹检测与可视化（视频输入 -> `tracked.mp4` + `mot_results.txt`）
2. TrackEval 指标评估（HOTA/CLEAR/Identity）
3. 生成详细 `Word` 实验报告（含参数释义与结果图）

## 目录结构

```text
trans_graph_analysis/
├─ requirements.txt
├─ README.md
├─ src/
│  ├─ tracking_pipeline.py
│  ├─ prepare_trackeval_demo.py
│  ├─ evaluate_trackeval.py
│  ├─ generate_word_report.py
│  └─ utils/
│     └─ mot_io.py
├─ tests/
│  └─ test_mot_io.py
└─ report/
   ├─ figures/
   └─ OPEN_APPLICATION_REPORT_exp2_detailed.docx
```

## 运行流程（Windows PowerShell）

```powershell
cd d:\Desktop\trans_graph_analysis
```

### 1) 轨迹检测

```powershell
D:/Anaconda/Scripts/conda.exe run -p D:\Anaconda --no-capture-output python .\src\tracking_pipeline.py --source .\runs\exp2\test_1.mp4 --output-dir .\runs\exp2_rerun --model yolo11n.pt --device cpu --conf 0.25
```

### 2) 准备 TrackEval 数据并评测

```powershell
D:/Anaconda/Scripts/conda.exe run -p D:\Anaconda --no-capture-output python .\src\prepare_trackeval_demo.py --mot-result .\runs\exp2_rerun\mot_results.txt --benchmark MOT17 --split train --seq-name EXP2-01 --tracker-name Exp2Tracker --root .\data\trackeval_exp2 --fps 30 --width 1920 --height 1080
D:/Anaconda/Scripts/conda.exe run -p D:\Anaconda --no-capture-output python .\src\evaluate_trackeval.py --trackeval-dir .\external\TrackEval --gt-folder .\data\trackeval_exp2\gt --trackers-folder .\data\trackeval_exp2\trackers --benchmark MOT17 --split train --tracker-name Exp2Tracker --do-preproc False --classes-to-eval pedestrian
```

### 3) 生成详细 Word 报告（含抽帧图）

```powershell
D:/Anaconda/Scripts/conda.exe run -p D:\Anaconda --no-capture-output python .\src\generate_word_report.py --run-meta .\runs\exp2_rerun\run_meta.json --trackeval-dir .\data\trackeval_exp2\trackers\MOT17-train\Exp2Tracker --video-path .\runs\exp2\test_1.mp4 --tracked-video .\runs\exp2_rerun\tracked.mp4 --mot-result .\runs\exp2_rerun\mot_results.txt --figures-dir .\report\figures\exp2 --num-frames 6 --output .\report\OPEN_APPLICATION_REPORT_exp2_detailed.docx
```

## 自测

```powershell
D:/Anaconda/Scripts/conda.exe run -p D:\Anaconda --no-capture-output python -m unittest discover -s .\tests -p "test_*.py"
```
