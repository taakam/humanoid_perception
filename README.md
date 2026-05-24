# Humanoid Perception

A modular spatial AI pipeline for reconstructing semantically-aware 3D environments from monocular video.

The project is driven by a single pipeline script:

```bash
python pipeline.py --video <path_to_video> --name <run_name>
```

The pipeline combines:

* Frame extraction from monocular video
* COLMAP camera pose estimation
* Gaussian Splatting reconstruction
* YOLO-World semantic object detection
* Multi-frame semantic fusion
* 3D semantic association
* Scene graph generation
* Browser-based Gaussian scene visualization

The system is designed with downstream robotics and embodied AI applications in mind.

---

# Features

* End-to-end video-to-semantic-3D pipeline
* COLMAP-based camera pose estimation
* Gaussian Splatting scene reconstruction
* YOLO-World indoor semantic object detection
* Affordance tagging for detected objects
* Multi-frame object fusion
* 3D semantic object association
* Scene graph export as JSON
* Interactive browser viewer for Gaussian reconstruction and semantic data
* Reproducible setup through `setup.sh`

---

# Pipeline Overview

```text
Input Video
    ↓
Frame Extraction
    ↓
2D Semantic Detection
    ↓
COLMAP Sparse Reconstruction
    ↓
Dataset Undistortion
    ↓
Gaussian Splatting Training
    ↓
Gaussian Rendering
    ↓
Semantic Fusion
    ↓
3D Semantic Association
    ↓
Scene Graph Generation
    ↓
Viewer Data Export
```

---

# Repository Structure

```text
humanoid_perception/
│
├── pipeline.py
├── setup.sh
├── requirements.txt
├── scripts/
│   ├── extract_frames.py
│   ├── run_colmap.py
│   ├── undistort.py
│   ├── train_gaussians.py
│   ├── render_scene.py
│   ├── semantic_detect.py
│   ├── fuse_semantics.py
│   ├── associate_3d.py
│   └── build_scene_graph.py
│
├── viewer/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   └── data/
│
├── weights/          # downloaded by setup.sh, ignored by git
├── third_party/      # Gaussian Splatting repo, ignored by git
├── runs/             # generated outputs, ignored by git
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/taakam/humanoid_perception.git
cd humanoid_perception
```

Create and activate a Python environment:

```bash
python -m venv venv
source venv/bin/activate
```

Run setup:

```bash
chmod +x setup.sh
./setup.sh
```

The setup script installs:

* System dependencies such as FFmpeg and COLMAP
* PyTorch with CUDA 12.8
* Python requirements
* YOLO-World model weights into `weights/`
* Gaussian Splatting repository and CUDA extensions
* Node.js viewer dependencies

Large model weights and third-party dependencies are not committed to the repository.

---

# Usage

Run the complete pipeline:

```bash
python pipeline.py \
  --video videos/room.mp4 \
  --name demo_room \
  --fps 2 \
  --custom-classes "slippers,napkin,curtain"
```

This creates:

```text
runs/demo_room/
├── frames/
├── colmap/
├── dataset_undistorted/
├── gs_output/
└── semantics/
    ├── frame_detections.json
    ├── frame_semantic_objects.json
    ├── semantic_objects_3d.json
    ├── scene_graph.json
    └── visualizations/
```

The pipeline also exports viewer-ready files to:

```text
viewer/data/
├── point_cloud.ply
├── semantic_objects_3d.json
└── scene_graph.json
```

---

# Useful Pipeline Flags

Skip Gaussian training:

```bash
python pipeline.py --video videos/room.mp4 --name demo_room --skip-training
```

Skip rendering:

```bash
python pipeline.py --video videos/room.mp4 --name demo_room --skip-render
```

Skip semantic stages:

```bash
python pipeline.py --video videos/room.mp4 --name demo_room --skip-semantics
```

Change semantic confidence:

```bash
python pipeline.py \
  --video videos/room.mp4 \
  --name demo_room \
  --semantic-conf 0.35
```

Add custom semantic classes:

```bash
python pipeline.py \
  --video videos/room.mp4 \
  --name demo_room \
  --custom-classes "slippers,napkin,curtain"
```

---

# Viewer

After the pipeline finishes, launch the interactive viewer:

```bash
cd viewer
npm run dev
```

Then open the local Vite URL printed in the terminal.

The viewer loads:

* Gaussian Splatting point cloud
* semantic object data
* scene graph data

---

# Semantic Scene Graph

The pipeline exports a scene graph in JSON format:

```text
runs/<run_name>/semantics/scene_graph.json
```

Example object node:

```json
{
  "id": "laptop_001",
  "type": "object",
  "label": "laptop",
  "affordances": [
    "electronic_object",
    "fragile",
    "movable"
  ]
}
```

Example relationships include:

* `contains`
* `near`
* `above`
* `below`
* `has_affordance`

---

# Affordance Reasoning

Detected objects are automatically assigned affordance labels such as:

* `graspable`
* `movable`
* `traversable`
* `support_surface`
* `obstacle`
* `fragile`
* `electronic_object`
* `openable`
* `deformable`

This is intended to support downstream robotic reasoning and embodied AI interaction.

---

# Outputs

Important generated outputs include:

```text
runs/<run_name>/gs_output/point_cloud/iteration_30000/point_cloud.ply
runs/<run_name>/semantics/semantic_objects_3d.json
runs/<run_name>/semantics/scene_graph.json
viewer/data/point_cloud.ply
viewer/data/semantic_objects_3d.json
viewer/data/scene_graph.json
```

---

# Large Assets

Large files such as:

* raw input videos
* Gaussian Splatting outputs
* full reconstruction videos
* large point clouds
* trained weights

are not stored directly in GitHub.

External asset links can be added here:

```text
Google Drive / Hugging Face / Dropbox link
```

---

# Design Choices and Tradeoffs

## Why Gaussian Splatting?

Gaussian Splatting was chosen because it provides:

* significantly faster rendering than NeRF-style volumetric methods,
* interactive visualization capabilities,
* high-quality geometry and appearance reconstruction,
* compatibility with downstream spatial AI applications.

The tradeoff is that Gaussian Splatting is primarily a rendering-oriented representation rather than a physics-aware world model.

---

## Why YOLO-World?

YOLO-World enables flexible open-vocabulary object detection while remaining lightweight and easy to integrate into a modular pipeline.

The tradeoff is that purely 2D detections can produce noisy associations when projected into sparse 3D geometry.

---

## Why Monocular Video?

The pipeline intentionally uses monocular video only in order to:

* reduce hardware requirements,
* improve accessibility,
* simplify data collection.

The tradeoff is reduced geometric robustness compared to RGB-D or multi-camera systems.

---

## Why Semantic Scene Graphs?

The semantic scene graph provides a higher-level structured representation of the environment that can support:

* robotic reasoning,
* embodied AI interaction,
* navigation,
* affordance-aware planning.

Rather than storing only geometry, the system attempts to encode semantic relationships between objects.

---

## Design Philosophy

The project prioritizes:

* modularity,
* readability,
* reproducibility,
* practical experimentation.

The goal was not to build a state-of-the-art reconstruction method, but rather a clean and extensible spatial AI pipeline that integrates geometry, semantics, and interactive visualization.

---

# Notes and Limitations

* Reconstruction quality depends heavily on input video quality, camera motion, lighting, and scene texture.
* Sparse COLMAP points may be too limited for perfect 3D semantic association in low-texture indoor scenes.
* The semantic 3D association is a baseline and can be improved with segmentation masks, rendered depth, or Gaussian-aware association.
* Gaussian Splatting is used primarily as a visual/spatial representation, not a physics-ready simulation environment.

---

# Future Work

* Better semantic-to-3D association using dense Gaussian geometry
* Segmentation-mask-based object fusion
* Interactive object selection in the viewer
* Semantic scene graph querying
* Collision proxy extraction for robotics simulators
* Integration with Isaac Sim / Isaac Lab control policies
* Real-time SLAM-style updates

---

# Technologies

* Python
* PyTorch
* COLMAP
* Gaussian Splatting
* OpenCV
* YOLO-World
* Three.js / Vite
* Node.js

---

# License

MIT License
