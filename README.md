# STILL BETA : stl2openscad
Some python scripts to decompose stl to openscad primitives


# STL to OpenSCAD Voxel Converter

A Python script that converts STL 3D models into OpenSCAD code using voxelization and greedy merging, with optional debug coloring.

## Features

- Converts STL files to OpenSCAD cube structures
- Adjustable voxel resolution
- Greedy merging algorithm reduces cube count
- Progress bars for voxelization and merging
- Debug mode with random colors for visualization
- Handles large files efficiently

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/stl2openscad.git
cd stl2openscad
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Usage
```
python stl2openscad.py input.stl output.scad --resolution 1.0  --debug
```

3.1 Parameters
-r, --resolution	: (Voxel size in mm, default = 1.0)
--debug	: Enable colored debug mode	(default False)

## Third-party components and their licenses:

1. Drububu's Voxelizer techniques - MIT License
2. PyVista voxelization methods - MIT License
3. CuPy GPU acceleration - MIT License
4. PyOpenCL implementation - Apache License 2.0
