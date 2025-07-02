#!/usr/bin/env python3
"""
Optimized STL to OpenSCAD voxel merger with progress bars and debug coloring
"""

import argparse
import numpy as np
import trimesh
import sys
import random
from tqdm import tqdm

def voxelize_with_progress(mesh, resolution):
    """Voxelize mesh with progress reporting"""
    bbox = mesh.bounds
    dims = np.ceil((bbox[1] - bbox[0]) / resolution).astype(int)
    voxels = np.zeros(dims, dtype=bool)
    origin = bbox[0]
    
    x = np.linspace(origin[0] + resolution/2, origin[0] + (dims[0]-0.5)*resolution, dims[0])
    y = np.linspace(origin[1] + resolution/2, origin[1] + (dims[1]-0.5)*resolution, dims[1])
    z = np.linspace(origin[2] + resolution/2, origin[2] + (dims[2]-0.5)*resolution, dims[2])
    
    print("Voxelizing...")
    with tqdm(total=dims[2], desc="Voxelizing", unit="layer") as pbar:
        for zi in range(dims[2]):
            grid = np.meshgrid(x, y, z[zi], indexing='ij')
            points = np.vstack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()]).T
            contains = mesh.contains(points)
            voxels[:, :, zi] = contains.reshape(dims[0], dims[1])
            pbar.update(1)
    
    return voxels, origin

def greedy_merge(voxels):
    """Greedy merge filled voxels in X/Y/Z directions"""
    visited = np.zeros_like(voxels, dtype=bool)
    cubes = []
    total_slices = voxels.shape[2]

    for z in tqdm(range(total_slices), desc="Merging voxels", unit="layer"):
        for y in range(voxels.shape[1]):
            x = 0
            while x < voxels.shape[0]:
                if voxels[x, y, z] and not visited[x, y, z]:
                    # Expand in X direction
                    x_end = x + 1
                    while x_end < voxels.shape[0] and voxels[x_end, y, z] and not visited[x_end, y, z]:
                        x_end += 1

                    # Expand in Y direction
                    y_end = y + 1
                    while y_end < voxels.shape[1] and np.all(voxels[x:x_end, y_end, z] & ~visited[x:x_end, y_end, z]):
                        y_end += 1

                    # Expand in Z direction
                    z_end = z + 1
                    while z_end < voxels.shape[2] and np.all(voxels[x:x_end, y:y_end, z_end] & ~visited[x:x_end, y:y_end, z_end]):
                        z_end += 1

                    visited[x:x_end, y:y_end, z:z_end] = True
                    cubes.append(((x, y, z), (x_end, y_end, z_end)))
                    x = x_end
                else:
                    x += 1

    return cubes

def write_scad(cubes, origin, resolution, out_path, debug=False):
    """Write OpenSCAD file from merged cubes with optional debug coloring"""
    with open(out_path, 'w') as f:
        f.write("// Voxel-merged STL -> OpenSCAD\n")
        
        if debug:
            f.write("// Debug mode with random colors\n")
            f.write("module colored_cube(size, color) {\n")
            f.write("    color(color) cube(size);\n")
            f.write("}\n\n")
        
        f.write("union() {\n")
        
        for (x0, y0, z0), (x1, y1, z1) in cubes:
            pos = origin + np.array([x0, y0, z0]) * resolution
            size = (np.array([x1 - x0, y1 - y0, z1 - z0])) * resolution
            
            if debug:
                # Generate random RGB color
                color = [random.random() for _ in range(3)]
                f.write(f"  translate([{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])\n")
                f.write(f"    colored_cube([{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}], [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]);\n")
            else:
                f.write(f"  translate([{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])\n")
                f.write(f"    cube([{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]);\n")
        
        f.write("}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert STL to OpenSCAD via voxel merging")
    parser.add_argument("input", help="Input STL file")
    parser.add_argument("output", help="Output SCAD file")
    parser.add_argument("-r", "--resolution", type=float, default=1.0,
                       help="Voxel size in mm (default: 1.0)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with random colors for each cube")
    args = parser.parse_args()

    print(f"Loading mesh: {args.input}")
    mesh = trimesh.load(args.input, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()
    mesh.process(validate=True)

    voxels, origin = voxelize_with_progress(mesh, args.resolution)
    print(f"Voxel grid shape: {voxels.shape}")

    cubes = greedy_merge(voxels)
    print(f"Total merged cubes: {len(cubes)}")

    print(f"Writing to OpenSCAD: {args.output}")
    write_scad(cubes, origin, args.resolution, args.output, debug=args.debug)
    print("Finished.")

if __name__ == "__main__":
    main()