#!/usr/bin/env python3
"""
Enhanced STL to OpenSCAD converter with primitive detection
Adapted from mMerlin/stl2scad with added primitive recognition
"""

import numpy as np
from stl import mesh
import argparse
from collections import defaultdict
from scipy.optimize import least_squares
from scipy.spatial import KDTree
import math

def main():
    parser = argparse.ArgumentParser(description='Convert STL to OpenSCAD with primitive detection')
    parser.add_argument('input', help='Input STL file')
    parser.add_argument('output', help='Output SCAD file')
    parser.add_argument('--tolerance', type=float, default=0.1,
                       help='Tolerance for primitive detection (default: 0.1mm)')
    parser.add_argument('--min-size', type=float, default=1.0,
                       help='Minimum feature size to detect as primitive (default: 1.0mm)')
    parser.add_argument('--simplify', action='store_true',
                       help='Attempt to simplify using primitives')
    args = parser.parse_args()

    # Load STL file
    stl_mesh = mesh.Mesh.from_file(args.input)
    vertices = stl_mesh.vectors.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)

    with open(args.output, 'w') as scad_file:
        scad_file.write(f"// Generated from {args.input}\n")
        scad_file.write("$fn = 64;\n\n")

        if args.simplify:
            # Advanced conversion with primitive detection
            primitives = detect_primitives(vertices, faces, args.tolerance, args.min_size)
            generate_primitive_scad(primitives, scad_file)
            
            # Calculate and show simplification stats
            original_tris = len(faces)
            converted_tris = sum(len(p['faces']) for p in primitives)
            scad_file.write(f"\n/*\nConversion statistics:\n")
            scad_file.write(f"Original triangles: {original_tris}\n")
            scad_file.write(f"Converted to {len(primitives)} primitives ")
            scad_file.write(f"(covering {converted_tris} triangles)\n")
            scad_file.write(f"Simplification ratio: {converted_tris/original_tris:.1%}\n*/\n")
        else:
            # Basic conversion to polyhedron
            generate_polyhedron_scad(vertices, faces, scad_file)

def detect_primitives(vertices, faces, tolerance=0.1, min_feature_size=1.0):
    """Main primitive detection function"""
    primitives = []
    face_used = np.zeros(len(faces), dtype=bool)
    kd_tree = KDTree(vertices)
    
    # Cluster connected faces
    face_groups = cluster_connected_faces(vertices, faces, tolerance)
    
    for group in face_groups:
        if len(group) < 2:  # Skip single triangles
            continue
            
        group_faces = faces[group]
        group_vertices = np.unique(group_faces.flatten())
        group_points = vertices[group_vertices]
        
        # Try detecting different primitives
        detected = False
        
        # 1. Check for planar surfaces (potential cubes)
        if is_planar(group_faces, vertices, tolerance):
            cuboids = detect_cuboid(group_faces, vertices, tolerance, min_feature_size)
            if cuboids:
                primitives.extend(cuboids)
                face_used[group] = True
                detected = True
        
        # 2. Check for cylinders
        if not detected:
            cylinders = detect_cylinder(group_faces, vertices, tolerance, min_feature_size)
            if cylinders:
                primitives.extend(cylinders)
                face_used[group] = True
                detected = True
                
        # 3. Check for spheres
        if not detected:
            spheres = detect_sphere(group_faces, vertices, tolerance, min_feature_size)
            if spheres:
                primitives.extend(spheres)
                face_used[group] = True
                detected = True
    
    # Add remaining faces as polyhedron
    remaining_faces = faces[~face_used]
    if len(remaining_faces) > 0:
        primitives.append({
            'type': 'polyhedron',
            'vertices': vertices,
            'faces': remaining_faces
        })
    
    return primitives

# [Previous helper functions (is_planar, detect_cuboid, etc.) would be included here]
# ... (Include all the helper functions from the previous implementation)

def generate_primitive_scad(primitives, scad_file):
    """Generate OpenSCAD code from detected primitives"""
    for prim in primitives:
        if prim['type'] == 'cube':
            scad_file.write(f"translate({list(prim['position'])}) ")
            scad_file.write(f"cube({list(prim['dimensions'])});\n")
            
        elif prim['type'] == 'cylinder':
            scad_file.write(f"translate({list(prim['center'])}) ")
            scad_file.write(f"rotate({list(prim['axis'])}) ")
            scad_file.write(f"cylinder(h={prim['height']}, r1={prim['radius']}, "
                          f"r2={prim['radius']});\n")
            
        elif prim['type'] == 'sphere':
            scad_file.write(f"translate({list(prim['center'])}) ")
            scad_file.write(f"sphere(r={prim['radius']});\n")
            
        elif prim['type'] == 'polyhedron':
            generate_polyhedron_scad(prim['vertices'], prim['faces'], scad_file)

def generate_polyhedron_scad(vertices, faces, scad_file):
    """Generate standard polyhedron code"""
    scad_file.write("polyhedron(\n")
    scad_file.write("  points=[\n")
    for v in vertices:
        scad_file.write(f"    [{v[0]}, {v[1]}, {v[2]}],\n")
    scad_file.write("  ],\n")
    scad_file.write("  faces=[\n")
    for f in faces:
        scad_file.write(f"    [{f[0]}, {f[1]}, {f[2]}],\n")
    scad_file.write("  ]\n")
    scad_file.write(");\n")

if __name__ == '__main__':
    main()