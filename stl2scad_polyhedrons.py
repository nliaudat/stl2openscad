#!/usr/bin/env python3
"""
Enhanced STL to OpenSCAD converter with proper mesh validation
"""

import numpy as np
from stl import mesh
import argparse
import sys
import os
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Convert STL to OpenSCAD')
    parser.add_argument('input', help='Input STL file')
    parser.add_argument('output', nargs='?', help='Output SCAD file (default: input with .scad extension)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verification output')
    args = parser.parse_args()

    # Set default output filename if not provided
    if not args.output:
        args.output = os.path.splitext(args.input)[0] + '.scad'

    try:
        # Load and verify STL file
        stl_mesh = mesh.Mesh.from_file(args.input)
        if not validate_mesh(stl_mesh, args.quiet):
            sys.exit(1)

        # Prepare data for OpenSCAD
        vertices = stl_mesh.vectors.reshape(-1, 3)
        unique_vertices, unique_indices = np.unique(vertices, axis=0, return_inverse=True)
        faces = unique_indices.reshape(-1, 3)

        # Generate OpenSCAD file
        with open(args.output, 'w') as scad_file:
            write_scad_header(args.input, scad_file)
            generate_polyhedron(unique_vertices, faces, scad_file)

        if not args.quiet:
            print(f"Successfully converted {args.input} to {args.output}")
            print(f"Vertices: {len(unique_vertices)}, Faces: {len(faces)}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def validate_mesh(mesh_obj, quiet=False):
    """Validate the STL mesh meets basic requirements"""
    if not quiet:
        print("\nValidating STL mesh...")

    checks = {
        'Non-empty mesh': len(mesh_obj.vectors) > 0,
        'Valid vertices': np.all(np.isfinite(mesh_obj.vectors)),
        'Valid normals': np.all(np.isfinite(mesh_obj.normals)),
        'Correct shape': mesh_obj.vectors.shape[1:] == (3, 3),
        'Closed manifold': check_manifold(mesh_obj)
    }

    if not quiet:
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"{status} {check}")

    return all(checks.values())

def check_manifold(mesh_obj):
    """Basic manifold check by verifying each edge is shared by exactly 2 faces"""
    edge_count = defaultdict(int)
    for face in mesh_obj.vectors:
        for i in range(3):
            edge = tuple(sorted((tuple(face[i]), tuple(face[(i+1)%3]))))
            edge_count[edge] += 1

    # All edges should appear exactly twice (once in each direction)
    return all(count == 2 for count in edge_count.values())

def write_scad_header(input_file, scad_file):
    """Write the OpenSCAD file header"""
    scad_file.write(f"// Generated from {os.path.basename(input_file)}\n")
    scad_file.write("$fn = $preview ? 32 : 64;\n\n")

def generate_polyhedron(vertices, faces, scad_file):
    """Generate OpenSCAD polyhedron code with proper formatting"""
    scad_file.write("polyhedron(\n")
    scad_file.write("  points=[\n")
    for v in vertices:
        scad_file.write(f"    [{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}],\n")
    scad_file.write("  ],\n")
    scad_file.write("  faces=[\n")
    for f in faces:
        scad_file.write(f"    [{f[0]}, {f[1]}, {f[2]}],\n")
    scad_file.write("  ],\n")
    scad_file.write("  convexity=10\n")
    scad_file.write(");\n")

if __name__ == '__main__':
    main()