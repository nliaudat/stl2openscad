#!/usr/bin/env python3
"""
Enhanced STL to OpenSCAD Converter with Open3D Primitive Extraction
Combining voxelization with primitive detection for optimal results
"""

import argparse
import numpy as np
import trimesh
import random
import sys
from tqdm import tqdm
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from typing import List, Dict, Tuple, Optional

class GPUSupport:
    def __init__(self):
        """Initialize GPU support detection with Open3D"""
        self.open3d_available = True
        self.open3d_cuda_available = False
        self.open3d_error = None
        self._detect_gpu_support()

    def _detect_gpu_support(self):
        """Detect available GPU support through Open3D"""
        try:
            # Check if Open3D was built with CUDA support
            if o3d.core.cuda.is_available():
                self.open3d_cuda_available = True
        except Exception as e:
            self.open3d_error = str(e)
            self.open3d_cuda_available = False

    def print_info(self):
        """Display detailed GPU support information"""
        print("\nGPU Support Information:")
        if self.open3d_cuda_available:
            print("- Open3D with CUDA acceleration available")
            devices = o3d.core.cuda.get_devices()
            for i, device in enumerate(devices):
                print(f"  Device {i}: {device.get_name()}")
        else:
            print("- Open3D running in CPU mode")
            if self.open3d_error:
                print(f"  Error: {self.open3d_error}")

class PrimitiveExtractor:
    def __init__(self, mesh: trimesh.Trimesh, resolution: float, use_cuda: bool = False):
        """Initialize with mesh and processing parameters"""
        self.mesh = mesh
        self.resolution = resolution
        self.use_cuda = use_cuda
        self.origin = mesh.bounds[0]
        
        # Convert trimesh to Open3D
        self.o3d_mesh = o3d.geometry.TriangleMesh()
        self.o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        self.o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        
    def voxelize_basic(self) -> np.ndarray:
        """Basic voxel grid approach returning 3D boolean array"""
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
            self.o3d_mesh,
            voxel_size=self.resolution
        )
        
        # Convert to dense 3D array
        bbox = self.mesh.bounds
        dims = np.ceil((bbox[1] - bbox[0]) / self.resolution).astype(int)
        voxels = np.zeros(dims, dtype=bool)
        
        for voxel in voxel_grid.get_voxels():
            i, j, k = voxel.grid_index
            if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
                voxels[i, j, k] = True
                
        return voxels
    
    def voxelize_with_surface_sampling(self, num_points: int = 100000) -> np.ndarray:
        """Voxelize using surface point sampling for better accuracy"""
        # Sample points from mesh surface
        pcd = self.o3d_mesh.sample_points_poisson_disk(number_of_points=num_points)
        
        # Create voxel grid from point cloud
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd,
            voxel_size=self.resolution
        )
        
        # Convert to dense 3D array
        bbox = self.mesh.bounds
        dims = np.ceil((bbox[1] - bbox[0]) / self.resolution).astype(int)
        voxels = np.zeros(dims, dtype=bool)
        
        for voxel in voxel_grid.get_voxels():
            i, j, k = voxel.grid_index
            if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
                voxels[i, j, k] = True
                
        return voxels
    
    def detect_planar_surfaces(self, distance_threshold: float = 0.01, 
                              ransac_n: int = 3, 
                              num_iterations: int = 1000) -> List[Dict]:
        """Detect planar surfaces using RANSAC"""
        pcd = self.o3d_mesh.sample_points_poisson_disk(number_of_points=50000)
        planes = []
        
        # Iteratively find planes
        remaining_pcd = pcd
        for _ in range(10):  # Max 10 planes
            if len(remaining_pcd.points) < ransac_n * 10:
                break
                
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            
            if len(inliers) < 100:  # Minimum plane size
                break
                
            # Extract plane points
            plane_pcd = remaining_pcd.select_by_index(inliers)
            plane_points = np.asarray(plane_pcd.points)
            
            # Calculate plane bounds
            if len(plane_points) > 2:
                try:
                    hull = ConvexHull(plane_points[:, :2])  # Project to 2D
                    bounds = plane_points[hull.vertices]
                except:
                    bounds = [
                        plane_points.min(axis=0),
                        plane_points.max(axis=0)
                    ]
            else:
                bounds = [
                    plane_points.min(axis=0),
                    plane_points.max(axis=0)
                ]
            
            planes.append({
                'type': 'plane',
                'equation': plane_model,
                'points': plane_points,
                'bounds': bounds,
                'thickness': distance_threshold * 2
            })
            
            # Remove detected plane points
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            
        return planes
    
    def detect_geometric_primitives(self, min_points: int = 100) -> List[Dict]:
        """Cluster point cloud to detect geometric primitives"""
        pcd = self.o3d_mesh.sample_points_poisson_disk(number_of_points=50000)
        points = np.asarray(pcd.points)
        
        # Cluster points using DBSCAN
        clustering = DBSCAN(eps=self.resolution*2, min_samples=min_points).fit(points)
        labels = clustering.labels_
        
        # Analyze clusters for primitive shapes
        primitives = []
        for label in np.unique(labels):
            if label == -1:  # Skip noise
                continue
                
            cluster_points = points[labels == label]
            
            # Basic primitive classification
            if len(cluster_points) > 1000:  # Large cluster
                # Try to fit plane first
                try:
                    _, _, normal = np.linalg.svd(cluster_points - cluster_points.mean(axis=0))
                    normal = normal[2]  # Third singular vector is normal
                    if np.abs(np.dot(normal, [0, 0, 1])) > 0.9:  # Mostly horizontal/vertical
                        prim_type = 'plane'
                    else:
                        prim_type = 'unknown_surface'
                except:
                    prim_type = 'unknown'
            else:
                prim_type = 'unknown'
            
            primitives.append({
                'type': prim_type,
                'points': cluster_points,
                'bounds': [
                    cluster_points.min(axis=0),
                    cluster_points.max(axis=0)
                ]
            })
            
        return primitives
    
    def hybrid_voxelization(self) -> Tuple[np.ndarray, List[Dict]]:
        """Combine primitive detection with voxelization for optimal results"""
        # Detect large planar surfaces first
        planes = self.detect_planar_surfaces()
        
        # Create mask for planar regions
        bbox = self.mesh.bounds
        dims = np.ceil((bbox[1] - bbox[0]) / self.resolution).astype(int)
        plane_mask = np.zeros(dims, dtype=bool)
        
        for plane in planes:
            plane_points = plane['points']
            if len(plane_points) > 0:
                # Convert plane points to voxel indices
                indices = ((plane_points - self.origin) / self.resolution).astype(int)
                for idx in indices:
                    if 0 <= idx[0] < dims[0] and 0 <= idx[1] < dims[1] and 0 <= idx[2] < dims[2]:
                        plane_mask[idx[0], idx[1], idx[2]] = True
        
        # Voxelize the remaining geometry
        voxels = self.voxelize_with_surface_sampling()
        
        # Combine results (voxels not in planes)
        combined = np.logical_and(voxels, np.logical_not(plane_mask))
        
        return combined, planes

def verify_stl_mesh(mesh: trimesh.Trimesh, quiet: bool = False) -> None:
    """Verify STL mesh integrity before processing"""
    if not quiet:
        print("\nVerifying STL mesh...")
    
    checks = {
        'Watertight': mesh.is_watertight,
        'Valid bounds': np.all(np.isfinite(mesh.bounds)),
        'Non-empty': len(mesh.vertices) > 0 and len(mesh.faces) > 0,
        'Manifold': mesh.is_volume if hasattr(mesh, 'is_volume') else True
    }
    
    if not quiet:
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"{status} {check}")
    
    if not all(checks.values()):
        raise ValueError("Invalid STL mesh detected")

def greedy_merge(voxels: np.ndarray, quiet: bool = False) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Greedy merging algorithm with optimizations"""
    visited = np.zeros_like(voxels, dtype=bool)
    cubes = []
    total_slices = voxels.shape[2]

    if not quiet:
        print("\nMerging voxels into optimized cubes...")
        main_pbar = tqdm(total=total_slices, desc="Main merging", position=0)

    try:
        for z in range(total_slices):
            if not quiet:
                slice_pbar = tqdm(total=voxels.shape[1], desc=f"Slice {z+1}/{total_slices}", position=1, leave=False)
            
            for y in range(voxels.shape[1]):
                x = 0
                while x < voxels.shape[0]:
                    if voxels[x, y, z] and not visited[x, y, z]:
                        # Find maximum possible cube
                        max_size = min(voxels.shape[0]-x, voxels.shape[1]-y, voxels.shape[2]-z)
                        size = 1
                        
                        # Check how far we can extend
                        while size < max_size:
                            if (voxels[x:x+size+1, y:y+size+1, z:z+size+1].all() and 
                                not visited[x:x+size+1, y:y+size+1, z:z+size+1].any()):
                                size += 1
                            else:
                                break
                        
                        visited[x:x+size, y:y+size, z:z+size] = True
                        cubes.append(((x, y, z), (x+size, y+size, z+size)))
                        x += size
                    else:
                        x += 1
                
                if not quiet:
                    slice_pbar.update(1)
            
            if not quiet:
                slice_pbar.close()
                main_pbar.update(1)
        
        return cubes
        
    except Exception as e:
        print(f"\nMerging Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        if not quiet:
            main_pbar.close()

def write_scad(cubes: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]], 
              planes: List[Dict],
              origin: np.ndarray, 
              resolution: float, 
              out_path: str, 
              debug: bool = False) -> None:
    """Write OpenSCAD file with optimized cube placement and primitives"""
    try:
        with open(out_path, 'w') as f:
            f.write("// Enhanced STL to OpenSCAD Converter\n")
            f.write(f"// Resolution: {resolution} mm\n")
            f.write("// Includes both voxelized geometry and detected primitives\n\n")
            
            if debug:
                f.write("module colored_cube(size, color) {\n")
                f.write("    color(color) cube(size);\n")
                f.write("}\n\n")
            
            f.write("union() {\n")
            
            # Write detected planes first (as large thin boxes)
            for i, plane in enumerate(planes):
                if plane['type'] == 'plane':
                    bounds = plane['bounds']
                    size = bounds[1] - bounds[0]
                    thickness = plane['thickness']
                    
                    # Make sure we have some thickness
                    min_dim = np.argmin(size)
                    size[min_dim] = max(thickness, resolution)
                    
                    pos = bounds[0] + size/2
                    
                    if debug:
                        color = [random.random() for _ in range(3)]
                        f.write(f"  // Plane {i}\n")
                        f.write(f"  translate([{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])\n")
                        f.write(f"    colored_cube([{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}], [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]);\n")
                    else:
                        f.write(f"  // Plane {i}\n")
                        f.write(f"  translate([{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])\n")
                        f.write(f"    cube([{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}], center=true);\n")
            
            # Sort cubes by size (largest first) for better rendering
            cubes.sort(key=lambda c: (c[1][0]-c[0][0])*(c[1][1]-c[0][1])*(c[1][2]-c[0][2]), reverse=True)
            
            # Write voxel cubes
            for (x0, y0, z0), (x1, y1, z1) in cubes:
                pos = origin + np.array([x0, y0, z0]) * resolution
                size = (np.array([x1 - x0, y1 - y0, z1 - z0])) * resolution
                
                if debug:
                    color = [random.random() for _ in range(3)]
                    f.write(f"  translate([{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])\n")
                    f.write(f"    colored_cube([{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}], [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]);\n")
                else:
                    f.write(f"  translate([{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])\n")
                    f.write(f"    cube([{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]);\n")
            
            f.write("}\n")
    except IOError as e:
        print(f"\nFile Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function with Open3D integration"""
    parser = argparse.ArgumentParser(
        description="Convert STL to OpenSCAD with advanced Open3D processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help="Input STL file")
    parser.add_argument("output", help="Output SCAD file")
    parser.add_argument("-r", "--resolution", type=float, default=1.0,
                      help="Base voxel size in mm")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with random colors for each cube")
    parser.add_argument("--quiet", action="store_true",
                      help="Disable progress bars and most output")
    parser.add_argument("--gpu", action="store_true",
                      help="Enable GPU acceleration (if available)")
    parser.add_argument("--method", choices=['basic', 'sampling', 'hybrid'], default='basic',
                      help="Voxelization method (basic, sampling, or hybrid)")
    args = parser.parse_args()

    GPU_SUPPORT = GPUSupport()
    
    if not args.quiet:
        print(f"Loading mesh: {args.input}")
        GPU_SUPPORT.print_info()

    try:
        # Load and verify mesh
        mesh = trimesh.load(args.input, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()
        mesh.process(validate=True)
        verify_stl_mesh(mesh, args.quiet)

        # Process mesh with selected method
        extractor = PrimitiveExtractor(mesh, args.resolution, args.gpu)
        
        if args.method == 'basic':
            if not args.quiet:
                print("\nUsing basic voxelization method")
            voxels = extractor.voxelize_basic()
            planes = []
        elif args.method == 'sampling':
            if not args.quiet:
                print("\nUsing surface sampling voxelization")
            voxels = extractor.voxelize_with_surface_sampling()
            planes = []
        else:  # hybrid
            if not args.quiet:
                print("\nUsing hybrid voxelization with primitive detection")
            voxels, planes = extractor.hybrid_voxelization()

        if not args.quiet:
            print(f"\nVoxel grid shape: {voxels.shape}")
            print(f"Filled voxels: {np.sum(voxels):,}")
            print(f"Detected primitives: {len(planes)}")

        # Merge and output
        cubes = greedy_merge(voxels, args.quiet)
        
        if not args.quiet:
            print(f"\nTotal merged cubes: {len(cubes):,}")
            print(f"Writing to OpenSCAD: {args.output}")

        write_scad(cubes, planes, extractor.origin, args.resolution, args.output, debug=args.debug)
        
        if not args.quiet:
            print("\nFinished.")

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()