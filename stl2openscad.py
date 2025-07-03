#!/usr/bin/env python3
"""
STL to OpenSCAD Voxel Converter with GPU Acceleration
"""

import argparse
import numpy as np
import trimesh
import random
import sys
from tqdm import tqdm

class GPUSupport:
    def __init__(self):
        self.cuda_available = False
        self.cuda_version = None
        self.cuda_error = None
        self.opencl_available = False
        self.opencl_error = None
        self._detect_gpu_support()

    def _detect_gpu_support(self):
        """Detect available GPU support with version checking"""
        # Check CUDA
        try:
            import cupy as cp
            try:
                self.cuda_version = cp.cuda.runtime.runtimeGetVersion()
                _ = cp.array([1, 2, 3])  # Test basic functionality
                self.cuda_available = True
            except Exception as e:
                self.cuda_error = f"CUDA test failed: {str(e)}"
        except ImportError:
            self.cuda_error = "CuPy not installed"
        except Exception as e:
            self.cuda_error = f"CUDA detection error: {str(e)}"

        # Check OpenCL
        try:
            import pyopencl as cl
            try:
                ctx = cl.create_some_context()
                self.opencl_available = True
            except Exception as e:
                self.opencl_error = f"OpenCL context creation failed: {str(e)}"
        except ImportError:
            self.opencl_error = "PyOpenCL not installed"
        except Exception as e:
            self.opencl_error = f"OpenCL detection error: {str(e)}"

    def print_info(self):
        """Display GPU support information"""
        print("\nGPU Support Information:")
        if self.cuda_available:
            major = self.cuda_version // 1000
            minor = (self.cuda_version % 1000) // 10
            print(f"- CUDA {major}.{minor} available")
        elif self.cuda_error:
            print(f"- CUDA unavailable: {self.cuda_error}")
        
        if self.opencl_available:
            print("- OpenCL available")
        elif self.opencl_error:
            print(f"- OpenCL unavailable: {self.opencl_error}")

GPU_SUPPORT = GPUSupport()

def verify_stl_mesh(mesh, quiet=False):
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

def calculate_batch_size(dims, resolution, quiet=False):
    """Determine optimal batch size based on grid dimensions"""
    total_voxels = np.prod(dims)
    y_dim = dims[1]
    
    # Calculate base batch size
    if total_voxels > 10_000_000:  # Very large grid
        base_batch = max(1000, int(total_voxels / 1000))
    elif total_voxels > 1_000_000:  # Large grid
        base_batch = max(5000, int(total_voxels / 100))
    else:  # Normal grid
        base_batch = min(50000, max(1000, int(total_voxels / 10)))
    
    # Ensure batch size is multiple of y-dimension for GPU compatibility
    batch_size = (base_batch // y_dim) * y_dim
    batch_size = max(y_dim, batch_size)  # Ensure at least one full row
    
    if not quiet and total_voxels > 1_000_000:
        print(f"Large grid detected ({total_voxels:,} voxels), using batch size: {batch_size}")
    
    return batch_size

def voxelize_cpu(mesh, resolution, quiet=False):
    """CPU-based voxelization with nested progress bars"""
    bbox = mesh.bounds
    dims = np.ceil((bbox[1] - bbox[0]) / resolution).astype(int)
    batch_size = calculate_batch_size(dims, resolution, quiet)
    
    if not quiet:
        print(f"\nCPU Voxelization with grid: {dims[0]} x {dims[1]} x {dims[2]}")
        print(f"Estimated voxels: {np.prod(dims):,}")
        print(f"Using batch size: {batch_size}")
        print("\nCPU Voxelization progress:")
        main_pbar = tqdm(total=dims[2], desc="Main voxelization", position=0)

    try:
        voxels = np.zeros(dims, dtype=bool)
        origin = bbox[0]
        
        x = np.linspace(origin[0] + resolution/2, origin[0] + (dims[0]-0.5)*resolution, dims[0])
        y = np.linspace(origin[1] + resolution/2, origin[1] + (dims[1]-0.5)*resolution, dims[1])
        z = np.linspace(origin[2] + resolution/2, origin[2] + (dims[2]-0.5)*resolution, dims[2])

        for zi in range(dims[2]):
            if not quiet:
                layer_pbar = tqdm(total=dims[0]*dims[1], desc=f"Layer {zi+1}/{dims[2]}", 
                                position=1, leave=False)
            
            grid = np.meshgrid(x, y, z[zi], indexing='ij')
            points = np.vstack([grid[0].ravel(), grid[1].ravel(), grid[2].ravel()]).T
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                contains = mesh.contains(batch)
                voxels[:, :, zi].flat[i:i+batch_size] = contains
                
                if not quiet:
                    layer_pbar.update(len(batch))
            
            if not quiet:
                layer_pbar.close()
                main_pbar.update(1)
        
        return voxels, origin
        
    except MemoryError:
        print("\nERROR: Ran out of memory during voxelization!", file=sys.stderr)
        print(f"Try using a larger resolution (current: {resolution}mm)", file=sys.stderr)
        sys.exit(1)
    finally:
        if not quiet:
            main_pbar.close()

def voxelize_gpu_cuda(mesh, resolution, quiet=False):
    """CUDA-accelerated voxelization with improved batch handling"""
    try:
        import cupy as cp
        
        bbox = mesh.bounds
        dims = np.ceil((bbox[1] - bbox[0]) / resolution).astype(int)
        batch_size = calculate_batch_size(dims, resolution, quiet)
        
        if not quiet:
            print(f"\nCUDA Voxelization with grid: {dims[0]} x {dims[1]} x {dims[2]}")
            print(f"Estimated voxels: {np.prod(dims):,}")
            print(f"Using batch size: {batch_size}")
            print("\nCUDA Voxelization progress:")
            main_pbar = tqdm(total=dims[2], desc="Main voxelization", position=0)

        voxels_gpu = cp.zeros(dims, dtype=bool)
        origin = bbox[0]
        
        for zi in range(dims[2]):
            if not quiet:
                layer_pbar = tqdm(total=dims[0]*dims[1], desc=f"Layer {zi+1}/{dims[2]}", 
                                position=1, leave=False)
            
            z_val = origin[2] + (zi + 0.5) * resolution
            x = cp.linspace(origin[0] + resolution/2, 
                           origin[0] + (dims[0]-0.5)*resolution, 
                           dims[0])
            y = cp.linspace(origin[1] + resolution/2,
                           origin[1] + (dims[1]-0.5)*resolution,
                           dims[1])
            
            # Process in complete rows to avoid reshaping issues
            rows_per_batch = batch_size // dims[1]
            for row_start in range(0, dims[0], rows_per_batch):
                row_end = min(row_start + rows_per_batch, dims[0])
                
                grid_x, grid_y = cp.meshgrid(x[row_start:row_end], y, indexing='ij')
                points = cp.stack([
                    grid_x.ravel(),
                    grid_y.ravel(),
                    cp.full((row_end-row_start)*dims[1], z_val)
                ], axis=1)
                
                points_cpu = cp.asnumpy(points)
                contains = mesh.contains(points_cpu)
                contains_gpu = cp.array(contains).reshape(row_end-row_start, dims[1])
                
                voxels_gpu[row_start:row_end, :, zi] = contains_gpu
                
                if not quiet:
                    layer_pbar.update((row_end-row_start)*dims[1])
            
            if not quiet:
                layer_pbar.close()
                main_pbar.update(1)
        
        return cp.asnumpy(voxels_gpu), origin
        
    except Exception as e:
        print(f"\nCUDA Error: {str(e)}", file=sys.stderr)
        print("Falling back to CPU voxelization", file=sys.stderr)
        return voxelize_cpu(mesh, resolution, quiet)
    finally:
        if not quiet:
            main_pbar.close()

def voxelize_gpu_opencl(mesh, resolution, quiet=False):
    """OpenCL-accelerated voxelization"""
    try:
        import pyopencl as cl
        
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        
        bbox = mesh.bounds
        dims = np.ceil((bbox[1] - bbox[0]) / resolution).astype(int)
        batch_size = calculate_batch_size(dims, resolution, quiet)
        
        if not quiet:
            print(f"\nOpenCL Voxelization with grid: {dims[0]} x {dims[1]} x {dims[2]}")
            print(f"Estimated voxels: {np.prod(dims):,}")
            print(f"Using batch size: {batch_size}")
            print("\nOpenCL Voxelization progress:")
            main_pbar = tqdm(total=dims[2], desc="Main voxelization", position=0)

        voxels = np.zeros(dims, dtype=np.bool_)
        origin = bbox[0]

        for zi in range(dims[2]):
            if not quiet:
                layer_pbar = tqdm(total=dims[0]*dims[1], desc=f"Layer {zi+1}/{dims[2]}", 
                                 position=1, leave=False)
            
            z_val = origin[2] + (zi + 0.5) * resolution
            x = np.linspace(origin[0] + resolution/2, 
                           origin[0] + (dims[0]-0.5)*resolution, 
                           dims[0])
            y = np.linspace(origin[1] + resolution/2,
                           origin[1] + (dims[1]-0.5)*resolution,
                           dims[1])
            
            # Process in complete rows
            rows_per_batch = batch_size // dims[1]
            for row_start in range(0, dims[0], rows_per_batch):
                row_end = min(row_start + rows_per_batch, dims[0])
                
                grid_x, grid_y = np.meshgrid(x[row_start:row_end], y, indexing='ij')
                points = np.stack([
                    grid_x.ravel(),
                    grid_y.ravel(),
                    np.full((row_end-row_start)*dims[1], z_val)
                ], axis=1)
                
                contains = mesh.contains(points)
                voxels[row_start:row_end, :, zi] = contains.reshape(row_end-row_start, dims[1])
                
                if not quiet:
                    layer_pbar.update((row_end-row_start)*dims[1])
            
            if not quiet:
                layer_pbar.close()
                main_pbar.update(1)
        
        return voxels, origin
        
    except Exception as e:
        print(f"\nOpenCL Error: {str(e)}", file=sys.stderr)
        print("Falling back to CPU voxelization", file=sys.stderr)
        return voxelize_cpu(mesh, resolution, quiet)
    finally:
        if not quiet:
            main_pbar.close()

def greedy_merge(voxels, quiet=False):
    """Greedy merging algorithm with nested progress bars"""
    visited = np.zeros_like(voxels, dtype=bool)
    cubes = []
    total_slices = voxels.shape[2]

    if not quiet:
        print("\nMerging progress:")
        main_pbar = tqdm(total=total_slices, desc="Main merging", position=0)

    try:
        for z in range(total_slices):
            if not quiet:
                slice_pbar = tqdm(total=voxels.shape[1], desc=f"Slice {z+1}/{total_slices}", 
                                 position=1, leave=False)
            
            for y in range(voxels.shape[1]):
                x = 0
                while x < voxels.shape[0]:
                    if voxels[x, y, z] and not visited[x, y, z]:
                        # Expansion sub-process
                        x_end = x + 1
                        while x_end < voxels.shape[0] and voxels[x_end, y, z] and not visited[x_end, y, z]:
                            x_end += 1

                        y_end = y + 1
                        while y_end < voxels.shape[1] and np.all(voxels[x:x_end, y_end, z] & ~visited[x:x_end, y_end, z]):
                            y_end += 1

                        z_end = z + 1
                        while z_end < voxels.shape[2] and np.all(voxels[x:x_end, y:y_end, z_end] & ~visited[x:x_end, y:y_end, z_end]):
                            z_end += 1

                        visited[x:x_end, y:y_end, z:z_end] = True
                        cubes.append(((x, y, z), (x_end, y_end, z_end)))
                        x = x_end
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

def write_scad(cubes, origin, resolution, out_path, debug=False):
    """Write OpenSCAD file from merged cubes"""
    try:
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
                    color = [random.random() for _ in range(3)]
                    f.write(f"  translate([{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])\n")
                    f.write(f"    colored_cube([{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}], [{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}]);\n")
                else:
                    f.write(f"  translate([{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])\n")
                    f.write(f"    cube([{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]);\n")
            
            f.write("}\n")
    except IOError as e:
        print(f"\nFile Error: Could not write to {out_path}", file=sys.stderr)
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Convert STL to OpenSCAD via voxel merging with GPU support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help="Input STL file")
    parser.add_argument("output", help="Output SCAD file")
    parser.add_argument("-r", "--resolution", type=float, default=1.0,
                      help="Voxel size in mm")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode with random colors for each cube")
    parser.add_argument("--quiet", action="store_true",
                      help="Disable progress bars and most output")
    parser.add_argument("--gpu", choices=['cuda', 'opencl', 'auto'], default='auto',
                      help="GPU acceleration mode (cuda, opencl, or auto-detect)")
    args = parser.parse_args()

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

        # Select voxelization method
        if args.gpu == 'cuda':
            if GPU_SUPPORT.cuda_available:
                if not args.quiet:
                    print("\nUsing CUDA acceleration")
                voxels, origin = voxelize_gpu_cuda(mesh, args.resolution, args.quiet)
            else:
                if not args.quiet:
                    print("\nCUDA requested but not available. Falling back to CPU.")
                voxels, origin = voxelize_cpu(mesh, args.resolution, args.quiet)
        elif args.gpu == 'opencl':
            if GPU_SUPPORT.opencl_available:
                if not args.quiet:
                    print("\nUsing OpenCL acceleration")
                voxels, origin = voxelize_gpu_opencl(mesh, args.resolution, args.quiet)
            else:
                if not args.quiet:
                    print("\nOpenCL requested but not available. Falling back to CPU.")
                voxels, origin = voxelize_cpu(mesh, args.resolution, args.quiet)
        else:  # auto
            if GPU_SUPPORT.cuda_available:
                if not args.quiet:
                    print("\nAuto-selecting CUDA acceleration")
                voxels, origin = voxelize_gpu_cuda(mesh, args.resolution, args.quiet)
            elif GPU_SUPPORT.opencl_available:
                if not args.quiet:
                    print("\nAuto-selecting OpenCL acceleration")
                voxels, origin = voxelize_gpu_opencl(mesh, args.resolution, args.quiet)
            else:
                if not args.quiet:
                    print("\nNo GPU acceleration available. Using CPU.")
                voxels, origin = voxelize_cpu(mesh, args.resolution, args.quiet)

        if not args.quiet:
            print(f"\nVoxel grid shape: {voxels.shape}")
            print(f"Filled voxels: {np.sum(voxels):,}")

        # Merge and output
        cubes = greedy_merge(voxels, args.quiet)
        
        if not args.quiet:
            print(f"\nTotal merged cubes: {len(cubes):,}")
            print(f"Compression ratio: {len(cubes)/np.sum(voxels):.1f}x" if np.sum(voxels) > 0 else "No voxels found")
            print(f"Writing to OpenSCAD: {args.output}")

        write_scad(cubes, origin, args.resolution, args.output, debug=args.debug)
        
        if not args.quiet:
            print("\nFinished.")

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()