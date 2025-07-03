# Optimization Sources and References

## Core Algorithms
- **Greedy Meshing Algorithm**: Adapted from Drububu's Voxelizer  
  [http://drububu.com/project/voxelizer/](http://drububu.com/project/voxelizer/)
- **Voxelization Techniques**: Inspired by PyVista's implementation  
  [https://docs.pyvista.org/examples/01-filter/voxelize.html](https://docs.pyvista.org/examples/01-filter/voxelize.html)

## GPU Accelerations
### CUDA Optimizations
- **Memory Management**: NVIDIA CUDA Best Practices Guide  
  [https://developer.nvidia.com/cuda-best-practices-guide](https://developer.nvidia.com/cuda-best-practices-guide)
- **Stream Processing**: CUDA Stream Documentation  
  [https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- **CuPy Implementation**: Used for GPU array operations  
  [https://docs.cupy.dev/en/stable/user_guide/performance.html](https://docs.cupy.dev/en/stable/user_guide/performance.html)

### OpenCL Optimizations
- **PyOpenCL Best Practices**:  
  [https://documen.tician.de/pyopencl/](https://documen.tician.de/pyopencl/)
- **Khronos OpenCL Docs**:  
  [https://www.khronos.org/opencl/](https://www.khronos.org/opencl/)
- **Buffer Management**:  
  [https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateBuffer.html](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateBuffer.html)

## Mesh Processing
- **Trimesh Validation**: Mesh quality checking methods  
  [https://trimsh.org/trimesh.html](https://trimsh.org/trimesh.html)
- **Watertight Repair**: Techniques from MeshLab  
  [https://www.meshlab.net/#documentation](https://www.meshlab.net/#documentation)

## Output Generation
- **OpenSCAD Optimization**: Performance tuning guidelines  
  [https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Performance_Tuning](https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Performance_Tuning)
- **3D Printing Best Practices**: Cube placement strategies  
  [https://help.prusa3d.com/article/openSCAD-tips_1255](https://help.prusa3d.com/article/openSCAD-tips_1255)

## Memory Management
- **Chunked Processing**: PyVista large data handling  
  [https://docs.pyvista.org/user-guide/large.html](https://docs.pyvista.org/user-guide/large.html)
- **Batch Size Calculation**: NVIDIA Memory Management  
  [https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)

## Related Projects
- **ImplicitCAD**: Inspiration for geometric optimizations  
  [http://www.implicitcad.org/docs/reference](http://www.implicitcad.org/docs/reference)
- **Thingiverse Customizer**: Debug rendering approach  
  [https://www.thingiverse.com/developers/customizer](https://www.thingiverse.com/developers/customizer)
