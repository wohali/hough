# -*- coding: utf-8 -*-
# vim: ts=4:sw=4:et:tw=88:nowrap

try:
    import cupy as cp
    import cv2

    # thank you, https://github.com/rapidsai/cucim/issues/329 !
    def cv_cuda_gpumat_from_cp_array(arr: cp.ndarray) -> cv2.cuda.GpuMat:
        assert len(arr.shape) in (
            2,
            3,
        ), "CuPy array must have 2 or 3 dimensions to be a valid GpuMat"
        type_map = {
            cp.dtype("uint8"): cv2.CV_8U,
            cp.dtype("int8"): cv2.CV_8S,
            cp.dtype("uint16"): cv2.CV_16U,
            cp.dtype("int16"): cv2.CV_16S,
            cp.dtype("int32"): cv2.CV_32S,
            cp.dtype("float32"): cv2.CV_32F,
            cp.dtype("float64"): cv2.CV_64F,
        }
        depth = type_map.get(arr.dtype)
        assert depth is not None, f"Unsupported CuPy array dtype {arr.dtype}"
        channels = 1 if len(arr.shape) == 2 else arr.shape[2]
        # equivalent to unexposed opencv C++ macro CV_MAKETYPE(depth,channels):
        # (depth&7) + ((channels - 1) << 3)
        mat_type = depth + ((channels - 1) << 3)
        # TODO: do we need [1::-1] here to invert the matrix?
        mat = cv2.cuda.createGpuMatFromCudaMemory(
            arr.__cuda_array_interface__["shape"][1::-1],
            mat_type,
            arr.__cuda_array_interface__["data"][0],
        )
        return mat

    def cp_array_from_cv_cuda_gpumat(mat: cv2.cuda.GpuMat) -> cp.ndarray:
        class CudaArrayInterface:
            def __init__(self, gpu_mat: cv2.cuda.GpuMat):
                w, h = gpu_mat.size()
                type_map = {
                    cv2.CV_8U: "|u1",
                    cv2.CV_8S: "|i1",
                    cv2.CV_16U: "<u2",
                    cv2.CV_16S: "<i2",
                    cv2.CV_32S: "<i4",
                    cv2.CV_32F: "<f4",
                    cv2.CV_64F: "<f8",
                }
                self.__cuda_array_interface__ = {
                    "version": 3,
                    "shape": (
                        (h, w, gpu_mat.channels()) if gpu_mat.channels() > 1 else (h, w)
                    ),
                    "typestr": type_map[gpu_mat.depth()],
                    "descr": [("", type_map[gpu_mat.depth()])],
                    "stream": 1,
                    "strides": (
                        (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1())
                        if gpu_mat.channels() > 1
                        else (gpu_mat.step, gpu_mat.elemSize())
                    ),
                    "data": (gpu_mat.cudaPtr(), False),
                }

        arr = cp.asarray(CudaArrayInterface(mat))
        return arr

except ModuleNotFoundError:
    pass
