FROM ubuntu:22.04
ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.2 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=450,driver<451
ENV NV_CUDA_CUDART_VERSION=11.2.72-1
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-11-2
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends     gnupg2 curl ca-certificates &&     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - &&     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list &&     apt-get purge --autoremove -y curl     && rm -rf /var/lib/apt/lists/* # buildkit
ENV CUDA_VERSION=11.2.0
RUN apt-get update && apt-get install -y --no-install-recommends     cuda-cudart-11-2=${NV_CUDA_CUDART_VERSION}     ${NV_CUDA_COMPAT_PACKAGE}     && ln -s cuda-11.2 /usr/local/cuda &&     rm -rf /var/lib/apt/lists/* # buildkit
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf     && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf # buildkit
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NV_CUDA_LIB_VERSION=11.2.0-1
ENV NV_NVTX_VERSION=11.2.67-1
ENV NV_LIBNPP_VERSION=11.2.1.68-1
ENV NV_LIBNPP_PACKAGE=libnpp-11-2=11.2.1.68-1
ENV NV_LIBCUSPARSE_VERSION=11.3.1.68-1
ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-11-2
ENV NV_LIBCUBLAS_VERSION=11.3.1.68-1
ENV NV_LIBCUBLAS_PACKAGE=libcublas-11-2=11.3.1.68-1
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.8.4-1
ENV NCCL_VERSION=2.8.4-1
ENV NV_LIBNCCL_PACKAGE=libnccl2=2.8.4-1+cuda11.2
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends     cuda-libraries-11-2=${NV_CUDA_LIB_VERSION}     ${NV_LIBNPP_PACKAGE}     cuda-nvtx-11-2=${NV_NVTX_VERSION}     libcusparse-11-2=${NV_LIBCUSPARSE_VERSION}     ${NV_LIBCUBLAS_PACKAGE}     ${NV_LIBNCCL_PACKAGE}     && rm -rf /var/lib/apt/lists/* # buildkit
RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME} # buildkit
ENV NV_CUDA_LIB_VERSION=11.2.0-1
ENV NV_CUDA_CUDART_DEV_VERSION=11.2.72-1
ENV NV_NVML_DEV_VERSION=11.2.67-1
ENV NV_LIBCUSPARSE_DEV_VERSION=11.3.1.68-1
ENV NV_LIBNPP_DEV_VERSION=11.2.1.68-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-11-2=11.2.1.68-1
ENV NV_LIBCUBLAS_DEV_VERSION=11.3.1.68-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-11-2
ENV NV_LIBCUBLAS_DEV_PACKAGE=libcublas-dev-11-2=11.3.1.68-1
ENV NV_NVPROF_VERSION=11.2.67-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-11-2=11.2.67-1
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.8.4-1
ENV NCCL_VERSION=2.8.4-1
ENV NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.8.4-1+cuda11.2
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends     libtinfo5 libncursesw5     cuda-cudart-dev-11-2=${NV_CUDA_CUDART_DEV_VERSION}     cuda-command-line-tools-11-2=${NV_CUDA_LIB_VERSION}     cuda-minimal-build-11-2=${NV_CUDA_LIB_VERSION}     cuda-libraries-dev-11-2=${NV_CUDA_LIB_VERSION}     cuda-nvml-dev-11-2=${NV_NVML_DEV_VERSION}     ${NV_NVPROF_DEV_PACKAGE}     ${NV_LIBNPP_DEV_PACKAGE}     libcusparse-dev-11-2=${NV_LIBCUSPARSE_DEV_VERSION}     ${NV_LIBCUBLAS_DEV_PACKAGE}     ${NV_LIBNCCL_DEV_PACKAGE}     && rm -rf /var/lib/apt/lists/* # buildkit
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME} # buildkit
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV NV_CUDNN_VERSION=8.1.1.33
ENV NV_CUDNN_PACKAGE_NAME=libcudnn8
ENV NV_CUDNN_PACKAGE=libcudnn8=8.1.1.33-1+cuda11.2
ENV NV_CUDNN_PACKAGE_DEV=libcudnn8-dev=8.1.1.33-1+cuda11.2
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends     ${NV_CUDNN_PACKAGE}     ${NV_CUDNN_PACKAGE_DEV}     && apt-mark hold ${NV_CUDNN_PACKAGE_NAME}     && rm -rf /var/lib/apt/lists/* # buildkit

