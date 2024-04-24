#include <torch/extension.h>
// for (uint32_t i = 0; i < N_DIMS; ++i) {
//   atomicMax(out + index[idx] * N_DIMS + i, src[idx * N_DIMS + i]);
    // }
__device__ static inline float atomicMax(float *address, const float val) {
  unsigned int *address_as_ui = (unsigned int *)address;  // NOLINT
  unsigned int old = *address_as_ui;                       // NOLINT
  unsigned int assumed;                                    // NOLINT
  do {
    assumed = old;
    // printf("old = %d",old);
    old = atomicCAS(address_as_ui, *address_as_ui, __float_as_uint(val));
    // old = atomicCAS(address_as_ui, assumed, __float_as_uint(max(val, __uint_as_float(assumed))));
  } while (assumed != old);  // NOLINT
  return __longlong_as_double(old);


  
}

template <typename scalar_t, uint32_t N_DIMS>
__global__ void scatter_max_kernel(

  const scalar_t* src,
  const int64_t* index,
  scalar_t*out,
  int src_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // std::cout << "N_DIMS = " << N_DIMS<<"\n";
    // printf("blockDim.x = %d",blockDim.x);
  if (idx < src_size) {
    _Pragma("unroll")
    for (uint32_t i = 0; i < N_DIMS; ++i) {
      atomicMax(out + index[idx] * N_DIMS + i, src[idx * N_DIMS + i]);
      // old = atomicCAS(out + index[idx] * N_DIMS + i, *(out + index[idx] * N_DIMS + i), __float_as_uint(src[idx * N_DIMS + i]));

    }
  }
}

// #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor scatter_max(
  torch::Tensor src,
  torch::Tensor index,
  int dim,
  int dim_size) {
  // CHECK_INPUT(src);
  // CHECK_INPUT(index);
  dim = dim < 0 ? src.dim() + dim : dim;
  // printf("dim = %d",dim);
  auto size = src.sizes().vec();
  // std::cout << "size1 = " << size<<"\n";
  size[dim] = dim_size;
  // std::cout << "size2 = " << size<<"\n";

  // assert(src.sizes()[1] == 8);
  // assert(src.type().scalarType() == at::ScalarType::Float);
  // assert(index.type().scalarType() == at::ScalarType::Long);

    // std::cout << "src = " << src.sizes()<<"\n";

  auto result = src.new_zeros(size);
//  auto result = result.sizes().vec();
    // std::cout << "size1 = " << size<<"\n";

  result.fill_(std::numeric_limits<float>::lowest());

    // std::cout << "size = " << size<<"\n";
    // std::cout << "result = " << result.sizes() <<"\n";
    // std::cout << "result = " << result <<"\n";

    // std::cout << "src.sizes()[0] = " << src.sizes()[0] <<"\n";

  const int threads = 1024;
  const int blocks = src.sizes()[0] / threads + 1;
  scatter_max_kernel<float, 8><<<blocks, threads>>>(
    // src.data<float>(),
    src.data<float>(),
    index.data<int64_t>(),
    result.data<float>(),
    src.sizes()[0]);

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scatter_max", &scatter_max, "scatter_max");
}
