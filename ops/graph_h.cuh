#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
__device__ static inline float layernorm(float *address, float *weight, float *bias, int embed)
{

    float sum1 = 0;
    float sum2 = 0;
    float mean = 0;
    float var = 0;

    float eps = 1e-5;

    for (int i = 0; i < embed; i++)
    {
        sum1 += address[i];
        sum2 += address[i] * address[i];
    }

    mean = sum1 / embed;
    var = (sum2 / embed) - mean * mean;

    for (int i = 0; i < embed; i++)
    {
        address[i] = ((address[i] - mean) * rsqrt(var + eps)) * weight[i] + bias[i];
    }
}

__device__ static inline float linear(float *output, float *weight, float *bias, float *input,
                                      int out_channel, int in_channel)
{
    for (uint32_t i = 0; i < out_channel; ++i)
    {
        output[i] = bias[i];
        for (uint32_t j = 0; j < in_channel; j++)
            output[i] += input[j] * weight[i * in_channel + j];
    }
}

__device__ static inline float relu(float *center_embed, int embed)
{
    for (int i = 0; i < embed; i++)
    {
        center_embed[i] = (center_embed[i] > float(0)) ? center_embed[i] : float(0);
    }
}

__device__ static inline float debug(float *temp, float *test_output, float embed)
{
    for (int i = 0; i < embed; i++)
    {
        test_output[i] = temp[i];
    }
}

__device__ static inline float clear(float *temp, float embed)
{
    for (int i = 0; i < embed; i++)
    {
        temp[i] = 0;
    }
}

__device__ static inline float add(float *output, float *input, float embed)
{
    for (int i = 0; i < embed; i++)
    {
        output[i] += input[i];
    }
}

__device__ static inline float move(float *output, float *input, float embed)
{
    for (int i = 0; i < embed; i++)
    {
        output[i] = input[i];
    }
}

__device__ static inline float mul(float *output, float *input1, float *input2, int embed)
{
    for (int i = 0; i < embed; i++)
    {
        output[i] = input1[i] * input2[i];
    }
}

__device__ static inline float sum_scale_max(float *output, float *input, int head)
{
    for (int i = 0; i < head; i++)
    {
       if(output[i] < input[i])
       {
           output[i] = input[i];
       }
    }
}

__device__ static inline float sum_scale(float *output, float *input, int head)
{
    for (int i = 0; i < head; i++)
    {
        output[i] = 0;
        for (int j = 0; j < head; j++)
        {
            output[i] += input[i * head + j];
        }

        output[i] /= sqrt(head);
    }
}

__device__ static inline float store(float *output, float *input , int head)
{
    for (int i = 0; i < head; i++)
    {
        output[i] = input[i] ;
    }
}

__device__ static inline float sum(float *output, float *input , int head)
{
    
    for (int i = 0; i < head; i++)
    {
        output[i] += input[i] ;
    }
}

__device__ static inline float pull(float *output, float *input , int head)
{
    for (int i = 0; i < head; i++)
    {
        output[i] = input[i] ;
    }
}

__device__ static inline float sub(float *output, float *input , int head)
{
    for (int i = 0; i < head; i++)
    {
        output[i] -= input[i] ;
    }
}

__device__ static inline float my_exp(float *output, int head)
{
    for (int i = 0; i < head; i++)
    {
        output[i] = exp(output[i]) ;
    }
}

__device__ static inline float divide(float *output, float* input , int head)
{
    for (int i = 0; i < head; i++)
    {
        output[i] = output[i] / input[i];
    }
}

__device__ static inline float value_mul_alpha(float *output, float* input)
{
    for(int i = 0 ; i < 8 ; i++)
    {
        float temp = input[i];
        for (int j = 0; j < 8; j++)
        {
            output[i*8 + j] = output[i*8 + j] * temp;
        }
    }
    
}

__device__ static inline float sigmoid(float *output, int embed)
{
    for(int i = 0 ; i < embed ; i++)
    {
       output[i] = 1 / ( exp(-output[i]) + 1 );
    }
    
}


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    }
}

#define checkCudaErrors(x) check((x), #x, __FILE__, __LINE__)

template <typename scalar_t, uint32_t N_DIMS>
__global__ void graph_max_0_kernel(

    scalar_t *x,
    int64_t *edge_index_1,
    int64_t *edge_index_0,
    scalar_t *edge_attr,
    bool *bos_mask,
    scalar_t *rotate_mat,

    scalar_t *mat,
    scalar_t *mat_1,
    scalar_t *center_embed,

    int total_size,
    int edge_size,
    int x_size,

    scalar_t *embed0_weight,
    scalar_t *embed0_bias,
    scalar_t *embed3_weight,
    scalar_t *embed3_bias,
    scalar_t *embed6_weight,
    scalar_t *embed6_bias,

    scalar_t *embed1_weight,
    scalar_t *embed1_bias,
    scalar_t *embed4_weight,
    scalar_t *embed4_bias,
    scalar_t *embed7_weight,
    scalar_t *embed7_bias,

    scalar_t *bos_token,

    scalar_t *norm1_weight,
    scalar_t *norm1_bias,

    scalar_t *test_output

);

template <typename scalar_t, uint32_t N_DIMS>
__global__ void graph_max_1_kernel(

    scalar_t *x,
    int64_t *edge_index_1,
    int64_t *edge_index_0,
    scalar_t *edge_attr,
    bool *bos_mask,
    scalar_t *rotate_mat,

    scalar_t *mat,
    scalar_t *mat_1,
    scalar_t *center_embed,

    int total_size,
    int edge_size,
    int x_size,

    int64_t *output,
    int64_t *counts,
    int64_t *point_index,

    scalar_t *nbr_0_embed0_weight,
    scalar_t *nbr_0_embed0_bias,
    scalar_t *nbr_0_embed1_weight,
    scalar_t *nbr_0_embed1_bias,
    scalar_t *nbr_0_embed3_weight,
    scalar_t *nbr_0_embed3_bias,

    scalar_t *nbr_1_embed0_weight,
    scalar_t *nbr_1_embed0_bias,
    scalar_t *nbr_1_embed1_weight,
    scalar_t *nbr_1_embed1_bias,
    scalar_t *nbr_1_embed3_weight,
    scalar_t *nbr_1_embed3_bias,

    scalar_t *aggr_embed0_weight,
    scalar_t *aggr_embed0_bias,
    scalar_t *aggr_embed2_weight,
    scalar_t *aggr_embed2_bias,
    scalar_t *aggr_embed3_weight,
    scalar_t *aggr_embed3_bias,

    scalar_t *lin_q_weight,
    scalar_t *lin_q_bias,
    scalar_t *lin_k_weight,
    scalar_t *lin_k_bias,
    scalar_t *lin_v_weight,
    scalar_t *lin_v_bias,

    scalar_t *test_output

);