#include "graph_h.cuh"

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

)
{
    int in_channel = 2;
    int embed = 64;
    int historcal_step = 20;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("idx = %d", idx);
    int small_idx = idx % x_size;

    int edge_begin = 0;
    int edge_end = 0;

    float temp1[64] = {0}; //store
    float temp2[64] = {0};

    float bmm1[2] = {0};
    float bmm2[2] = {0};

    if (idx >= total_size)
    {
        return;
    }
    else
    {
        bmm1[0] += x[idx * 2] * rotate_mat[small_idx * 4];
        bmm1[0] += x[idx * 2 + 1] * rotate_mat[small_idx * 4 + 2];
        bmm1[1] += x[idx * 2] * rotate_mat[small_idx * 4 + 1];
        bmm1[1] += x[idx * 2 + 1] * rotate_mat[small_idx * 4 + 3];
        // debug(bmm1, test_output + idx * 2, 2);

        linear(temp1, embed0_weight, embed0_bias, bmm1, embed, in_channel);
        layernorm(temp1, embed1_weight, embed1_bias, embed);
        relu(temp1, embed);
        linear(temp2, embed3_weight, embed3_bias, temp1, embed, embed);
        layernorm(temp2, embed4_weight, embed4_bias, embed);
        relu(temp2, embed);
        linear(temp1, embed6_weight, embed6_bias, temp2, embed, embed);
        layernorm(temp1, embed7_weight, embed7_bias, embed);

        if (bos_mask[idx / 64] == true)
        {
            for (int i = 0; i < embed; i++)
            {
                temp1[i] = bos_token[(idx / x_size) * 64 + i];
            }
        }

        layernorm(temp1, norm1_weight, norm1_bias, embed); //center_embed store in temp1
        debug(temp1, center_embed + idx * 64, 64);
    }
    //  __syncthreads();
}

template <typename scalar_t, uint32_t N_DIMS>
__global__ void graph_max_1_kernel(

    scalar_t *x,
    int64_t *edge_index_1,
    int64_t *edge_index_0,
    scalar_t *edge_attr,
    scalar_t *edge_ati_attr,
    bool *bos_mask,
    scalar_t *rotate_mat,

    scalar_t *mat,
    scalar_t *mat_1,
    scalar_t *center_embed,

    int total_size, //size[0];
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

    scalar_t *lin_ih_weight,
    scalar_t *lin_ih_bias,
    scalar_t *lin_hh_weight,
    scalar_t *lin_hh_bias,
    scalar_t *lin_self_weight,
    scalar_t *lin_self_bias,
    scalar_t *out_proj_weight,
    scalar_t *out_proj_bias,

    scalar_t *norm2_weight,
    scalar_t *norm2_bias,

    scalar_t *mlp0_weight,
    scalar_t *mlp0_bias,
    scalar_t *mlp3_weight,
    scalar_t *mlp3_bias,


    scalar_t *test_output

)
{
    int in_channel = 2;
    int embed = 64;
    int historcal_step = 20;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // printf("idx = %d", idx);
    int small_idx = idx % x_size;

    int edge_begin = 0;
    int edge_end = 0;

    float temp1[64] = {0}; //store
    float temp2[64] = {0};
    float temp3[64] = {0};
    float temp_center_embed[64] = {0};

    float big_temp[256] = {0};

    float alpha_temp[8] = {0};
    float alpha_max[8] = {-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10, -1e10};
    float alpha_sum[8] = {0};

    float bmm1[2] = {0};
    float bmm2[2] = {0};

    int flag = 0;

    if (idx >= total_size)
    {
        return;
    }
    else
    {

        if (idx != 0)
        {
            edge_begin = point_index[idx - 1];
            edge_end = point_index[idx];
        }
        else
        {
            edge_begin = 0;
            edge_end = point_index[idx]; //no include point_index[idx]!!
        }

        scalar_t* ptr = (scalar_t*)malloc((edge_end - edge_begin)*8*sizeof(scalar_t));
        scalar_t* v = (scalar_t*)malloc((edge_end - edge_begin)*64*sizeof(scalar_t));  
  
        for (int i = edge_begin; i < edge_end; i++)
        {
            float temp_0 = x[edge_index_1[i] * 2];
            float temp_1 = x[edge_index_1[i] * 2 + 1];

            linear(temp_center_embed, lin_q_weight, lin_q_bias, center_embed + edge_index_0[i] * 64, embed, embed);   //have no problem      

            bmm1[0] = temp_0 * rotate_mat[edge_index_0[i] % x_size * 4];
            bmm1[0] += temp_1 * rotate_mat[edge_index_0[i] % x_size * 4 + 2];
            bmm1[1] = temp_0 * rotate_mat[edge_index_0[i] % x_size * 4 + 1];
            bmm1[1] += temp_1 * rotate_mat[edge_index_0[i] % x_size * 4 + 3];

            linear(temp1, nbr_0_embed0_weight, nbr_0_embed0_bias, bmm1, embed, in_channel);
            layernorm(temp1, nbr_0_embed1_weight, nbr_0_embed1_bias, embed);
            relu(temp1, embed);
            linear(temp2, nbr_0_embed3_weight, nbr_0_embed3_bias, temp1, embed, embed);
            move(temp3, temp2, embed); //store in temp3

            bmm2[0] = edge_ati_attr[i * 2] * rotate_mat[edge_index_0[i] % x_size * 4] + edge_ati_attr[i * 2 + 1] * rotate_mat[edge_index_0[i] % x_size * 4 + 2];
            bmm2[1] = edge_ati_attr[i * 2] * rotate_mat[edge_index_0[i] % x_size * 4 + 1] + edge_ati_attr[i * 2 + 1] * rotate_mat[edge_index_0[i] % x_size * 4 + 3];

            linear(temp1, nbr_1_embed0_weight, nbr_1_embed0_bias, bmm2, embed, in_channel);
            layernorm(temp1, nbr_1_embed1_weight, nbr_1_embed1_bias, embed);
            relu(temp1, embed);
            linear(temp2, nbr_1_embed3_weight, nbr_1_embed3_bias, temp1, embed, embed);

            add(temp2, temp3, embed); //此处有问题
            layernorm(temp2, aggr_embed0_weight, aggr_embed0_bias, embed);
            relu(temp2, embed);
            linear(temp1, aggr_embed2_weight, aggr_embed2_bias, temp2, embed, embed);
            layernorm(temp1, aggr_embed3_weight, aggr_embed3_bias, embed); //store in temp1  //no problem

            linear(temp2, lin_k_weight, lin_k_bias, temp1, embed, embed); //k
            linear(temp3, lin_v_weight, lin_v_bias, temp1, embed, embed); //v

            store(v + (i - edge_begin)*64 , temp3 , 64); 

            mul(temp1 , temp_center_embed , temp2 , embed); //store in temp1

            sum_scale(alpha_temp , temp1 , 8);

            store(ptr + (i - edge_begin)*8 , alpha_temp , 8); 

            sum_scale_max(alpha_max , alpha_temp , 8);

        }

        for (int i = edge_begin; i < edge_end; i++)
        {
            pull(alpha_temp , ptr + (i - edge_begin)*8  , 8);
            sub(alpha_temp , alpha_max , 8);
            my_exp(alpha_temp , 8);
            sum(alpha_sum ,alpha_temp , 8);
            store(ptr + (i - edge_begin)*8 , alpha_temp , 8); 
        }

        clear(temp_center_embed , 64);

        for (int i = edge_begin; i < edge_end; i++)
        {
            pull(alpha_temp , ptr + (i - edge_begin)*8  , 8);
            divide(alpha_temp , alpha_sum , 8);
            pull(temp3 , v + (i - edge_begin)*64 , 64);
            value_mul_alpha(temp3 , alpha_temp );
            sum(temp_center_embed ,temp3 , 64); //temp_center-embed = input
        }
        free(ptr);
        free(v);

        store(temp2 , center_embed + idx*64 , 64); //center_embed store in temp2

        linear(temp1, lin_ih_weight , lin_ih_bias , temp_center_embed , 64 , 64);
        linear(temp3 , lin_hh_weight , lin_hh_bias , temp2 , 64 , 64);
        add(temp1 , temp3 , 64);
        sigmoid(temp1 , 64); //gate store in temp1

        linear(temp3 , lin_self_weight , lin_self_bias , temp2 , 64 , 64); //self.lin_self(center_embed)

        sub(temp3 , temp_center_embed , 64);
        mul(temp1 , temp1 , temp3 , 64);
        add(temp_center_embed , temp1 , 64) ; 

        linear(temp1 , out_proj_weight , out_proj_bias , temp_center_embed , 64 , 64);
        add(temp1 , temp2 , 64); //center_embed = temp1

        store(temp2 , temp1 , 64); //center_embed = temp2 ;here is correct

        layernorm(temp1, norm2_weight, norm2_bias, embed); //self.norm2(center_embed)
        linear(big_temp ,mlp0_weight , mlp0_bias , temp1 , 64*4 , 64);
        relu(big_temp , 64*4);
        linear(temp1 , mlp3_weight , mlp3_bias ,big_temp, 64 , 64 *4); //_ff_block

        add(temp2 , temp1 , 64);
        debug(temp2 , test_output + idx * 64, 64); 

        





       
    }
}

at::Tensor graph_max(
    torch::Tensor x,
    torch::Tensor edge_index_1,
    torch::Tensor edge_index_0,

    torch::Tensor edge_attr,
    torch::Tensor edge_ati_attr,
    torch::Tensor bos_mask,
    torch::Tensor rotate_mat,
    torch::Tensor mat_1,

    torch::Tensor embed0_weight,
    torch::Tensor embed0_bias,
    torch::Tensor embed3_weight,
    torch::Tensor embed3_bias,
    torch::Tensor embed6_weight,
    torch::Tensor embed6_bias,

    torch::Tensor embed1_weight,
    torch::Tensor embed1_bias,
    torch::Tensor embed4_weight,
    torch::Tensor embed4_bias,
    torch::Tensor embed7_weight,
    torch::Tensor embed7_bias,

    torch::Tensor bos_token,

    torch::Tensor norm1_weight,
    torch::Tensor norm1_bias,

    torch::Tensor output,
    torch::Tensor counts,

    torch::Tensor nbr_0_embed0_weight,
    torch::Tensor nbr_0_embed0_bias,
    torch::Tensor nbr_0_embed1_weight,
    torch::Tensor nbr_0_embed1_bias,
    torch::Tensor nbr_0_embed3_weight,
    torch::Tensor nbr_0_embed3_bias,

    torch::Tensor nbr_1_embed0_weight,
    torch::Tensor nbr_1_embed0_bias,
    torch::Tensor nbr_1_embed1_weight,
    torch::Tensor nbr_1_embed1_bias,
    torch::Tensor nbr_1_embed3_weight,
    torch::Tensor nbr_1_embed3_bias,

    torch::Tensor aggr_embed0_weight,
    torch::Tensor aggr_embed0_bias,
    torch::Tensor aggr_embed2_weight,
    torch::Tensor aggr_embed2_bias,
    torch::Tensor aggr_embed3_weight,
    torch::Tensor aggr_embed3_bias,

    torch::Tensor lin_q_weight,
    torch::Tensor lin_q_bias,
    torch::Tensor lin_k_weight,
    torch::Tensor lin_k_bias,
    torch::Tensor lin_v_weight,
    torch::Tensor lin_v_bias,


    torch::Tensor lin_ih_weight,
    torch::Tensor lin_ih_bias,
    torch::Tensor lin_hh_weight,
    torch::Tensor lin_hh_bias,
    torch::Tensor lin_self_weight,
    torch::Tensor lin_self_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias,

    torch::Tensor norm2_weight,
    torch::Tensor norm2_bias,

    torch::Tensor mlp0_weight,
    torch::Tensor mlp0_bias,
    torch::Tensor mlp3_weight,
    torch::Tensor mlp3_bias,
 
    torch::Tensor test_weight_linear,
    torch::Tensor test_bias_linear,
    torch::Tensor input

)

{

    auto size = x.sizes().vec(); // x ; point_vertex_0
    auto edge_index_size = edge_index_1.sizes();

    /////
    auto test_output = x.new_zeros({size[0], 64}); // you need modify it when test!!!!
    /////

    auto mat = x.new_zeros(size);
    auto center_embed = x.new_zeros({size[0], 64});
    auto result = x.new_zeros({size[0], 64});

    auto point_index = x.new_zeros({size[0]}, torch::kInt64);

    const int threads = 8;
    const int blocks = (x.sizes()[0] / threads) + 1;
    const int total_size = x.sizes()[0]; //x_size

    int x_size = x.sizes()[0] / 20;
    int edge_size = edge_index_size[0];

    auto point_vertex_0 = x.new_zeros({size[0]});

    int64_t *output_ptr = output.data<int64_t>();
    int64_t *counts_ptr = counts.data<int64_t>();
    int64_t *point_index_ptr = point_index.data<int64_t>();

    int64_t *output_cpu = new int64_t[output.sizes()[0]];
    int64_t *point_index_cpu = new int64_t[point_index.sizes()[0]];
    int64_t *counts_cpu = new int64_t[counts.sizes()[0]];

    checkCudaErrors(cudaMemcpy(output_cpu, output_ptr, output.sizes()[0] * sizeof(int64_t), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(point_index_cpu, point_index_ptr, point_index.sizes()[0] * sizeof(int64_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(counts_cpu, counts_ptr, counts.sizes()[0] * sizeof(int64_t), cudaMemcpyDeviceToHost));

    int j = -1;
    int i = 0;
    for (i = 0; i < size[0]; i++)
    {
        if (j < 0)
        {
            point_index_cpu[i] = 0;
            if (i == output_cpu[0])
                j++;
            else
                continue;
        }
        if (i == output_cpu[j])
        {
            point_index_cpu[i] = counts_cpu[j];

            j++;
        }
        else
        {
            point_index_cpu[i] = counts_cpu[j - 1];
        }
    }

    checkCudaErrors(cudaMemcpy(point_index_ptr, point_index_cpu, point_index.sizes()[0] * sizeof(int64_t), cudaMemcpyHostToDevice));

    delete[] output_cpu;
    delete[] counts_cpu;
    delete[] point_index_cpu;

    checkCudaErrors(cudaDeviceSynchronize());

    graph_max_0_kernel<float, 8><<<blocks, threads>>>(
        x.data<float>(),
        edge_index_1.data<int64_t>(),
        edge_index_0.data<int64_t>(),
        edge_attr.data<float>(),
        bos_mask.data<bool>(),
        rotate_mat.data<float>(),

        mat.data<float>(),
        mat_1.data<float>(),
        center_embed.data<float>(),

        total_size,
        edge_size,
        x_size, //mat

        embed0_weight.data<float>(),
        embed0_bias.data<float>(),
        embed3_weight.data<float>(),
        embed3_bias.data<float>(),
        embed6_weight.data<float>(),
        embed6_bias.data<float>(),

        embed1_weight.data<float>(),
        embed1_bias.data<float>(),
        embed4_weight.data<float>(),
        embed4_bias.data<float>(),
        embed7_weight.data<float>(),
        embed7_bias.data<float>(),

        bos_token.data<float>(),

        norm1_weight.data<float>(),
        norm1_bias.data<float>(),

        test_output.data<float>()

    );
    checkCudaErrors(cudaDeviceSynchronize());

    graph_max_1_kernel<float, 8><<<blocks, threads>>>(
        x.data<float>(),
        edge_index_1.data<int64_t>(),
        edge_index_0.data<int64_t>(),
        edge_attr.data<float>(),
        edge_ati_attr.data<float>(),
        bos_mask.data<bool>(),
        rotate_mat.data<float>(),

        mat.data<float>(),
        mat_1.data<float>(),
        center_embed.data<float>(),

        total_size,
        edge_size,
        x_size, //mat

        output.data<int64_t>(),
        counts.data<int64_t>(),
        point_index.data<int64_t>(),

        nbr_0_embed0_weight.data<float>(),
        nbr_0_embed0_bias.data<float>(),
        nbr_0_embed1_weight.data<float>(),
        nbr_0_embed1_bias.data<float>(),
        nbr_0_embed3_weight.data<float>(),
        nbr_0_embed3_bias.data<float>(),

        nbr_1_embed0_weight.data<float>(),
        nbr_1_embed0_bias.data<float>(),
        nbr_1_embed1_weight.data<float>(),
        nbr_1_embed1_bias.data<float>(),
        nbr_1_embed3_weight.data<float>(),
        nbr_1_embed3_bias.data<float>(),

        aggr_embed0_weight.data<float>(),
        aggr_embed0_bias.data<float>(),
        aggr_embed2_weight.data<float>(),
        aggr_embed2_bias.data<float>(),
        aggr_embed3_weight.data<float>(),
        aggr_embed3_bias.data<float>(),

        lin_q_weight.data<float>(),
        lin_q_bias.data<float>(),
        lin_k_weight.data<float>(),
        lin_k_bias.data<float>(),
        lin_v_weight.data<float>(),
        lin_v_bias.data<float>(),


        lin_ih_weight.data<float>(),
        lin_ih_bias.data<float>(),
        lin_hh_weight.data<float>(),
        lin_hh_bias.data<float>(),
        lin_self_weight.data<float>(),
        lin_self_bias.data<float>(),
        out_proj_weight.data<float>(),
        out_proj_bias.data<float>(),

        norm2_weight.data<float>(),
        norm2_bias.data<float>(),

        mlp0_weight.data<float>(),
        mlp0_bias.data<float>(),
        mlp3_weight.data<float>(),
        mlp3_bias.data<float>(),

        test_output.data<float>()

    );
    checkCudaErrors(cudaDeviceSynchronize());

    // return input;  //for test (relu or layernorm )

    return test_output; // for test (linear and all module)

    return center_embed;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("graph_max", &graph_max, "graph_max");
}
