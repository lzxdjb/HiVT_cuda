from typing import Tuple, Optional
from torch import Tensor
from torch.utils.cpp_extension import load


def subgraph(
    subset: Tensor,
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:

    node_mask = subset

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
    return edge_index, edge_attr


def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    # print("dim",dim)
    size = [1] * ref.dim()
    # print("size",size)
    size[dim] = -1
    # print("ref",ref.shape)
    # print("before",src.shape)
    # print("after",src.view(size).shape)
    # print("final",src.view(size).expand_as(ref).shape)

    # print("size",size)
    # print("src.view",src.view(size).shape)
    # print("src.view(size).expand_as(ref)",src.view(size).expand_as(ref).shape)
    return src.view(size).expand_as(ref)


def scatter(src: Tensor, index: Tensor, dim: int,
            dim_size: int, reduce: str = 'sum') -> Tensor:
    # print("dim",src.dim())
    # print("dim",dim)
    dim = src.dim() + dim if dim < 0 else dim

    # print("dim_scatter",dim) #?? always 0 
    # print("src.shape",src.shape)
    size = list(src.size())
    
    # print("src.size",src.size())

    # print("size",size)
    # print("dim_size",dim_size)
    size[dim] = dim_size #why exe this code
    # print("dim",dim)
    # print("size[dim]",size[dim])

    # print("after size",size)
    # For "sum" and "mean" reduction, we make use of `scatter_add_`:
    if reduce == 'sum' or reduce == 'add':
        # print("before index",index.shape)
        index = broadcast(index, src, dim)
        # print("after index",index.shape)
        # print("src",src.shape)
        # print("src.new_zeros(size)",src.new_zeros(size).shape)
        # print("src.new_zeros(size).scatter_add_(dim, index, src)",src.new_zeros(size).scatter_add_(dim, index, src).shape)
        return src.new_zeros(size).scatter_add_(dim, index, src)

    # For "min" and "max" reduction, we prefer `scatter_reduce_` on CPU or
    # in case the input does not require gradients:
    if reduce == 'min' or reduce == 'max':
        index = broadcast(index, src, dim)
        return src.new_zeros(size).scatter_reduce_(
            dim, index, src, reduce=f'a{reduce}', include_self=False)

    raise ValueError(f"Encountered invalid `reduce` argument '{reduce}'")

# gat_cu = load(name="gat", sources=["ops/gat.cu"])


def softmax(
    src: Tensor,
    index: Tensor,
    num_nodes: int,
    dim: int = 0,
) -> Tensor:
    # print("num_nodes",num_nodes)
    N = num_nodes
    # print("src",src.shape)
    # print("index",index.shape)
    src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')  #why detach
    # print("src_max",src_max)
    # print("dim",dim)

    # src_max = gat_cu.scatter_max(src.detach(), index, dim, N) 

    # src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')

    out = src - src_max.index_select(dim, index)
    # print("out",out)
    out = out.exp()
    # print("out.exp()",out)

    out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
    out_sum = out_sum.index_select(dim, index)


    # print("result" , out / out_sum)
    return out / out_sum
