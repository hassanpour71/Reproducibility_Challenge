import torch
from torch.nn.modules.conv import _ConvNd, Conv2d
from torch.nn.modules.utils import _single, _pair, _triple





# Linear Approximation
def linear_bernoulli_sampling(A,B,bias,sample_ratio, minimal_k, scale,sample_ratio_bwd=None,minimal_k_bwd=None,sample_ratio_wu=None,minimal_k_wu=None):

    in_features = A.size()[-1]
    device = A.device    

    # calculate the number of column-row pairs to sample
    k_candidate = int(float(in_features)*sample_ratio)

    # make k at least minimal_k
    k = min(max(k_candidate,minimal_k),in_features)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full matmul instead of approximating
    if k == in_features:
        # no need to sample. perform normal matmul
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C = torch.addmm(bias, A, B)
        else:
            C = torch.matmul(A, B)
            if bias is not None:
                C += bias
        return C

    with torch.no_grad():
        # calculate norms of the columns of A and rows of B
        if A.dim() == 2:   
            a_col_norms = torch.norm(A,dim=0)
        else:
            # since we sample across in_featuers, consider other dimensions as a single dimension for sampling purpuses
            a_col_norms = torch.norm(A.view(-1,in_features),dim=0)    
        
        b_row_norms = torch.norm(B,dim=1)

        # multiply both norms element-wise
        norm_mult = torch.mul(a_col_norms,b_row_norms)
        sum_norm_mult = norm_mult.sum()       
 
        # calculate number of nonzero elements in norm_mult. this serves 
        # two purposes:
        # 1. Possibly reduce number of sampled pairs, as zero elements in norm_mult will not contribute to the result
        # 2. Prevents scaling of zero values
        nnz = (norm_mult!=0).sum()
        if nnz == 0:
            #print("zero multiply detected! scenario not optimzied (todo)")
            return torch.nn.functional.linear(A, B.t(),bias)
            
        k = min(k,nnz)
        
        prob_dist = k * torch.div(norm_mult,sum_norm_mult)
        prob_dist = prob_dist.clamp(min=0, max=1)

        # use epsilon-optimal sampling to allow learning random weights and to bound the scaling factor
        epsilon = 0.1
        if epsilon > 0:
            uniform = torch.ones_like(prob_dist)/in_features 
            prob_dist = (1-epsilon)*prob_dist + epsilon*uniform

        indices = torch.bernoulli(prob_dist).nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            print("no elements selected - hmm")
            indices = torch.arange(k, device=device)
    
    # sample column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A, dim=-1, index=indices)
    B_top_k_rows = torch.index_select(B, dim=0, index=indices)
    
    if scale == True:
        # scale column-row pairs by 1/(p_i) to get unbiased estimation
        with torch.no_grad():
            scale_factors = torch.div(1,prob_dist)
            scale_matrix = torch.diag(scale_factors[indices])
        A_top_k_cols = torch.matmul(A_top_k_cols, scale_matrix)
            
    # multiply smaller matrices
    if sample_ratio_bwd is None and sample_ratio_wu is None:
        if A.dim() == 2 and bias is not None:
            # fused op is marginally faster
            C_approx = torch.addmm(bias, A_top_k_cols, B_top_k_rows)
        else:
            C_approx = torch.matmul(A_top_k_cols, B_top_k_rows)
            if bias is not None:
                C_approx += bias
    else:
        # The following code will be used to apply additional sampling in the backward pass but update only the
        # sub-tensors sampled in the forward pass.
        # For simplicity, we don't optimize for torch.addmm usage in this case
        C_approx = matmul_approx_bwd_func.apply(A_top_k_cols, B_top_k_rows,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)
        if bias is not None:
            C_approx += bias
    
    return C_approx


def approx_linear_forward_xA_b(input,weight,bias,sample_ratio,minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu):
    r"""
    Applies approximate linear transformation to the incoming data: :math:`y = xA + b`.
    Note: A is assumed not transposed
    the matrix multiply xA is approximated
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(in\_features, out\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    #return linear_top_k(input,weight,bias,sample_ratio, minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)
    #return linear_top_k_approx(input,weight,bias,sample_ratio, minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)
    #return linear_uniform_sampling(input,weight,bias,sample_ratio, minimal_k)
    #return linear_random_sampling(input,weight,bias,sample_ratio, minimal_k, with_replacement=True, optimal_prob=True, scale=True,sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd=minimal_k_bwd, sample_ratio_wu=sample_ratio_wu, minimal_k_wu=minimal_k_wu)
    #return approx_linear_xA_b.topk(input,weight,bias,sample_ratio, minimal_k)
    return linear_bernoulli_sampling(input,weight,bias,sample_ratio, minimal_k, scale=True,sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd=minimal_k_bwd, sample_ratio_wu=sample_ratio_wu, minimal_k_wu=minimal_k_wu)
''' Approximates the matrix multiply A*B+b by sampling the column-row pairs with the largest norm
    A - input matrix, shape (N,*,in_features) where '*' means any number of additional dimensions
    B - input matrices, shape (in_features, out_features)
    bias - bias vector, shape (out_features)
    sample_ratio - Ratio of column-row pairs to sample
    minimal_k - Minimal number of column-row pairs to keep in the sampling
    note: B is not transposed
    output: A*B+b, shape (N,*,out_features)
'''

def approx_linear_forward(input,weight,bias,sample_ratio,minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu):
    r"""
    Applies approximate linear transformation to the incoming data: :math:`y = xA^T + b`.
    the matrix multiply xA^T is approximated
    note: weight transposition is done in this function
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """


    #return torch.nn.functional.linear(input,weight,bias)
    return approx_linear_forward_xA_b(input,weight.t(),bias,sample_ratio, minimal_k,sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu)


class approx_linear_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias,sample_ratio,minimal_k, sample_ratio_bwd, minimal_k_bwd, sample_ratio_wu, minimal_k_wu):
        ctx.save_for_backward(inputs,weights,bias)
        #store non-tensor objects in ctx
        ctx.sample_ratio_bwd = sample_ratio_bwd
        ctx.minimal_k_bwd = minimal_k_bwd
        ctx.sample_ratio_wu = sample_ratio_wu
        ctx.minimal_k_wu = minimal_k_wu
        #return approx_linear_forward(inputs, weights, bias,sample_ratio,minimal_k,None,None,None,None)
        return torch.nn.functional.linear(inputs,weights,bias)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights,bias = ctx.saved_tensors
        if bias is not None:
            grad_bias = torch.sum(grad_output,0)
        else:
            grad_bias = None        

        if ctx.minimal_k_bwd is None:
            grad_input = torch.matmul(grad_output, weights)
        else:
            grad_input = approx_linear_forward(grad_output, weights.t(), None,ctx.sample_ratio_bwd,ctx.minimal_k_bwd,None,None,None,None)

        if ctx.minimal_k_wu is None:
            grad_weight = torch.matmul(grad_output.t(),inputs)
        else:
            grad_weight = approx_linear_forward(grad_output.t(), inputs.t(), None,ctx.sample_ratio_wu,ctx.minimal_k_wu,None,None,None,None)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class approx_Linear(torch.nn.modules.Linear):
    """Applies approximate Linear transformation to the incoming data:



    """

    def __init__(self, in_features, out_features, bias=True,
                 sample_ratio=1.0, minimal_k=1,
                 sample_ratio_bwd=None, minimal_k_bwd = None,
                 sample_ratio_wu=None, minimal_k_wu=None):
        self.sample_ratio = sample_ratio
        self.minimal_k = minimal_k
        self.sample_ratio_bwd = sample_ratio_bwd
        self.minimal_k_bwd = minimal_k_bwd
        self.sample_ratio_wu = sample_ratio_wu
        self.minimal_k_wu = minimal_k_wu
        super(approx_Linear, self).__init__(
            in_features,out_features,bias)

    def forward(self, input):
        if self.training is True:
            # Use approximation in training only.
            return approx_linear_func.apply(input, self.weight, self.bias, self.sample_ratio, self.minimal_k,self.sample_ratio_bwd,self.minimal_k_bwd,self.sample_ratio_wu,self.minimal_k_wu)
            #return approx_linear_forward(input, self.weight, self.bias, self.sample_ratio, self.minimal_k,self.sample_ratio_bwd,self.minimal_k_bwd,self.sample_ratio_wu,self.minimal_k_wu)
        else:
            # For evaluation, perform the exact computation
            return torch.nn.functional.linear(input, self.weight, self.bias)


class matmul_approx_bwd_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, sample_ratio_bwd,minimal_k_bwd,sample_ratio_wu,minimal_k_wu):
        ctx.save_for_backward(inputs,weights)
        #store non-tensor objects in ctx
        ctx.sample_ratio_bwd = sample_ratio_bwd
        ctx.minimal_k_bwd = minimal_k_bwd
        ctx.sample_ratio_wu = sample_ratio_wu
        ctx.minimal_k_wu = minimal_k_wu
        return torch.matmul(inputs, weights)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors

        #print('calculating matmul_approx_bwd_func bwd pass! sample_ratio_bwd={},minimal_k_bwd={},sample_ratio_wu={},minimal_k_wu={}'.format(ctx.sample_ratio_bwd,ctx.minimal_k_bwd,ctx.sample_ratio_wu,ctx.minimal_k_wu))
        
        #grad_input = torch.matmul(grad_output, weights.t())
        grad_input = approx_linear_forward_xA_b(grad_output, weights.t(), None, ctx.sample_ratio_bwd, ctx.minimal_k_bwd,None,None,None,None)
        
        #grad_weight = torch.matmul(inputs.t(),grad_output)
        grad_weight = approx_linear_forward_xA_b(inputs.t(), grad_output, None, ctx.sample_ratio_wu, ctx.minimal_k_wu, None, None, None, None)
        
        return grad_input, grad_weight, None, None, None, None


# Convolution Approximation







def topk_indices(data, k):
    
    return data.argsort(descending=True)[:k]

def conv2d_top_k(A,B,bias, stride, padding, dilation, groups, sample_ratio, minimal_k):
    
    in_channels = A.size()[1]
    
    # calculate the number of channels to sample for the forward propagation phase
    k_candidate = int(float(in_channels)*sample_ratio)

    # make k at least min_clrows (similar to meProp)
    k = min(max(k_candidate,minimal_k),in_channels)
    
    # if because of minimal_k or sample_ratio k equals the number of features, perform full conv2d instead of approximating
    if k == in_channels:
        return torch.nn.functional.conv2d(input=A, weight=B, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # calculate norms of the columns of A and rows of B
    with torch.no_grad():
        a_channel_norms = torch.norm(A.view(A.size()[0],A.size()[1],-1),p=2, dim=2)
        a_channel_norms = torch.norm(a_channel_norms, dim=0, p=2)
        a_channel_norms = torch.squeeze(a_channel_norms)

        b_channel_norms = torch.norm(B.view(B.size()[0],B.size()[1],-1),p=2, dim=2)
        b_channel_norms = torch.norm(b_channel_norms, dim=0, p=2)
        b_channel_norms = torch.squeeze(b_channel_norms)

        # multiply both norms element-wise to and pick the indices of the top K column-row pairs
        norm_mult = torch.mul(a_channel_norms,b_channel_norms)

        #top_k_indices = torch.topk(norm_mult,k)[1]
        top_k_indices = topk_indices(norm_mult,k)

    # pick top-k column-row pairs to form new smaller matrices
    A_top_k_cols = torch.index_select(A,dim = 1, index = top_k_indices)
    B_top_k_rows = torch.index_select(B,dim = 1, index = top_k_indices)

    # multiply smaller matrices
    C_approx = torch.nn.functional.conv2d(input=A_top_k_cols, weight=B_top_k_rows, bias=bias, stride=stride, padding=padding,
                                          dilation=dilation, groups=groups)
    return C_approx


''' Approximates 2d convolution with channel sampling according to largest norm
    A - input tensor, shape (batch, in_channels, h, w)
    B - input matrices, shape (out_channels, in_channels, kw, kw)
    bias - bias vector, shape (out_channels)
    stride, padding, dilation, groups as in regular conv2d
    sample_ratio - Ratio of in_channels to sample
    minimal_k - Minimal number of in_channels to keep in the sampling
'''

def approx_conv2d_func_forward(A,B,bias, stride, padding, dilation, groups, sample_ratio,minimal_k):
    return conv2d_top_k(A=A.float(),B=B.float(),bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups, sample_ratio=sample_ratio, minimal_k=minimal_k)
   




class approx_Conv2d(_ConvNd):
    """Applies approximate 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,padding_mode='zeros',
                 sample_ratio=1.0, minimal_k=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.sample_ratio = sample_ratio
        self.minimal_k = minimal_k
        super(approx_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,padding_mode)

    def forward(self, input):
        # Use approximation in training only.
        if self.training is True:
            return approx_conv2d_func_forward(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.sample_ratio, self.minimal_k)
        else:
            # For evaluation, perform the exact computation
            return torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
