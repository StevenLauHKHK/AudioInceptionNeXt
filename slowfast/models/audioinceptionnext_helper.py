import torch.nn as nn


def get_trans_func(name):
    """
    Retrieves the transformation module by name.
    """
    trans_funcs = {
        "ms_threeway_dw_bottleneck_transform": MSThreeWayDWBottleneckTransform,
    }
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]


class MSThreeWayDWBottleneckTransform(nn.Module):
    """
    Bottleneck transformation: Tx1, 1x3, 1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        freq_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm2d,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            freq_kernel_size (int): the frequency kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                 default is nn.BatchNorm2d.
        """
        super(MSThreeWayDWBottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self.freq_kernel_size = freq_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._stride_1x1 = stride_1x1
        self.dim_in = dim_in
        self.dim_out = dim_out
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            dilation,
            norm_module,
        )
        

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        dilation,
        norm_module,
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # 1x1, BN, ReLU.
        if dim_in != dim_out:
            self.a_head = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=[1, 1],
                stride=stride,
                padding=[0, 0],
                bias=False,
            )
            self.a_head_bn = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
            
            self.a = nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=[self.temp_kernel_size, 1],
                stride=[1, str1x1],
                padding=[int(self.temp_kernel_size // 2), 0],
                bias=False,
                groups=dim_out
            )

            self.a_small = nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=[3, 1],
                stride=[1, str1x1],
                padding=[int(3 // 2), 0],
                bias=False,
                groups=dim_out
            )

            self.a_medium = nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=[11, 1],
                stride=[1, str1x1],
                padding=[int(11 // 2), 0],
                bias=False,
                groups=dim_out
            )
        else:
            self.a_head = nn.Conv2d(
                dim_out,
                dim_out,
                kernel_size=[1, 1],
                stride=[1, 1],
                padding=[0, 0],
                bias=False,
            )
            self.a_head_bn = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )

            self.a = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=[self.temp_kernel_size, 1],
                stride=[1, str1x1],
                padding=[int(self.temp_kernel_size // 2), 0],
                bias=False,
                groups=dim_out
            )

            self.a_small = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=[3, 1],
                stride=[1, str1x1],
                padding=[int(3 // 2), 0],
                bias=False,
                groups=dim_out
            )

            self.a_medium = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=[11, 1],
                stride=[1, str1x1],
                padding=[int(11 // 2), 0],
                bias=False,
                groups=dim_out
            )
        self.a_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)


        # 1xF, BN, ReLU.
        self.b = nn.Conv2d(
            dim_out,
            dim_out,
            [1, self.freq_kernel_size],
            stride=[1, 1],
            padding=[0, int(self.freq_kernel_size // 2)],
            groups=dim_out,
            bias=False,
            dilation=[1, dilation],
        )
        self.b_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        self.a_bn_small = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu_small = nn.ReLU(inplace=self._inplace_relu)

        self.a_bn_medium = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.a_relu_medium = nn.ReLU(inplace=self._inplace_relu)

        # 1x3, BN, ReLU.
        self.b_small = nn.Conv2d(
            dim_out,
            dim_out,
            [1, 3],
            stride=[1, 1],
            padding=[0, int(3 // 2)],
            groups=dim_out,
            bias=False,
            dilation=[1, dilation],
        )
        self.b_bn_small = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu_small = nn.ReLU(inplace=self._inplace_relu)

        self.b_medium = nn.Conv2d(
            dim_out,
            dim_out,
            [1, 11],
            stride=[1, 1],
            padding=[0, int(11 // 2)],
            groups=dim_out,
            bias=False,
            dilation=[1, dilation],
        )
        self.b_bn_medium = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.b_relu_medium = nn.ReLU(inplace=self._inplace_relu)

        # 1x1, BN. Expand layer
        self.c = nn.Conv2d(
            dim_out,
            dim_inner,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            bias=False,
        )
        self.c_bn = norm_module(
            num_features=dim_inner, eps=self._eps, momentum=self._bn_mmt
        )

        # 1x1, BN.
        self.d = nn.Conv2d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            bias=False,
        )
        self.d_bn = norm_module(
            num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
        )
        self.d_bn.transform_final_bn = True

    def forward(self, x):
        x_head = self.a_head(x)
        x_head = self.a_head_bn(x_head)

        x_1 = self.a(x_head)
        x_1 = self.a_bn(x_1)
        x_1 = self.a_relu(x_1)

        x_1 = self.b(x_1)
        x_1 = self.b_bn(x_1)
        x_1 = self.b_relu(x_1)

        x_3 = self.a_medium(x_head)
        x_3 = self.a_bn_medium(x_3)
        x_3 = self.a_relu_medium(x_3)

        x_3 = self.b_medium(x_3)
        x_3 = self.b_bn_medium(x_3)
        x_3 = self.b_relu_medium(x_3)

        # Branch2a_small_kernel.
        x_2 = self.a_small(x_head)
        x_2 = self.a_bn_small(x_2)
        x_2 = self.a_relu_small(x_2)

        # Branch2b_small_kernel.
        x_2 = self.b_small(x_2)
        x_2 = self.b_bn_small(x_2)
        x_2 = self.b_relu_small(x_2)

        x_comb = x_1 + x_2 + x_3

        # Branch2c
        x_comb = self.c(x_comb)
        x_comb = self.c_bn(x_comb)

        x_comb = self.d(x_comb)
        x_comb = self.d_bn(x_comb)

        return x_comb

class AudioInceptionNeXtBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        freq_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        dilation=1,
        norm_module=nn.BatchNorm2d,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            dilation (int): size of dilation.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                 default is nn.BatchNorm2d.
        """
        super(AudioInceptionNeXtBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._bn_mmt = bn_mmt
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            freq_kernel_size,
            stride,
            trans_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        freq_kernel_size,
        stride,
        trans_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
    ):
        
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride[0] != 1) or (stride[1] != 1):
            self.branch1 = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[stride[0], stride[1]],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = norm_module(
                num_features=dim_out, eps=self._eps, momentum=self._bn_mmt
            )
        self.branch2 = trans_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            freq_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
            norm_module=norm_module,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def forward(self, x):
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + self.branch2(x)
        else:
            x = x + self.branch2(x)
        x = self.relu(x)
        return x

class AudioInceptionNeXtStage(nn.Module):
    """
    Stage of 2D ResNet. It expects to have one or more tensors as input for
        single pathway (Slow, Fast), and multi-pathway (SlowFast) cases.
        More details can be found here:

        Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen.
        "Auditory Slow-Fast Networks for Audio Recognition"

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "SlowFast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_sizes,
        freq_kernel_sizes,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        num_block_freq_kernel,
        dilation,
        trans_func_name="bottleneck_transform",
        stride_1x1=False,
        inplace_relu=True,
        norm_module=nn.BatchNorm2d,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            freq_kernel_sizes (list): list of the p frequency kernel sizes of the
                convolution in the bottleneck. Different freq_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            num_block_freq_kernel (list): extent the freq_kernel_sizes to
                num_block_freq_kernel blocks, then fill frequency kernel size
                of 1 for the rest of the layers.
            dilation (list): size of dilation for each pathway.
            trans_func_name (string): name of the the transformation function apply
                on the network.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                 default is nn.BatchNorm2d.
        """
        super(AudioInceptionNeXtStage, self).__init__()
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        assert all(
            (
                num_block_freq_kernel[i] <= num_blocks[i]
                for i in range(len(freq_kernel_sizes))
            )
        )
        self.num_blocks = num_blocks
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        self.freq_kernel_sizes = [
            (freq_kernel_sizes[i] * num_blocks[i])[: num_block_freq_kernel[i]]
            + [1] * (num_blocks[i] - num_block_freq_kernel[i])
            for i in range(len(freq_kernel_sizes))
        ]

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(freq_kernel_sizes),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                }
            )
            == 1
        )
        self.num_pathways = len(self.num_blocks)
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            trans_func_name,
            stride_1x1,
            inplace_relu,
            dilation,
            norm_module,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        trans_func_name,
        stride_1x1,
        inplace_relu,
        dilation,
        norm_module,
    ):
        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                # Retrieve the transformation function.
                trans_func = get_trans_func(trans_func_name)
                # Construct the block.
                res_block = AudioInceptionNeXtBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    self.freq_kernel_sizes[pathway][i],
                    [stride[0][pathway], stride[1][pathway]] if i == 0 else [1, 1],
                    trans_func,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=norm_module,
                )
                self.add_module("pathway{}_res{}".format(pathway, i), res_block)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
            output.append(x)

        return output