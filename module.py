import math

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from paddle.fluid.dygraph.nn import Conv2D, BatchNorm

class ConvBNLayer(fluid.dygraph.Layer):
    #Standard convolution
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=1,
                 act="leaky",
                 is_test=True):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            num_channels=ch_in,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False,
            act=None)

        self.batch_norm = BatchNorm(
            num_channels=ch_out,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.)))
        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out
    
class Bottleneck(fluid.dygraph.Layer):
    # Standard bottleneck
    def __init__(self, ch_in, ch_out, shortcut=True, groups=1, e=0.5, is_test=True):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(ch_out * e)  # hidden channels
        self.cv1 = ConvBNLayer(ch_in, c_, 1, 1, 1, 0, is_test=is_test)
        self.cv2 = ConvBNLayer(c_, ch_out, 3, 1, groups=groups, is_test=is_test)
        self.add = shortcut and ch_in == ch_out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(fluid.dygraph.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, ch_in, ch_out, n=1, shortcut=True, groups=1, e=0.5, is_test=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(ch_out * e)  # hidden channels
        self.cv1 = ConvBNLayer(ch_in, c_, 1, 1, 1, 0, is_test=is_test)
        self.cv2 = Conv2D(ch_in, c_, 1, 1, bias_attr=False)
        self.cv3 = Conv2D(c_, c_, 1, 1, bias_attr=False)
        self.cv4 = ConvBNLayer(ch_out, ch_out, 1, 1, 1, 0, is_test=is_test)
        self.bn = BatchNorm(
            num_channels=2 * c_, # applied to cat(cv2, cv3)
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.)))
        self.m = fluid.dygraph.Sequential(*[Bottleneck(c_, c_, shortcut, groups, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        concated = fluid.layers.concat([y1, y2], axis=1)
        acted = fluid.layers.leaky_relu(x=self.bn(concated), alpha=0.1)
        return self.cv4(acted)

class SPP(fluid.dygraph.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, ch_in, ch_out, k=(5, 9, 13), is_test=True):
        super(SPP, self).__init__()
        c_ = ch_in // 2  # hidden channels
        self.cv1 = ConvBNLayer(ch_in, c_, 1, 1, 1, 0, is_test=is_test)
        self.cv2 = ConvBNLayer(c_ * (len(k) + 1), ch_out, 1, 1, 1, 0, is_test=is_test)
        self.m = fluid.dygraph.LayerList([fluid.dygraph.Pool2D(pool_size=x, pool_type='max', pool_stride=1, pool_padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(fluid.layers.concat([x] + [m(x) for m in self.m], axis=1))

class Focus(fluid.dygraph.Layer):
    # Focus wh information into c-space
    def __init__(self, ch_in, ch_out, k=1, is_test=True):
        super(Focus, self).__init__()
        self.conv = ConvBNLayer(ch_in*4, ch_out, k, 1, 1, 0, is_test=is_test)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        #return self.conv(fluid.layers.concat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], axis=1))
        #return self.conv(fluid.layers.space_to_depth(x, blocksize=2))
        return self.conv(x)

