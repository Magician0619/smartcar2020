from module import *
from paddle import fluid
from paddle.fluid import Conv2D as Conv2d
from paddle.fluid.layers import concat


depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

anchors = [[10,13, 16,30, 33,23],  # P3/8
[30,61, 62,45, 59,119],  # P4/16
[116,90, 156,198, 373,326]]  # P5/32
na = len(anchors) // 2

nc = 80

class V5_Net(fluid.dygraph.Layer):
    def __init__(self):
        super(V5_Net, self).__init__()

        self.focus = Focus(ch_in=3, ch_out=36, k=3)
        self.con = ConvBNLayer(36, 64, 3, 2)

        self.bottleneck = BottleneckCSP(64, 64, 1)
        self.con1 = ConvBNLayer(64, 128, 3, 2)

        self.bottlenckcsp = BottleneckCSP(128, 128, 3)
        self.con2 = ConvBNLayer(128, 256, 3, 2)

        self.bottlenckcsp1 = BottleneckCSP(256, 256, 3)
        self.con3 = ConvBNLayer(256, 512, 3, 2)

        self.spp = SPP(512, 512, [5, 9, 13])

        self.bottlenckcsp2 = BottleneckCSP(512, 512, 2)
        self.bottlenckcsp3 = BottleneckCSP(512, 512, 1, False)
        self.conv = Conv2d(512, na*(nc + 5), 1, 1)                             # 调整类别改输出通道

        self.con4 = ConvBNLayer(768, 256, 1, 1, padding=0)
        self.bottlenckcsp4 = BottleneckCSP(256, 256, 1, False)
        self.conv2 = Conv2d(256, na*(nc + 5), 1, 1)                             # 调整类别改输出通道

        self.con5 = ConvBNLayer(384, 128, 1, 1, padding=0)
        self.bottlenckcsp5=BottleneckCSP(128, 128, 1, False)
        self.conv3 = Conv2d(128, na*(nc + 5), 1, 1)                             # 调整类别改输出通道

        # self.detect=Detect

    def forward(self, x):
        x = self.focus(x)                   # 320*320
        print(x.shape)
        x = self.con(x)                     # 160*160
        x = self.bottleneck(x)
        x = self.con1(x)
        print(x.shape)
        e = self.bottlenckcsp(x)            # 80*80
        print(e.shape)
        f = self.con2(e)                    # 40*40
        print(f.shape)
        g = self.bottlenckcsp1(f)           # 40*40       80*80
        print(g.shape)
        h = self.con3(g)                    #
        i = self.spp(h)                     #

        j = self.bottlenckcsp2(i)           #
        k = self.bottlenckcsp3(j)           #
        head1 = self.conv(k)                #

        m = fluid.layers.resize_nearest(k, out_shape=[40, 40])
        print('m:{}'.format(m.shape))           # 80*80
        n = concat([m, g], axis=1)        ## 80*80
        print(n.shape)
        o = self.con4(n)                    # 82 ?
        print('o:{}'.format(o.shape))
        p = self.bottlenckcsp4(o)           #
        print('p:{}'.format(p.shape))
        head2 = self.conv2(p)

        r = fluid.layers.resize_nearest(p, out_shape=[80, 80])
        s = concat([r, e], axis=1)        # 80*80
        t = self.con5(s)
        u = self.bottlenckcsp5(t)
        head3 = self.conv3(u)

        print('l:{}'.format(head1.shape))
        print('q:{}'.format(head2.shape))
        print("v:{}".format(head3.shape))
        return head1, head2, head3

if __name__ == '__main__':
    import numpy as np
    with fluid.dygraph.guard():
        x = np.random.randn(1, 3, 640, 640).astype('float32')
        x1 = fluid.dygraph.to_variable(x[..., ::2, ::2])
        x2 = fluid.dygraph.to_variable(x[..., 1::2, ::2])
        x3 = fluid.dygraph.to_variable(x[..., ::2, 1::2])
        x4 = fluid.dygraph.to_variable(x[..., 1::2, 1::2])
        x = fluid.layers.concat([x1, x2, x3, x4], axis=1)
        x = fluid.dygraph.to_variable(x)

        print(x.shape)
        net = V5_Net()
        head1, head2, head3 = net(x)
        print(head1.shape, head2.shape, head3.shape)
