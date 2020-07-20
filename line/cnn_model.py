import paddle.fluid as fluid


def cnn_model(image):
    conv1 = fluid.layers.conv2d(input=image, num_filters=32, filter_size=5, stride=2, act='relu')
    conv2 = fluid.layers.conv2d(input=conv1, num_filters=32, filter_size=5, stride=2)
    bn0 = fluid.layers.batch_norm(input=conv2,act='relu')
    conv3 = fluid.layers.conv2d(input=bn0, num_filters=64, filter_size=5, stride=2, act='relu')
    conv4 = fluid.layers.conv2d(input=conv3, num_filters=64, filter_size=3, stride=2)
    bn1 = fluid.layers.batch_norm(input=conv4,act='relu')
    conv5 = fluid.layers.conv2d(input=bn1, num_filters=128, filter_size=3, stride=1)
    # conv6 = fluid.layers.conv2d(input=conv5, num_filters=64, filter_size=3, stride=1)
    bn2 = fluid.layers.batch_norm(input=conv5,act='relu')
    # conv7 = fluid.layers.conv2d(input=conv6, num_filters=64, filter_size=3, stride=1, act='relu')
    fc1 = fluid.layers.fc(input=bn2, size=128, act=None)
    #drop_fc1 = fluid.layers.dropout(fc1, dropout_prob=0.1)
    fc2 = fluid.layers.fc(input=fc1, size=64, act=None)
    #drop_fc2 = fluid.layers.dropout(fc2, dropout_prob=0.1)
    predict = fluid.layers.fc(input=fc2, size=1)
    return predict




