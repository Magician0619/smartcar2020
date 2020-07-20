import paddle.fluid as fluid


def cnn_model(image):
    temp = fluid.layers.conv2d(input=image, num_filters=32, filter_size=5, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=32, filter_size=5, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=5, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=3, stride=2, act='relu')
    temp = fluid.layers.conv2d(input=temp, num_filters=128, filter_size=3, stride=1, act='relu')
    # temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=3, stride=1, act='relu')
    # temp = fluid.layers.conv2d(input=temp, num_filters=64, filter_size=3, stride=1, act='relu')
    fc1 = fluid.layers.fc(input=temp, size=128, act=None)
    drop_fc1 = fluid.layers.dropout(fc1, dropout_prob=0.1)
    fc2 = fluid.layers.fc(input=drop_fc1, size=64, act=None)
    drop_fc2 = fluid.layers.dropout(fc2, dropout_prob=0.1)
    predict = fluid.layers.fc(input=drop_fc2, size=1, act=None)
    predict = fluid.layers.tanh(predict / 4)
    return predict




