import tensorflow as tf
import numpy as np

epsilon = 1e-9

class CapsLayer(object):
    '''
    capsule layers
    paramters:
        input: 4-D tensor(N,W,H,D)
        num_outputs: the number of capsule in this layer
        length: int, the length of output vector of capsule, only used between primaryCapsule and digitCapsule
        layer_type: string, there are two types of connection, CONV-CAP and CAP-CAP(FC)
        with_routing: bool, routing algorithm between primary and digit

    Return:
        4-D tensor
    '''

    def __init__(self, num_outputs, length, with_routing = False,layer_type = 'CONV'):
        self.num_outputs = num_outputs
        self.length = length
        self.layer_type = layer_type
        self.with_routing = with_routing

    def __call__(self, input_x, filter=None, stride=None):
        if self.layer_type =='CONV':
            self.filter = filter
            self.stride = stride

            if not self.with_routing:
                assert  input_x.get_shape() == [128, 20, 20, 256]

                #caps = tf.nn.conv2d(input_x, filter=self.filter, strides=self.stride, padding='VALID')
                caps = tf.contrib.layers.conv2d(input_x, 256, self.filter,self.stride, padding='VALID')
                caps = tf.nn.relu(caps)

                caps = tf.reshape(caps, (128, -1, self.length, 1))
                Primarycaps = squash(caps)

                assert Primarycaps.get_shape() == [128,1152,8,1]
                return Primarycaps

        if self.layer_type == 'FC':
            if self.with_routing:
                self.input_x = tf.reshape(input_x, shape = (128, -1, 1, input_x.shape[-2].value, 1))

                with tf.variable_scope('routing'):
                    # b_ij:[batch, num_caps_!, num_caps1_plus_!,1,1]
                    b_ij = tf.constant(np.zeros([128, input_x.shape[1].value, self.num_outputs, 1, 1]), dtype=np.float32)
                    caps = routing(self.input_x, b_ij)
                    capsule = tf.squeeze(caps, axis=1)

            return capsule

def routing(input_x, b_ij):
    '''
    :param input_x:  A tensor with [128, num_caps = 1152, 1, length = 8, 1]
                    shape, num_caps_1 meaning the number of capsule in the layer l.
    :param b_ij:  A tensor of shape [128, num_caps_1_plus_1, length(v_j) = 16, 1]

    Notes: u_i represents the vector output of capsule i in the layrer l, and v_j
            the vector output of capsules j in the layer l+1
    :return:
    '''

    w = tf.get_variable('weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01))

    #EG.2 calc u_hat
    # do tilling for input and w before matmul
    # input -> [128, 1152, 10, 8, 1]
    # w -> [128, 1152, 10, 8, 16]
    input = tf.tile(input_x, [1, 1, 10, 1, 1])
    w = tf.tile(w, [128, 1, 1, 1, 1])
    assert input.get_shape() == [128, 1152, 10, 8, 1]

    u_hat = tf.matmul(w, input, transpose_a=True)
    assert u_hat.get_shape() == [128, 1152, 10, 16, 1]


    # line 3 for r iteration do
    for r_iter in range(3):
        with tf.variable_scope('iter_' + str(r_iter)):
            #Line 4:
            # -> [1, 1152, 10, 1, 1]
            c_ij = tf.nn.softmax(b_ij, dim=2)

            #Line 5:
            # weighting u_hat with c_ij, element - wise in the last two dims
            # -> [128, 1152, 10, 16, 1]
            s_j = tf.multiply(c_ij, u_hat)
            s_j = tf.reduce_sum(s_j, axis = 1, keep_dims=True)
            assert s_j.get_shape() == [128, 1, 10, 16, 1]

            v_j = squash(s_j)
            assert v_j.get_shape() == [128, 1, 10, 16, 1]

            v_j_tiled = tf.tile(v_j, [1,1152, 1, 1, 1])
            u_produce_v = tf.matmul(u_hat, v_j_tiled, transpose_a=True)
            assert u_produce_v.get_shape() == [128, 1152, 10, 1, 1]

            if r_iter < 3 - 1:
                b_ij += u_produce_v


    return (v_j)

def squash(vector):
    '''
    squashing function corresponding to Eq. 1
    :param vector:
        vector: a tensor with shape [128, 1, num_caps, vec_len, 1] or [128, num_caps, vec_len, 1]
    :return:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension
    '''
    vec_squared_norm = tf.reduce_mean(tf.square(vector), -2, keep_dims=True)
    scaler_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-9)
    vec_squared = scaler_factor * vector
    return (vec_squared)

def fully_connected(input, num_output, vec_len, with_routing=True,
                    weights_initializers = tf.contrib.layers.xavier_initializer(),
                    reuse = None,
                    variable_collections = None,
                    scope = None):
    '''
    :param input: A tensor of at least rank 3, i.e [128, num_inputs, vec_len]
    :param num_output:
    :param vec_len:
    :param with_routing:
    :param weights_initializers:
    :param reuse:
    :param variable_collections:
    :param scope:
    :return:
    '''
    layer = CapsLayer(num_outputs=num_output, length=vec_len, with_routing=with_routing, layer_type='FC')
    return layer.apply(input)

def conv2d(input,
           filter,
           vec_len,
           kernel_size,
           stride,
           with_routing = False,
           reuse=None):

    layer = CapsLayer(num_outputs=filter,
                      length=vec_len,
                      with_routing=with_routing,
                      layer_type='CONV')
    return (layer(input, filter = kernel_size, stride = stride))

