import tensorflow as tf
import numpy as np
from config import cfg
from utils import get_batch_data
from capsLayer import CapsLayer


deviceType = "/gpu:0"

class CapsNet(object):
    def __init__(self,is_train = True):
        #tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_train:
                self.x, self.labels = get_batch_data()
                self.y = tf.one_hot(self.labels, depth=10, axis=1, dtype = tf.float32)

                self.model_builder()
                self.loss()
                self._summary()

                self.tmp = tf.Variable(0, name='tmp', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.final_loss, global_step=self.tmp)
            elif cfg.mask_with_y:
                self.x = tf.placeholder(tf.float32,
                                        shape=(cfg.batch_size, 28, 28, 1))
                self.y = tf.placeholder(tf.float32, shape=(cfg.batch_size, 10, 1))
                self.model_builder()
            else:
                self.x = tf.placeholder(tf.float32,
                                        shape=(cfg.batch_size, 28, 28, 1))
                self.model_builder()

    def model_builder(self):#build the 3-layer capsNet
            #First Conv Layer
            wConv = tf.get_variable("wConv", shape=[9, 9, 1, 256])
            bConv = tf.get_variable("bConv", shape=[256])
            conv1 = tf.nn.conv2d(self.x, wConv, strides=[1, 1, 1, 1],
                         padding='VALID') + bConv #out_put_size = 20 * 20 * 256

            #Primary Capsules layer
            #kernel = tf.get_variable(name = 'kernel_size', shape = [9,9,256,256])
            primary_capslayer = CapsLayer(num_outputs=32, length=8, with_routing=False, layer_type='CONV')
            caps_layer1 = primary_capslayer(conv1, filter=9, stride=2) #out_put_size = 6*6*8*32

            #Digits Capsules layer
            digit_capslayer = CapsLayer(num_outputs=10, length=16, with_routing=True, layer_type='FC')
            self.caps_layer2 = digit_capslayer(caps_layer1)

            #Decode the digit Capsules to pic again
            # Change the digitsCaps from [10, 16, 1] => [10, 1, 1]

            #cal the ||vc||
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps_layer2),
                                                  axis=2, keep_dims=True) + 1e-9)
            #softmax for the result
            self.softmax_res = tf.nn.softmax(self.v_length, dim=1)

            #Get the index of max softmax val of the 10 caps
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_res, axis=1))
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size,))

            if not cfg.mask_with_y:
                masked_res = []
                for i in range(cfg.batch_size):
                    v = self.caps_layer2[i][self.argmax_idx[i], :]
                    masked_res.append(tf.reshape(v, shape=[1, 1, 16, 1]))
                self.masked_res = np.vstack(masked_res)
            else:
                self.masked_res = tf.multiply(tf.squeeze(self.caps_layer2), tf.reshape(self.y, (-1, 10, 1)))
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps_layer2), axis=2, keep_dims=True) + 1e-9)

            #reconstruct the img from res using 3 fully connected Layer
                tmpv = tf.reshape(self.masked_res, shape=(cfg.batch_size, -1))
                #fully connected layer1
                fw1 = tf.get_variable("fw1", shape=[160, 512])  # so we have 256*64 parameter to calculate
                fb1 = tf.get_variable("fb1", shape=[512])
                fc_1 = tf.matmul(tmpv,fw1) + fb1
                # fully connected layer2
                fw2 = tf.get_variable("fw2", shape=[512, 1024])  # so we have 256*64 parameter to calculate
                fb2 = tf.get_variable("fb2", shape=[1024])
                fc_2 = tf.matmul(fc_1,fw2) + fb2
                # fully connected layer3
                self.decoded = tf.contrib.layers.fully_connected(fc_2, num_outputs=784, activation_fn=tf.sigmoid)

    def loss(self):
            # margin loss
            # part1 = max(0, m_plus-||v_c||)^2
            mloss1 = tf.maximum(0., cfg.m_plus - self.v_length) * tf.maximum(0., cfg.m_plus - self.v_length)
            # part2 = max(0,m_minus-||v_c||)^2
            mloss2 = tf.maximum(0., self.v_length - cfg.m_minus) * tf.maximum(0., self.v_length - cfg.m_minus)

            # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
            mloss1 = tf.reshape(mloss1, shape=(cfg.batch_size, -1))
            mloss2 = tf.reshape(mloss2, shape=(cfg.batch_size, -1))

            T_c = self.y #T_c = 1 if the class c is present
            Lc = T_c * mloss1 + cfg.lambda_val * (1 - T_c) * mloss2
            self.margin_loss = tf.reduce_mean(tf.reduce_sum(Lc, axis=1))

            #reconstruction loss
            original = tf.reshape(self.x, shape=(cfg.batch_size, -1))
            square_loss = tf.square(self.decoded - original)
            self.reconstruction_loss = tf.reduce_mean(square_loss)

            #final_loss
            self.final_loss =  self.margin_loss + cfg.regularization_scale * self.reconstruction_loss

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_loss))
        train_summary.append(tf.summary.scalar('train/total_loss', self.final_loss))
        reconstruction_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        train_summary.append(tf.summary.image('reconstruction_img', reconstruction_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.batch_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.test_acc = tf.placeholder_with_default(tf.constant(0.), shape=[])