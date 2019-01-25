from layers.convolution import cnn_layers
from layers.losses import mse
from layers.similarity import manhattan_similarity
from models.base_model import BaseSiameseNet
from utils.config_helpers import parse_list
from layers.basics import dropout


class AttentionCnn(BaseSiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        BaseSiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)
        self._conv_filter_size = parse_list(model_cfg['PARAMS']['filter_sizes'])
        self._conv_projection_size = 300
        self._attention_output_size = 200
        self._comparison_output_size = 100
        
    def _masked_softmax(self, values, lengths):
        with tf.name_scope('MaskedSoftmax'):
            mask = tf.expand_dims(tf.sequence_mask(lengths, tf.reduce_max(lengths), dtype=tf.float32), -2)
    
            inf_mask = (1 - mask) * -np.inf
            inf_mask = tf.where(tf.is_nan(inf_mask), tf.zeros_like(inf_mask), inf_mask)

            return tf.nn.softmax(tf.multiply(values, mask) + inf_mask)
        
    def _conv_pad(self, values):
        with tf.name_scope('convolutional_padding'):
            pad = tf.zeros([tf.shape(self.x1)[0], 1, self.embedding_size[1]])
            return tf.concat([pad, values, pad], axis=1)
        

    def _attention_layer(self):
        with tf.name_scope('attention_layer'):
            e_X1 = tf.layers.dense(self._X1_conv, self._attention_output_size, activation=tf.nn.relu, name='attention_nn')
            e_X2=tf.layers.dense(self._X2_conv,self._attention_output_size,activation=tf.nn.relu,
                                 name='attention_nn',reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._beta = tf.matmul(self._masked_softmax(e, sequence_len), self._X2_conv, name='beta2')
            self._alpha = tf.matmul(self._masked_softmax(tf.transpose(e, [0,2,1]), sequence_len), self._X1_conv, name='alpha2')
            
    def siamese_layer():
        with tf.name_scope('convolutional_layer'):
            X1_conv_1 = tf.layers.conv1d(
                #self._conv_pad(self.conc_X1),
                self._conv_pad(self.embedded_x1),
                self._conv_projection_size,
                self._conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_1',
            )
            
            X2_conv_1 = tf.layers.conv1d(
                #self._conv_pad(self.conc_X2),
                self._conv_pad(self.embedded_x2),
                self._conv_projection_size,
                self._conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_1',
                reuse=True
            )
            
            X1_conv_1 = tf.layers.dropout(X1_conv_1, rate=self.dropout, training=self.is_training)
            X2_conv_1 = tf.layers.dropout(X2_conv_1, rate=self.dropout, training=self.is_training)
            
            X1_conv_2 = tf.layers.conv1d(
                self._conv_pad(X1_conv_1),
                self._conv_projection_size,
                self._conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_2',
            )
            
            X2_conv_2 = tf.layers.conv1d(
                self._conv_pad(X2_conv_1),
                self._conv_projection_size,
                self._conv_filter_size,
                padding='valid',
                use_bias=False,
                name='conv_2',
                reuse=True
            )
            
            self._X1_conv = tf.layers.dropout(X1_conv_2, rate=self.dropout, training=self.is_training)
            self._X2_conv = tf.layers.dropout(X2_conv_2, rate=self.dropout, training=self.is_training)
            
        with tf.name_scope('attention_layer'):
            e_X1 = tf.layers.dense(self._X1_conv, self._attention_output_size, activation=tf.nn.relu, name='attention_nn')
            
            e_X2 = tf.layers.dense(self._X2_conv, self._attention_output_size, activation=tf.nn.relu, name='attention_nn', reuse=True)
            
            e = tf.matmul(e_X1, e_X2, transpose_b=True, name='e')
            
            self._beta = tf.matmul(self._masked_softmax(e, self.X2_len), self._X2_conv, name='beta2')
            self._alpha = tf.matmul(self._masked_softmax(tf.transpose(e, [0,2,1]), self.X1_len), self._X1_conv, name='alpha2')
            
        with tf.name_scope('comparison_layer'):
            X1_comp = tf.layers.dense(
                tf.concat([self._X1_conv, self._beta], 2),
                self._comparison_output_size,
                activation=tf.nn.relu,
                name='comparison_nn'
            )
            self._X1_comp = tf.multiply(
                tf.layers.dropout(X1_comp, rate=self.dropout, training=self.is_training),
                tf.expand_dims(tf.sequence_mask(self.sequence_length, tf.reduce_max(self.sequence_length), dtype=tf.float32), -1)
            )
            
            X2_comp = tf.layers.dense(
                tf.concat([self._X2_conv, self._alpha], 2),
                self._comparison_output_size,
                activation=tf.nn.relu,
                name='comparison_nn',
                reuse=True
            )
            self._X2_comp = tf.multiply(
                tf.layers.dropout(X2_comp, rate=self.dropout, training=self.is_training),
                tf.expand_dims(tf.sequence_mask(self.sequence_length, tf.reduce_max(self.sequence_length), dtype=tf.float32), -1)
            )
        
            X1_agg = tf.reduce_sum(self._X1_comp, 1)
            X2_agg = tf.reduce_sum(self._X2_comp, 1)
            
        
        return manhattan_similarity(X1_agg, X2_agg)