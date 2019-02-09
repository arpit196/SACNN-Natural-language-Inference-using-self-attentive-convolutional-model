import tensorflow as tf

from layers.losses import mse
from layers.recurrent import rnn_layer
from layers.similarity import manhattan_similarity
from models.base_model import BaseSiameseNet


class LSTMBasedSiameseNet(BaseSiameseNet):

    def __init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg):
        BaseSiameseNet.__init__(self, max_sequence_len, vocabulary_size, main_cfg, model_cfg, mse)

    def siamese_layer(self, sequence_len, model_cfg):
        hidden_size = model_cfg['PARAMS'].getint('hidden_size')
        cell_type = model_cfg['PARAMS'].get('cell_type')
        #BERT EMBEDDING
        
        elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
        embeddings = elmo(self.x1, signature="default", as_dict=True)["elmo"]
        
        outputs_sen1 = rnn_layer(self.embedded_x1, hidden_size, cell_type)
        outputs_sen2 = rnn_layer(self.embedded_x2, hidden_size, cell_type, reuse=True

    return manhattan_similarity(out1, out2)
