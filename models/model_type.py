from enum import Enum
from models.cnn import CnnSiameseNet
from models.lstm import LSTMBasedSiameseNet
from models.multihead_attention import MultiheadAttentionSiameseNet
from models.bcann import AttentionCnn
from models.bcsann import AttentionSCnn
from models.bcsann1 import AttentionS2Cnn
from models.twolayerbcnn import Attention2lyrCnn
from models.bcmultihead import AttentionMCnn
from models.bcsannlwr import AttentionDbCnn

class ModelType(Enum):
    multihead = 0,
    rnn = 1,
    cnn = 2,
    bcann = 3,
    bcsann = 4,
    bcsann1 = 5,
    twolayerbcnn=6,
    bcmultihead = 7,
    bcsannlwr = 8


MODELS = {
    ModelType.cnn.name: CnnSiameseNet,
    ModelType.rnn.name: LSTMBasedSiameseNet,
    ModelType.multihead.name: MultiheadAttentionSiameseNet,
    ModelType.bcann.name: AttentionCnn,
    ModelType.bcsann.name:AttentionSCnn,
    ModelType.bcsann1.name:AttentionS2Cnn,
    ModelType.twolayerbcnn.name:Attention2lyrCnn,
    ModelType.bcmultihead.name:AttentionMCnn,
    ModelType.bcsannlwr.name:AttentionDbCnn
}

