## The current state of the implementations

Generated using `state.txt` from [tree.nathanfriend.io](tree.nathanfriend.io)

```
src
├── utils
│   ├── basic
│   │   ├── freeze_params
│   │   └── subsequent_mask
│   ├── vocab
│   │   ├── Vocabulary
│   │   ├── TextVocabulary
│   │   ├── GlossVocabulary
│   │   ├── filter_min
│   │   ├── sort_and_cut
│   │   └── build_vocab
│   └── initialization
│       ├── orthogonal_rnn_init_
│       ├── lstm_forget_gate_init_
│       ├── xavier_uniform_n_
│       └── initialize_model
└── models
    └── components
        ├── attention
        │   ├── MultiHeadedAttention
        │   ├── PositionwiseFeedForward
        │   ├── PositionalEncoding
        │   ├── AttentionMechanism
        │   ├── BahdanauAttention
        │   └── LuongAttention
        ├── encoders
        │   ├── Encoder
        │   ├── TransformerEncoderLayer
        │   ├── RecurrentEncoder
        │   └── TransformerEncoder
        ├── decoders
        │   ├── Decoder
        │   ├── TransformerDecoderLayer
        │   ├── RecurrentDecoder
        │   └── TransformerDecoder
        └── signtransformer
```


To-do:
```python
# Probably this must be placed in src.utils
from signjoey.search import beam_search, greedy
from signjoey.batch import Batch

# This must be placed in src.models.components
from signjoey.embeddings import Embeddings, SpatialEmbeddings
import signjoey.model
```