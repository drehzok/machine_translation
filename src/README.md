## The current state of the implementations

Generated using `state.txt` from [tree.nathanfriend.io](tree.nathanfriend.io)

```
src
├── utils
│   └── basic
│       ├── freeze_params
│       └── subsequent_mask
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
        └── decoders
            ├── Decoder
            ├── TransformerDecoderLayer
            ├── RecurrentDecoder
            └── TransformerDecoder
```


To-do:
```python
# Probably this must be placed in src.utils
from signjoey.initialization import initialize_model
from signjoey.search import beam_search, greedy
from signjoey.batch import Batch

# This must be placed in src.models.components
from signjoey.embeddings import Embeddings, SpatialEmbeddings
import signjoey.model
```