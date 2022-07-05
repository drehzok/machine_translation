## The current state of the implementations

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