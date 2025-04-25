# ndt_moe_config.py
from transformers import PretrainedConfig

class NDTMoEConfig(PretrainedConfig):
    model_type = "ndt_moe" # Register the model type

    def __init__(
        self,
        # Standard BERT/Transformer parameters
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072, # Intermediate size for non-expert FFNs if any
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        # NDT-MoE specific parameters
        num_experts=8,                 # Number of experts per layer
        num_experts_per_tok=2,         # K for Top-K routing
        ndt_depth=6,                   # Depth of the router NDT
        ndt_entmax_alpha=1.5,          # Alpha for entmax in NDT router
        expert_type="ffn",             # Type of expert ('ffn' or potentially 'transformer')
        expert_ffn_intermediate_dim=None, # Intermediate dim for FFN experts (defaults to intermediate_size)
        load_balancing_loss_coef=0.01, # Coefficient for load balancing loss (delta)
        use_densenet_connections=True, # Use DenseNet style inputs to layers
        pooler_type="first_token_transform", # How to pool sequence output ('first_token_transform', 'mean', None)
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.ndt_depth = ndt_depth
        self.ndt_entmax_alpha = ndt_entmax_alpha
        self.expert_type = expert_type
        # Default expert FFN dim if not provided
        self.expert_ffn_intermediate_dim = expert_ffn_intermediate_dim or intermediate_size
        self.load_balancing_loss_coef = load_balancing_loss_coef
        self.use_densenet_connections = use_densenet_connections
        self.pooler_type = pooler_type
