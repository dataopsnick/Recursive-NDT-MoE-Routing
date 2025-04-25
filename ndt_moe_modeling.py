# ndt_moe_modeling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical # Potentially needed for Gumbel/Concrete
import math
from typing import Optional, Tuple, List, Dict, Any

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
# Import the config
from .ndt_moe_config import NDTMoEConfig

# === Helper Functions (Placeholder for Entmax) ===
# You might need to install a library like sparsemedoids or implement entmax:
# pip install sparsemedoids # (Check if this library is suitable)
# Or implement based on https://arxiv.org/abs/1905.05702
try:
    from sparsemedoids import entmax_alpha # Example import
except ImportError:
    print("WARNING: sparsemedoids not found. Using placeholder softmax for entmax.")
    def entmax_alpha(inputs, alpha=1.5, dim=-1):
        if alpha == 1.0:
            return F.softmax(inputs, dim=dim)
        elif alpha == 2.0:
             # Example placeholder for sparsemax, needs actual implementation
            return F.softmax(inputs, dim=dim) # Placeholder!
        else:
             # Example placeholder for entmax, needs actual implementation
            return F.softmax(inputs, dim=dim) # Placeholder!


# === NDT Router ===
class NDTRouter(nn.Module):
    """ Neural Oblivious Decision Tree Router using Entmax for sparsity """
    def __init__(self, config: NDTMoEConfig):
        super().__init__()
        self.depth = config.ndt_depth
        self.num_leaves = 2**config.ndt_depth
        self.num_experts = config.num_experts
        self.input_dim = config.hidden_size
        self.alpha = config.ndt_entmax_alpha
        self.k = config.num_experts_per_tok

        # Learnable parameters per tree level
        self.feature_selectors = nn.Parameter(torch.Tensor(self.depth, self.input_dim))
        self.thresholds = nn.Parameter(torch.Tensor(self.depth, 1)) # Learnable thresholds b_i
        self.log_temperatures = nn.Parameter(torch.Tensor(self.depth, 1)) # Learnable log scales log(tau_i)

        # Final layer mapping leaves to expert logits
        self.leaf_to_expert_logits = nn.Linear(self.num_leaves, self.num_experts)

        self._initialize_parameters(config.initializer_range)

    def _initialize_parameters(self, initializer_range):
        nn.init.normal_(self.feature_selectors, mean=0.0, std=initializer_range)
        nn.init.normal_(self.thresholds, mean=0.0, std=initializer_range)
        # Initialize temperatures to small positive values (e.g., log(1.0))
        nn.init.constant_(self.log_temperatures, 0.0)
        # Initialize linear layer
        nn.init.normal_(self.leaf_to_expert_logits.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(self.leaf_to_expert_logits.bias)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): Shape (batch_size, seq_len, hidden_size)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - expert_logits: Raw logits for load balancing (batch_size, seq_len, num_experts)
                - selected_experts: Indices of top-k experts (batch_size, seq_len, k)
                - routing_weights: Softmax weights for top-k experts (batch_size, seq_len, k)
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Calculate feature combinations f_i(x) for each level
        # Shape: (depth, batch_size, seq_len)
        feature_combinations = torch.einsum(
            'blh,dh->dbl', hidden_states, entmax_alpha(self.feature_selectors, alpha=self.alpha, dim=-1)
        )

        # Calculate split probabilities c_i(x)
        temperatures = torch.exp(self.log_temperatures) # Ensure tau > 0
        scaled_diffs = (feature_combinations - self.thresholds) / temperatures
        # Input to entmax: shape (depth, batch_size, seq_len, 2) - last dim is [scaled_diff, 0]
        entmax_input = torch.stack([scaled_diffs, torch.zeros_like(scaled_diffs)], dim=-1)
        # Split probs: shape (depth, batch_size, seq_len, 2) - last dim is [P(split_left), P(split_right)=0]
        split_probs_raw = entmax_alpha(entmax_input, alpha=self.alpha, dim=-1)
        # Shape: (depth, batch_size, seq_len) - P(go right, i.e., feature > threshold)
        right_probs = split_probs_raw[..., 0].unsqueeze(-1)
        left_probs = 1.0 - right_probs
        # Shape: (depth, batch_size, seq_len, 2) - [P(left), P(right)]
        split_probs = torch.cat([left_probs, right_probs], dim=-1)

        # Calculate leaf probabilities C(x) using outer products across depth
        # Shape: (batch_size, seq_len, 1)
        leaf_probs = hidden_states.new_ones(batch_size, seq_len, 1)
        for i in range(self.depth):
            # Reshape leaf_probs to (batch, seq, current_leaves, 1)
            # Reshape split_probs to (batch, seq, 1, 2)
            # Outer product: (batch, seq, current_leaves, 2) -> flatten last 2 dims
            num_current_leaves = 2**i
            leaf_probs = leaf_probs.view(batch_size, seq_len, num_current_leaves, 1)
            current_split_probs = split_probs[i].view(batch_size, seq_len, 1, 2)
            leaf_probs = leaf_probs * current_split_probs # Broadcasting
            leaf_probs = leaf_probs.view(batch_size, seq_len, num_current_leaves * 2) # Flatten

        # Leaf probs C(x): shape (batch_size, seq_len, num_leaves)
        assert leaf_probs.shape[-1] == self.num_leaves

        # Map leaf probabilities to expert logits
        # Shape: (batch_size, seq_len, num_experts)
        expert_logits = self.leaf_to_expert_logits(leaf_probs)

        # Get Top-K routing weights and indices
        routing_weights_sparse, selected_experts = torch.topk(expert_logits, self.k, dim=-1)
        # Softmax over the top-k logits only for weighting the outputs
        routing_weights = F.softmax(routing_weights_sparse, dim=-1)

        return expert_logits, selected_experts, routing_weights


# === FFN Expert ===
class FeedForwardExpert(nn.Module):
    def __init__(self, config: NDTMoEConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.expert_ffn_intermediate_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dense2 = nn.Linear(config.expert_ffn_intermediate_dim, config.hidden_size)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        x = self.dense1(hidden_states)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x

# === NDT-MoE Layer ===
class NDTMoELayer(nn.Module):
    def __init__(self, config: NDTMoEConfig):
        super().__init__()
        self.router = NDTRouter(config)
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([FeedForwardExpert(config) for _ in range(config.num_experts)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states (torch.Tensor): Shape (batch_size, seq_len, hidden_size)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - layer_output: Output tensor (batch_size, seq_len, hidden_size)
                - expert_logits: Router logits for load balancing (batch_size, seq_len, num_experts)
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        residual = hidden_states

        # Get routing decisions
        expert_logits, selected_experts, routing_weights = self.router(hidden_states)
        # expert_logits: (b, s, num_experts)
        # selected_experts: (b, s, k) - indices
        # routing_weights: (b, s, k) - softmax weights for top k

        # Flatten tokens for easier dispatch
        flat_hidden_states = hidden_states.view(-1, hidden_dim) # (b*s, h)
        flat_selected_experts = selected_experts.view(-1, self.num_experts_per_tok) # (b*s, k)
        flat_routing_weights = routing_weights.view(-1, self.num_experts_per_tok) # (b*s, k)
        num_tokens = flat_hidden_states.shape[0]

        # Initialize final output tensor
        final_output = torch.zeros_like(flat_hidden_states)

        # ---- Basic (Inefficient) Dispatch Loop ----
        # TODO: Replace with optimized sparse operation
        expert_outputs_list = [[] for _ in range(self.num_experts)]
        token_indices_list = [[] for _ in range(self.num_experts)]
        weights_list = [[] for _ in range(self.num_experts)]

        # Assign tokens to experts based on Top-K
        for k_idx in range(self.num_experts_per_tok):
            expert_indices_k = flat_selected_experts[:, k_idx] # (b*s,)
            weights_k = flat_routing_weights[:, k_idx]      # (b*s,)
            for token_idx in range(num_tokens):
                expert_id = expert_indices_k[token_idx].item()
                expert_outputs_list[expert_id].append(flat_hidden_states[token_idx])
                token_indices_list[expert_id].append(token_idx)
                weights_list[expert_id].append(weights_k[token_idx])

        # Process inputs for each expert
        for expert_id in range(self.num_experts):
            if expert_outputs_list[expert_id]:
                expert_input_batch = torch.stack(expert_outputs_list[expert_id], dim=0)
                expert_output_batch = self.experts[expert_id](expert_input_batch)
                original_indices = token_indices_list[expert_id]
                expert_weights = torch.stack(weights_list[expert_id], dim=0).unsqueeze(-1) # (num_tok_for_expert, 1)

                # Add weighted output to the final output tensor at original positions
                final_output.index_add_(0, torch.tensor(original_indices, device=final_output.device), expert_output_batch * expert_weights)
        # ---- End of Basic Dispatch Loop ----

        # Reshape final output and apply residual + norm
        layer_output = final_output.view(batch_size, sequence_length, hidden_dim)
        layer_output = self.layer_norm(residual + layer_output)

        return layer_output, expert_logits


# === Main NDT-MoE Model ===
class NDTMoEModel(PreTrainedModel):
    config_class = NDTMoEConfig

    def __init__(self, config: NDTMoEConfig):
        super().__init__(config)
        self.config = config
        # Use standard BertEmbeddings or implement your own
        from transformers.models.bert.modeling_bert import BertEmbeddings
        self.embeddings = BertEmbeddings(config) # Assumes BERT-compatible vocab/config

        self.encoder_layers = nn.ModuleList([NDTMoELayer(config) for _ in range(config.num_hidden_layers)])

        if self.config.pooler_type == "first_token_transform":
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        # For DenseNet connections
        self.layer_input_dims = [config.hidden_size] * config.num_hidden_layers
        if config.use_densenet_connections:
            current_dim = config.hidden_size
            for i in range(config.num_hidden_layers):
                 # Layer i receives concat(h_0, ..., h_{i-1})
                 self.layer_input_dims[i] = current_dim
                 # Output dim of layer i is always hidden_size
                 current_dim += config.hidden_size
                 # TODO: Need projection layers before NDTMoELayer if input dims change
                 # This part requires careful implementation - simplifying for now

        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # Initialize NDT Router parameters within NDTRouter._initialize_parameters

    def forward(
        self,
        input_ids=None,
        attention_mask=None, # Required for embeddings padding, might be unused otherwise
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # ... (Input validation similar to BertModel) ...

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = embedding_output
        all_hidden_states = (embedding_output,) if output_hidden_states else None
        all_expert_logits = () # Collect logits for load balancing loss

        layer_input = hidden_states
        collected_outputs = [hidden_states] # For DenseNet

        for i, layer_module in enumerate(self.encoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # TODO: Implement DenseNet connection logic here if self.config.use_densenet_connections
            # layer_input = torch.cat(collected_outputs, dim=-1)
            # May need projection if input dim changes: layer_input = projection(layer_input)

            layer_output, expert_logits = layer_module(layer_input)
            hidden_states = layer_output
            all_expert_logits = all_expert_logits + (expert_logits,)
            collected_outputs.append(hidden_states)

            layer_input = hidden_states # For next iteration if not DenseNet

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Pooling
        pooler_output = None
        if self.config.pooler_type == "first_token_transform" and self.pooler is not None:
            first_token_tensor = hidden_states[:, 0]
            pooler_output = self.pooler(first_token_tensor)
            pooler_output = self.pooler_activation(pooler_output)
        elif self.config.pooler_type == "mean":
             # Mask pooling (requires attention_mask)
             if attention_mask is None:
                 attention_mask = torch.ones_like(input_ids) # Assuming all non-pad tokens if no mask
             masked_sum = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
             num_non_padding = torch.sum(attention_mask, dim=1, keepdim=True)
             pooler_output = masked_sum / num_non_padding.clamp(min=1e-9) # Avoid division by zero
        # else: pooler_output is None

        if not return_dict:
            # Need to decide what base model output is. Include all_expert_logits?
            # Sticking to standard BaseModelOutputWithPooling fields for now.
            # The trainer can access all_expert_logits via the classifier model's output.
            return tuple(v for v in [hidden_states, pooler_output, all_hidden_states] if v is not None)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            hidden_states=all_hidden_states,
            attentions=None, # NDT-MoE doesn't have attentions in the traditional sense
            # Add expert logits here if needed for base model output
        )


# === Custom Output Class ===
@dataclass
class NDTMoESequenceClassifierOutput(SequenceClassifierOutput):
    """ Extends SequenceClassifierOutput to include expert logits for loss calculation. """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None # Will be None
    all_expert_logits: Optional[Tuple[torch.FloatTensor]] = None # ADDED


# === NDT-MoE for Sequence Classification ===
class NDTMoEForSequenceClassification(PreTrainedModel):
    config_class = NDTMoEConfig

    def __init__(self, config: NDTMoEConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.ndt_moe = NDTMoEModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None, # Labels are used by the Trainer, not directly here for loss
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ndt_moe(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True, # Force return_dict for easy access
        )

        pooler_output = outputs.pooler_output
        if pooler_output is None:
             # If no pooler, use first token of last hidden state
             pooler_output = outputs.last_hidden_state[:, 0]
             # Warning: This might not be optimal if no pooling was intended

        pooled_output = self.dropout(pooler_output)
        logits = self.classifier(pooled_output)

        # Loss is NOT calculated here by default.
        # The DistillationTrainer will handle loss calculation
        # using teacher logits, student logits (these ones), labels, and all_expert_logits.
        loss = None

        # Return custom output object including expert_logits
        return NDTMoESequenceClassifierOutput(
            loss=loss, # Loss will be calculated by the Trainer
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
            all_expert_logits=outputs.all_expert_logits, # Pass this through
        )
