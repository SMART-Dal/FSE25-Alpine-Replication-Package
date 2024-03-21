from typing import Optional

import torch
from pruning.PrunableEncoderLayer import PrunableEncoderLayer
from polp.nn.layers.attention import AttentionHeads, QkvMode, ScaledDotProductAttention, SelfAttention
from polp.nn.layers.feedforward import PointwiseFeedForward
from polp.nn.layers.transformer import TransformerDropouts, TransformerLayerNorms
from polp.nn.models.roberta.config import RoBERTaConfig
from polp.nn.models.roberta.encoder import RoBERTaEncoder
from torch import Tensor
from polp.nn.layers.attention import AttentionMask
from polp.nn.models.output import ModelOutput
from pruning.AttenPruner import IQRPruner
from pruning.metric_logger import MetricLogger
import logging


metric_logger = MetricLogger()
logger = logging.getLogger(__name__)

class PrunableModel(RoBERTaEncoder):
    """
    An RoBERTa-based model that supports attention-based pruning. Identical to the building blocks of the
    RoBERTa model, it only overrides the encoding layers with ones that allow for pruning.
    """
    
    def __init__(
            self,
            config: RoBERTaConfig,
            *,
            device: Optional[torch.device] = None):
        
        super().__init__(config=config, device=device)
        alpha = float(metric_logger.get('alpha'))
        logger.info(f"PrunableModel initiated with alpha={alpha}")
        self.layers = torch.nn.ModuleList(
            [
                PrunableEncoderLayer(
                    pruner=IQRPruner(idx=_, alpha=alpha),
                    attention_layer=SelfAttention(
                        attention_heads=AttentionHeads.uniform(
                            config.layer.attention.n_query_heads
                        ),
                    attention_scorer=ScaledDotProductAttention(
                        dropout_prob=config.layer.attention.dropout_prob,
                        linear_biases=None,
                        output_attention_probs=config.layer.attention.output_attention_probs,
                    ),
                    hidden_width=self.hidden_width,
                    qkv_mode=QkvMode.SEPARATE,
                    rotary_embeds=None,
                    use_bias=config.layer.attention.use_bias,
                    device=device,
                    ),
                    feed_forward_layer=PointwiseFeedForward(
                        activation=config.layer.feedforward.activation.module(),
                        hidden_width=self.hidden_width,
                        intermediate_width=config.layer.feedforward.intermediate_width,
                        use_bias=config.layer.feedforward.use_bias,
                        use_gate=config.layer.feedforward.use_gate,
                        device=device,
                    ),
                    dropouts=TransformerDropouts.layer_output_dropouts(
                        config.layer.dropout_prob
                    ),
                    layer_norms=TransformerLayerNorms(
                        attn_residual_layer_norm=self.layer_norm(),
                        ffn_residual_layer_norm=self.layer_norm(),
                    ),
                    use_parallel_attention=config.layer.attention.use_parallel_attention,
                )
                for _ in range(config.layer.n_hidden_layers)
            ]
        )

    def forward(
        self,
        piece_ids: Tensor,
        attention_mask: AttentionMask,
        *,
        positions: Optional[Tensor] = None,
        type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        embeddings = self.embeddings(piece_ids, positions=positions, type_ids=type_ids)
        layer_output = embeddings
        layer_outputs = []
        attention_probes = []
        for layer in self.layers:
            if self.config.layer.attention.output_attention_probs:
                layer_output, layer_attention_probes, updated_mask, _ = layer(layer_output, attention_mask)
                attention_mask = updated_mask
                attention_probes.append(layer_attention_probes)
            else:
                layer_output, _ = layer(layer_output, attention_mask)
            layer_outputs.append(layer_output)

        return ModelOutput(all_outputs=[embeddings, *layer_outputs], attention_probs=attention_probes)    
