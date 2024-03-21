from typing import Optional, Tuple
from torch import Tensor
from pruning.AttenPruner import AbstractAttenPruner
from polp.nn.layers.attention import AttentionMask
from polp.nn.layers.cache import KeyValueCache
from polp.nn.layers.transformer import _TransformerLayer
from pruning.utils import repack_tensor_and_create_mask
from pruning.metric_logger import MetricLogger

metric_logger = MetricLogger()


class PrunableEncoderLayer(_TransformerLayer):
    """An Encoder layer of the Transformer model where the pruning occurs. Specifically, the pruning occurs
    after the MHA layer and before the FFN layer. It is done by setting the output of the MHA to 0 using different
    pruning strategies.
    """

    def __init__(
            self,
            pruner: Optional[AbstractAttenPruner] = None,
            *args, 
            **kwargs):
        super(PrunableEncoderLayer, self).__init__(*args, **kwargs)
        self.pruner = pruner

    def forward(
        self,
        input: Tensor,
        attention_mask: AttentionMask,
        *,
        cache: Optional[KeyValueCache] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
        use_causal_mask: bool = False,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """Apply the transformer layer to the given piece hidden representations.

        Parameters
        ----------
        input : Tensor
            Hidden representations to apply the layer to.
        attention_mask : AttentionMask
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.
        use_causal_mask : bool
            Mask out succeeding sequence elements when ``True``.
        cache : Optional[KeyValueCache], optional
            Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen., by default None
        positions : Optional[Tensor], optional
           Input positions. Positions are needed to look up rotary embeddings, by default None
        store_cache : bool, optional
            Whether to cache the key/value representations for future reuse, by default False

        Returns
        -------
        Tuple[Tensor, Optional[KeyValueCache]]
            Layer output and the key/value cache.
        """
        prunable_layers = metric_logger.get('prunable_layers')
        residual = input

        attn_out, atten_probes, _ = self.mha(
            self.attn_input_layer_norm(input),
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
            use_causal_mask=use_causal_mask,
        )
        attn_out = self.attn_output_dropout(attn_out)

        
        
        residual = self.attn_residual_layer_norm(input + attn_out)

        # Remove tokens using the mask we got from pruning
        if self.pruner.idx in prunable_layers:
            mask_from_pruning = self.pruner(atten_probes, attention_mask.bool_mask)
            # In earlier versions, were performing element-wise mult. Now we completely remove the tokens.
            residual, updated_mask = repack_tensor_and_create_mask(residual, mask_from_pruning, True)
            updated_mask = AttentionMask(updated_mask)
        
        ffn_in = residual

        ffn_out = self.ffn(self.ffn_input_layer_norm(ffn_in))
        ffn_out = self.ffn_output_dropout(ffn_out)

        output = ffn_out

        layer_output = self.ffn_residual_layer_norm(residual + output)

        # We keep the same attention mask if this layer is not performing prunning
        if self.pruner.idx not in prunable_layers:
            updated_mask = attention_mask


        return layer_output, atten_probes, updated_mask, cache
    
