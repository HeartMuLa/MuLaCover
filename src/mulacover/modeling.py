from typing import Optional

import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.modules import RMSNorm
from transformers.modeling_utils import PreTrainedModel

from .configuration import MuLaCoverConfig


def llama3_2_3B() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=8192,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


def llama3_2_300M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=3,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )


FLAVORS = {
    "llama-3B": llama3_2_3B,
    "llama-300M": llama3_2_300M,
}


def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    model.tok_embeddings = nn.Identity()
    model.output = nn.Identity()
    return model, embed_dim


def _create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def _index_causal_mask(mask: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
    return mask[input_pos, :]


def _build_attention_mask(
    key_padding_mask: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    return (~key_padding_mask).unsqueeze(1).expand(-1, query.size(1), -1)


def _mask_condition(value: torch.Tensor, uncond_mask: torch.Tensor) -> torch.Tensor:
    mask = uncond_mask.view(-1, *([1] * (value.dim() - 1)))
    return torch.where(mask, -torch.ones_like(value), value)


def _multinomial_sample_one_no_sync(probs: torch.Tensor) -> torch.Tensor:
    noise = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / noise, dim=-1, keepdim=True).to(dtype=torch.int)


def sample_topk(logits: torch.Tensor, topk: int, temperature: float) -> torch.Tensor:
    logits = logits / temperature
    threshold = torch.topk(logits, topk)[0][..., -1, None]
    logits = logits.masked_fill(logits < threshold, -float("inf"))
    logits = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return _multinomial_sample_one_no_sync(probs)


def _build_symbolic_adaptor_decoder(
    hidden_dim: int, num_heads: int, num_layers: int
) -> torchtune.modules.transformer.TransformerDecoder:
    decoder = llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        embed_dim=hidden_dim,
        max_seq_len=5000,
        intermediate_dim=hidden_dim * 4,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )
    decoder.tok_embeddings = nn.Identity()
    decoder.output = nn.Identity()
    return decoder


class SymbolicAdaptor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        bottleneck_dim: int = 24,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.pianoroll_proj = nn.Linear(256, bottleneck_dim)
        self.drum_pianoroll_proj = nn.Linear(128, bottleneck_dim)
        self.chord_proj = nn.Linear(12, bottleneck_dim)
        self.fusion_proj = nn.Linear(bottleneck_dim * 3, hidden_dim)
        self.decoder = _build_symbolic_adaptor_decoder(
            hidden_dim, num_heads, num_layers
        )
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(
        self,
        pianoroll: torch.Tensor,
        drum_pianoroll: torch.Tensor,
        chord: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pianoroll = self.pianoroll_proj(pianoroll)
        drum_pianoroll = self.drum_pianoroll_proj(drum_pianoroll)
        chord = self.chord_proj(chord)
        hidden_states = self.fusion_proj(
            torch.cat([pianoroll, drum_pianoroll, chord], dim=-1)
        )
        attention_mask = _build_attention_mask(mask, hidden_states)
        hidden_states = self.decoder(hidden_states, mask=attention_mask)
        return self.out_proj(hidden_states)


class CrossAttentionRoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(
            num_heads * self.head_dim, embed_dim, bias=False
        )
        self.query_rope = Llama3ScaledRoPE(
            dim=self.head_dim,
            max_seq_len=8192,
            base=500_000,
            scale_factor=32,
        )
        self.key_rope = Llama3ScaledRoPE(
            dim=self.head_dim,
            max_seq_len=5000,
            base=500_000,
            scale_factor=32,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        context_len = context.shape[1]
        queries_per_kv = self.num_heads // self.num_kv_heads

        query = self.q_proj(x).view(
            batch_size, query_len, self.num_heads, self.head_dim
        )
        query = self.query_rope(query, input_pos=input_pos).transpose(1, 2)

        key = self.k_proj(context).view(
            batch_size, context_len, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(context).view(
            batch_size, context_len, self.num_kv_heads, self.head_dim
        )
        key = self.key_rope(key)

        key = key.unsqueeze(3).expand(
            batch_size,
            context_len,
            self.num_kv_heads,
            queries_per_kv,
            self.head_dim,
        )
        value = value.unsqueeze(3).expand_as(key)
        key = key.reshape(
            batch_size, context_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch_size, context_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_mask = _build_attention_mask(mask, x)
        attention_mask = attention_mask.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
        )
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.view(batch_size, query_len, -1)
        return self.output_proj(hidden_states)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        gate_dim: int = 12,
        gate_num_heads: int = 1,
    ):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps=1e-5)
        self.attn = CrossAttentionRoPE(embed_dim, num_heads, num_heads)
        self.x_gate_proj = nn.Linear(embed_dim, gate_dim)
        self.context_gate_proj = nn.Linear(embed_dim, gate_dim)
        self.gate = CrossAttentionRoPE(gate_dim, gate_num_heads, gate_num_heads)
        self.gate_proj = nn.Linear(gate_dim, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        attention_output = self.attn(
            hidden_states,
            context,
            mask,
            input_pos=input_pos,
        )
        gate_output = self.gate(
            self.x_gate_proj(hidden_states),
            self.context_gate_proj(context),
            mask,
            input_pos=input_pos,
        )
        gate = torch.tanh(self.gate_proj(gate_output))
        return residual + gate * attention_output


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(embed_dim, num_heads)
        self.adaptor = SymbolicAdaptor(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pianoroll: torch.Tensor,
        drum_pianoroll: torch.Tensor,
        chord: torch.Tensor,
        context_mask: torch.Tensor,
        previous_context: Optional[torch.Tensor],
        input_pos: torch.Tensor,
    ):
        current_context = self.adaptor(
            pianoroll,
            drum_pianoroll,
            chord,
            context_mask,
        )
        context = (
            current_context
            if previous_context is None
            else previous_context + current_context
        )
        hidden_states = self.cross_attn(
            hidden_states,
            context,
            context_mask,
            input_pos,
        )
        return hidden_states, context


class StyleMLP(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.mlp = nn.Linear(embed_dim, hidden_dim * 2)
        nn.init.zeros_(self.mlp.weight)
        nn.init.zeros_(self.mlp.bias)

    def forward(self, style_embedding: torch.Tensor):
        return self.mlp(self.act(style_embedding)).chunk(2, dim=-1)


class MuLaCover(PreTrainedModel):
    config_class = MuLaCoverConfig

    def __init__(self, config: MuLaCoverConfig):
        super().__init__(config)

        self.backbone, backbone_dim = _prepare_transformer(
            FLAVORS[config.backbone_flavor]()
        )
        self.decoder, decoder_dim = _prepare_transformer(
            FLAVORS[config.decoder_flavor]()
        )

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks,
            backbone_dim,
        )
        self.unconditional_text_embedding = nn.Embedding(1, backbone_dim)
        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(
            backbone_dim, config.audio_vocab_size, bias=False
        )
        self.audio_head = nn.Parameter(
            torch.empty(
                config.audio_num_codebooks - 1,
                decoder_dim,
                config.audio_vocab_size,
            )
        )
        self.muq_linear = nn.Linear(config.muq_dim, backbone_dim)
        self.qwen_linear = nn.Linear(config.qwen_dim, backbone_dim)

        first_peft_layer = len(self.backbone.layers) - config.peft_layer_num
        self.peft_layers = nn.ModuleList(
            [
                nn.Identity()
                if layer_index < first_peft_layer
                else CrossAttentionLayer(backbone_dim)
                for layer_index in range(len(self.backbone.layers))
            ]
        )

        self.tag_mlp_layers = nn.ModuleList(
            [
                nn.Identity()
                if layer_index < first_peft_layer
                else StyleMLP(config.qwen_dim, backbone_dim)
                for layer_index in range(len(self.backbone.layers))
            ]
        )

        self.post_init()
        for tag_mlp_layer in self.tag_mlp_layers:
            if isinstance(tag_mlp_layer, StyleMLP):
                nn.init.zeros_(tag_mlp_layer.mlp.weight)
                nn.init.zeros_(tag_mlp_layer.mlp.bias)
        for peft_layer in self.peft_layers:
            if isinstance(peft_layer, CrossAttentionLayer):
                nn.init.zeros_(peft_layer.cross_attn.gate_proj.weight)
                nn.init.zeros_(peft_layer.cross_attn.gate_proj.bias)

    def setup_caches(self, max_batch_size: int) -> None:
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        with device:
            for transformer, sequence_length in (
                (self.backbone, self.backbone.max_seq_len),
                (self.decoder, self.config.audio_num_codebooks),
            ):
                attention_modules = [
                    module for module in transformer.modules()
                    if hasattr(module, "kv_cache")
                ]
                rebuild = False
                for attention in attention_modules:
                    cache = attention.kv_cache
                    if (
                        cache is None
                        or cache.k_cache.shape[0] != max_batch_size
                        or cache.k_cache.shape[2] != sequence_length
                        or any(
                            buffer.device != device or buffer.dtype != dtype
                            for buffer in (cache.k_cache, cache.v_cache)
                        )
                    ):
                        rebuild = True
                        break
                if rebuild:
                    # Torchtune skips setup when a cache already exists.
                    for attention in attention_modules:
                        attention.kv_cache = None
                        attention.cache_enabled = False
                    transformer.setup_caches(
                        max_batch_size, dtype,
                        decoder_max_seq_len=sequence_length,
                    )
                else:
                    for attention in attention_modules:
                        attention.cache_enabled = True
                transformer.reset_caches()

        self.register_buffer(
            "backbone_causal_mask",
            _create_causal_mask(self.backbone.max_seq_len, device),
            persistent=False,
        )
        self.register_buffer(
            "decoder_causal_mask",
            _create_causal_mask(self.config.audio_num_codebooks, device),
            persistent=False,
        )

    def generate_frame(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        input_pos: torch.Tensor,
        temperature: float,
        topk: int,
        pianoroll: torch.Tensor,
        drum_pianoroll: torch.Tensor,
        chord: torch.Tensor,
        context_mask: torch.Tensor,
        qwen_embedding: torch.Tensor,
        qwen_indices: Optional[torch.Tensor],
        muq_embedding: Optional[torch.Tensor],
        muq_indices: Optional[torch.Tensor],
        first_step: bool = False,
        cfg_scale: float = 2.0,
    ) -> torch.Tensor:
        model_dtype = next(self.parameters()).dtype
        batch_size = tokens.size(0)
        backbone_mask = _index_causal_mask(self.backbone_causal_mask, input_pos)
        qwen_prefix_embedding = qwen_embedding.detach().clone()
        style_embedding = qwen_embedding.unsqueeze(1)

        uncond_mask = None
        if cfg_scale > 1.0 and batch_size > 1:
            condition_batch_size = batch_size // 2
            uncond_mask = torch.cat(
                [
                    torch.zeros(
                        condition_batch_size,
                        dtype=torch.bool,
                        device=tokens.device,
                    ),
                    torch.ones(
                        condition_batch_size,
                        dtype=torch.bool,
                        device=tokens.device,
                    ),
                ]
            )
            pianoroll = _mask_condition(pianoroll, uncond_mask)
            drum_pianoroll = _mask_condition(drum_pianoroll, uncond_mask)
            chord = _mask_condition(chord, uncond_mask)
            # Match the original float-mask arithmetic, including promotion
            # of BF16 inputs. This is raw Qwen, not its prefix projection.
            style_mask = uncond_mask.view(batch_size, 1, 1).float()
            style_embedding = (
                style_embedding * (1 - style_mask)
                + (-torch.ones_like(style_embedding)) * style_mask
            )

        embeddings = self._embed_tokens(tokens, uncond_mask)
        hidden_states = (embeddings * tokens_mask.unsqueeze(-1)).sum(dim=2)

        if first_step:
            batch_indices = torch.arange(batch_size, device=tokens.device)
            unconditional_embedding = self.unconditional_text_embedding(
                torch.zeros(1, dtype=torch.long, device=tokens.device)
            )

            qwen_prefix_embedding = self.qwen_linear(
                qwen_prefix_embedding.to(self.qwen_linear.weight.dtype)
            )
            muq_embedding = self.muq_linear(
                muq_embedding.to(self.muq_linear.weight.dtype)
            )
            if uncond_mask is not None:
                condition_mask = uncond_mask.view(batch_size, 1)
                qwen_prefix_embedding = torch.where(
                    condition_mask,
                    unconditional_embedding,
                    qwen_prefix_embedding,
                )
                muq_embedding = torch.where(
                    condition_mask,
                    unconditional_embedding,
                    muq_embedding,
                )

            hidden_states[batch_indices, qwen_indices] = qwen_prefix_embedding.to(
                hidden_states.dtype
            )
            hidden_states[batch_indices, muq_indices] = muq_embedding.to(
                hidden_states.dtype
            )

        previous_context = None
        for backbone_layer, peft_layer, tag_mlp_layer in zip(
            self.backbone.layers,
            self.peft_layers,
            self.tag_mlp_layers,
        ):
            if self.config.train_tag and not isinstance(tag_mlp_layer, nn.Identity):
                scale, shift = tag_mlp_layer(style_embedding)
                hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = backbone_layer(
                hidden_states,
                mask=backbone_mask,
                input_pos=input_pos,
            )
            if isinstance(peft_layer, nn.Identity):
                continue
            hidden_states, previous_context = peft_layer(
                hidden_states,
                pianoroll,
                drum_pianoroll,
                chord,
                context_mask,
                previous_context,
                input_pos,
            )

        hidden_states = self.backbone.norm(hidden_states)
        last_hidden_state = hidden_states[:, -1, :]
        codebook0_logits = self.codebook0_head(last_hidden_state)
        codebook0_sample = self._sample_with_cfg(
            codebook0_logits,
            topk,
            temperature,
            cfg_scale,
        )
        codebook0_embedding = self._embed_audio(0, codebook0_sample)

        self.decoder.reset_caches()
        decoder_input = torch.cat(
            [last_hidden_state.unsqueeze(1), codebook0_embedding],
            dim=1,
        )
        samples = codebook0_sample.clone()
        decoder_pos = torch.arange(
            decoder_input.size(1), device=decoder_input.device
        ).unsqueeze(0).repeat(decoder_input.size(0), 1)

        for codebook in range(1, self.config.audio_num_codebooks):
            decoder_mask = _index_causal_mask(
                self.decoder_causal_mask,
                decoder_pos,
            )
            decoder_hidden = self.decoder(
                self.projection(decoder_input),
                input_pos=decoder_pos,
                mask=decoder_mask,
            ).to(model_dtype)
            logits = torch.mm(
                decoder_hidden[:, -1, :],
                self.audio_head[codebook - 1],
            )
            sample = self._sample_with_cfg(
                logits,
                topk,
                temperature,
                cfg_scale,
            )
            decoder_input = self._embed_audio(codebook, sample)
            samples = torch.cat([samples, sample], dim=1)
            decoder_pos = decoder_pos[:, -1:] + 1

        return samples

    def reset_caches(self) -> None:
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _sample_with_cfg(
        self,
        logits: torch.Tensor,
        topk: int,
        temperature: float,
        cfg_scale: float,
    ) -> torch.Tensor:
        if cfg_scale > 1.0 and logits.size(0) > 1:
            logits = logits[1:] + (logits[0:1] - logits[1:]) * cfg_scale
            return sample_topk(logits, topk, temperature).repeat(2, 1)
        return sample_topk(logits, topk, temperature)

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(
            tokens + codebook * self.config.audio_vocab_size
        )

    def _embed_tokens(
        self,
        tokens: torch.Tensor,
        uncond_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = tokens.size(0)
        text_embeddings = self.text_embeddings(tokens[:, :, -1])
        if uncond_mask is not None:
            unconditional_embedding = self.unconditional_text_embedding(
                torch.zeros(1, dtype=torch.long, device=tokens.device)
            )
            text_mask = uncond_mask.view(batch_size, 1, 1).expand_as(
                text_embeddings
            )
            text_embeddings = torch.where(
                text_mask,
                unconditional_embedding,
                text_embeddings,
            )
        text_embeddings = text_embeddings.unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size
            * torch.arange(
                self.config.audio_num_codebooks,
                device=tokens.device,
            )
        )
        audio_embeddings = self.audio_embeddings(audio_tokens.reshape(-1)).reshape(
            tokens.size(0),
            tokens.size(1),
            self.config.audio_num_codebooks,
            -1,
        )
        return torch.cat([audio_embeddings, text_embeddings], dim=-2)
