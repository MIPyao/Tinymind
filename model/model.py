import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig
import torch.nn.functional as F
from transformers.activations import ACT2FN

class TinyMindConfig(PretrainedConfig):
    """
    TinyMind模型的配置类，继承自PretrainedConfig。用于管理和存储TinyMind模型的构建配置及超参数。

    核心功能：
        - 定义标准Transformer架构参数（如隐藏层大小、注意力头数等）。
        - 支持分组查询注意力(GQA)和旋转位置编码。
        - 支持可选的混合专家系统配置，包含路由专家与共享专家机制。
        - 支持推理时的YaRN位置缩放及Flash Attention加速。

    代码示例：
        >>> config = TinyMindConfig(hidden_size=768, use_moe=True, n_routed_experts=8)
        >>> print(config.hidden_size)
        768
        >>> print(config.rope_scaling) # 默认 inference_rope_scaling=False
        None

    构造函数参数：
        dropout (float): dropout概率，默认为0.0。
        bos_token_id (int): 序列起始标记ID，默认为1。
        eos_token_id (int): 序列结束标记ID，默认为2。
        hidden_act (str): 隐藏层激活函数，默认为"silu"。
        hidden_size (int): 隐藏层维度大小，默认为512。
        intermediate_size (int): FFN中间层维度大小，若为None则通常按hidden_size推算，默认为None。
        max_position_embeddings (int): 模型支持的最大序列长度，默认为32768。
        num_attention_heads (int): 查询注意力头数，默认为8。
        num_hidden_layers (int): Transformer隐藏层数，默认为8。
        num_key_value_heads (int): 键值注意力头数（用于GQA），默认为2。
        vocab_size (int): 词表大小，默认为6400。
        rms_norm_eps (float): RMSNorm的epsilon值，默认为1e-05。
        rope_theta (int): 旋转位置编码的theta基数值，默认为1000000。
        inference_rope_scaling (bool): 是否在推理时启用YaRN位置编码缩放，默认为False。
        flash_attention (bool): 是否使用Flash Attention加速，默认为True。
        use_moe (bool): 是否启用混合专家层，默认为False。
        num_experts_per_tok (int): 每个token激活的专家数量，默认为2。
        n_routed_experts (int): 路由专家的总数量，默认为4。
        n_shared_experts (int): 共享专家的数量，默认为1。
        scoring_func (str): 专家路由评分函数，默认为"softmax"。
        aux_loss_alpha (float): 辅助损失的权重系数，默认为0.01。
        seq_aux (bool): 是否在序列维度计算辅助损失，默认为True。
        norm_topk_prob (bool): 是否对top-k专家的概率进行归一化，默认为True。
        **kwargs: 传递给父类PretrainedConfig的额外参数。

    使用限制与副作用：
        - 当 inference_rope_scaling 设为 True 时，构造函数会自动生成固定的 YaRN 缩放配置并赋值给 
          self.rope_scaling，覆盖任何外部传入的同名参数。
        - 若启用MoE（use_moe=True），需确保 num_experts_per_tok 不大于 n_routed_experts + n_shared_experts。
    """

    model_type = "‌tinymind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Normalization) 层的实现。

    该类实现了 RMS 归一化算法，与 LayerNorm 相比，它去除了均值计算，仅通过计算输入张量在指定维度上的均方根（RMS）来进行归一化。
    这种方式在减少计算开销的同时，仍能保持与 LayerNorm 相当的模型性能，常用于大语言模型（如 LLaMA）等现代深度学习架构中。

    核心功能：
        - 对输入张量沿最后一个维度计算均方根并进行归一化。
        - 通过可训练的缩放参数（weight）对归一化后的结果进行仿射变换。

    使用限制与副作用：
        - 归一化计算在 float32 精度下进行，随后转换回输入的原始数据类型，这可能会在混合精度训练中引入微小的类型转换开销。
        - 归一化仅沿最后一个维度进行，如果输入张量的维度不符合预期，可能会导致计算结果错误。

    代码示例：
        >>> import torch
        >>> norm = RMSNorm(dim=512, eps=1e-6)
        >>> x = torch.randn(2, 10, 512)
        >>> output = norm(x)
        >>> print(output.shape)
        torch.Size([2, 10, 512])

    构造函数参数：
        dim (int): 输入特征的维度大小，同时也是可训练缩放参数 weight 的维度。
        eps (float, optional): 防止除以零的极小值，添加到均方根的平方中。默认值为 1e-5。
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        """初始化层实例。
        
        Args:
            dim (int): 特征的维度大小，用于定义权重参数的形状。
            eps (float, optional): 防止数值计算中除以零错误的小常数。默认为 1e-5。
        """
        super().__init__()
        self.eps = eps  #  设置一个小的 eps 值，用于防止数值计算中的除以零错误
        self.weight = nn.Parameter(
            torch.ones(dim)
        )  #  创建一个可训练的参数 weight，初始化为 dim 维的全1向量

    def _norm(self, x):
        """对输入张量在最后一个维度上进行均方根归一化。

        计算输入张量在最后一个维度上的均方值，加上一个极小值 eps 以防止除零错误，
        然后取其平方根的倒数，并与原输入张量相乘，从而实现归一化。

        Args:
            x (torch.Tensor): 待归一化的输入张量。

        Returns:
            torch.Tensor: 归一化后的张量，形状与输入张量相同。
        """
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * x

    def forward(self, x):
        """
        前向传播计算。
       
        对输入张量应用权重缩放和归一化处理。首先将输入转换为浮点数进行归一化计算，
        然后将结果转换回原始输入的数据类型，最后乘以可学习的权重参数。
        
        Args:
        x (torch.Tensor): 输入张量。
       
        Returns:
        torch.Tensor: 经过归一化和权重缩放后的张量，数据类型与输入 `x` 相同。
        """
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    """
    预计算旋转位置编码的复数频率，支持标准 RoPE 和 YaRN (Yet another RoPE extensioN) 缩放策略。

    Args:
        dim (int): 旋转位置编码的维度大小（通常等于注意力头的大小）。
        end (int): 序列的最大长度，即需要计算的位置数量。
        rope_base (float, optional): RoPE 的基础频率（theta），默认为 1e6。
        rope_scaling (Optional[dict], optional): YaRN 缩放配置字典。如果不为 None，则应用 YaRN 缩放。
            字典可包含以下键:
            - original_max_position_embeddings (int): 预训练时的原始最大序列长度，默认为 2048。
            - factor (float): 上下文扩展的缩放因子，默认为 16。
            - beta_fast (float): 高频边界比例（对应论文中的 α），大于此波长比例的维度不缩放，默认为 32.0。
            - beta_slow (float): 低频边界比例（对应论文中的 β），小于此波长比例的维度全量缩放，默认为 1.0。
            - attention_factor (float): 注意力温度补偿系数，用于缓解上下文扩展导致的注意力分布发散，默认为 1.0。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 包含两个张量的元组 (freqs_cos, freqs_sin)。
            - freqs_cos: 形状为 (end, dim) 的余弦频率张量，已乘以注意力补偿系数。
            - freqs_sin: 形状为 (end, dim) 的正弦频率张量，已乘以注意力补偿系数。
    """
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d))
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end > orig_max:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            def inv_dim(b):
                return (
                    (dim * math.log(orig_max / (b * 2 * math.pi)))
                    / (2 * math.log(rope_base))
                )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """对查询张量和键张量应用旋转位置编码。

    该函数通过将输入张量与余弦和正弦位置嵌入结合，实现了旋转位置编码。
    它将输入张量分成两半，将其中一半取反并交换位置，然后与正弦和余弦
    值相乘，最后将结果相加，从而实现向量的旋转。

    Args:
        q (torch.Tensor): 查询张量，形状通常为 [batch_size, seq_len, num_heads, head_dim]。
        k (torch.Tensor): 键张量，形状通常与 q 相同。
        cos (torch.Tensor): 余弦位置嵌入，形状通常为 [batch_size, seq_len, head_dim] 或可广播的形状。
        sin (torch.Tensor): 正弦位置嵌入，形状通常与 cos 相同。
        unsqueeze_dim (int, optional): 在 cos 和 sin 张量上增加维度的位置索引，
            以便与 q 和 k 的形状进行广播。默认为 1。

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 包含应用了旋转位置编码后的查询张量和键张量的元组，
            两者均保持与输入张量相同的数据类型。
    """
    def rotate_half(x):
        """
        将输入张量的后半部分取反后与前半部分拼接，实现旋转半周的操作。
        
        该函数常用于旋转位置编码（Rotary Position Embedding, RoPE）中，
        对特征向量的两半进行交替取反和位置互换。
        
        Args:
            x (torch.Tensor): 输入张量，要求其最后一个维度的长度必须是偶数。
            
        Returns:
            torch.Tensor: 旋转后的张量，形状与输入张量 `x` 相同。
        """
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )).to(k.dtype)

    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    对键值对张量中的头维度进行重复扩展，常用于大语言模型中的多查询注意力（MQA）或分组查询注意力（GQA）机制。
    
    该函数将输入张量在 `num_key_value_heads` 维度上重复 `n_rep` 次，以匹配查询的注意力头数量，
    从而在节省 KV Cache 内存占用的同时，支持更多的注意力头并行计算。

    Args:
        x (torch.Tensor): 输入的键或值张量，形状为 (bs, slen, num_key_value_heads, head_dim)。
        n_rep (int): 每个键值头需要重复的次数，通常等于 num_attention_heads // num_key_value_heads。

    Returns:
        torch.Tensor: 扩展后的张量，形状为 (bs, slen, num_key_value_heads * n_rep, head_dim)。
                      如果 n_rep == 1，则直接返回原始张量。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    基于 Grouped-Query Attention (GQA) 和 Rotary Position Embedding (RoPE) 的注意力层。
    
    本类实现了 Transformer 架构中的自注意力机制，支持多查询/分组查询注意力(MQA/GQA)以优化推理显存，
    集成了旋转位置编码，并支持 KV Cache 加速自回归生成。在计算注意力分数时，会根据运行环境和
    输入条件自动选择使用 PyTorch 原生的 Flash Attention 或手动实现的注意力计算。

    核心功能:
        - Q, K, V 线性投影及输出投影
        - 支持 Grouped-Query Attention (GQA)
        - 集成旋转位置编码
        - 支持 KV Cache 以加速自回归推理
        - 自动判断并应用 Flash Attention 或传统注意力计算
        - 注意力权重与残差输出的 Dropout

    构造函数参数:
        config (TinyMindConfig): 模型配置对象，需包含以下属性：
            - num_attention_heads (int): 查询的注意力头数。
            - num_key_value_heads (Optional[int]): 键和值的注意力头数。若为 None，则与查询头数相同（即标准 MHA）。
            - hidden_size (int): 隐藏层维度大小。
            - dropout (float): Dropout 概率。
            - flash_attention (bool): 是否在条件允许时启用 Flash Attention。

    使用限制与副作用:
        - 限制: `config.num_attention_heads` 必须能被 `config.num_key_value_heads` 整除，否则将触发断言错误。
        - 限制: Flash Attention 仅在序列长度大于1、无历史 KV Cache、且无自定义注意力掩码（或掩码全为1）时启用。
        - 副作用: 当 `use_cache=True` 时，会返回当前步的 KV 张量元组，需由外部管理其生命周期与显存占用。

    代码示例:
        >>> config = TinyMindConfig(
        ...     num_attention_heads=32, 
        ...     num_key_value_heads=8, 
        ...     hidden_size=4096, 
        ...     dropout=0.0, 
        ...     flash_attention=True
        ... )
        >>> attn = Attention(config)
        >>> x = torch.randn(2, 10, 4096)  # (batch_size, seq_len, hidden_size)
        >>> cos, sin = torch.randn(10, 64), torch.randn(10, 64)  # 假设 head_dim=128 的一半
        >>> output, past_kv = attn(x, position_embeddings=(cos, sin), use_cache=True)
    """
    def __init__(self, config: TinyMindConfig):
        """
        初始化注意力机制模块。

        根据提供的配置初始化查询、键、值和输出投影层，以及相关的注意力参数。
        支持分组查询注意力 和 Flash Attention。

        Args:
            config (TinyMindConfig): 模型配置对象，包含以下必要属性：
                - num_attention_heads (int): 查询的注意力头数。
                - num_key_value_heads (Optional[int]): 键和值的注意力头数。如果为None，则与num_attention_heads相同。
                - hidden_size (int): 隐藏层维度大小。
                - dropout (float): Dropout概率。
                - flash_attention (bool): 是否启用Flash Attention。

        Raises:
            AssertionError: 如果查询头数不能被键值头数整除。
        """
        super().__init__()

        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )

        assert config.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attention
        )
    
    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        模型注意力层的前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, hidden_size)。
            position_embeddings (Tuple[torch.Tensor, torch.Tensor]): 旋转位置编码的余弦和正弦值元组 (cos, sin)。
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): 过去的键值对缓存，用于加速自回归生成。默认为 None。
            use_cache (bool, optional): 是否返回当前的键值对以供后续生成步骤使用。默认为 False。
            attention_mask (Optional[torch.Tensor], optional): 注意力掩码张量，用于屏蔽特定位置的注意力计算。默认为 None。

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: 
                - output (torch.Tensor): 注意力层的输出张量，形状为 (batch_size, seq_len, hidden_size)。
                - past_kv (Optional[Tuple[torch.Tensor, torch.Tensor]]): 如果 use_cache 为 True，则返回当前时间步的键值对缓存，否则返回 None。
        """
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        if (
            self.flash
            and (seq_len > 1)
            and (past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores[:, :, :, -seq_len:] += torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    基于 SwiGLU 激活函数的前馈神经网络模块。

    该模块实现了带有门控机制的前馈神经网络层，主要用于 Transformer 等模型中，
    通过升维、非线性激活与门控、再降维的过程来增强模型的表达能力。

    核心功能：
        - 门控线性单元：结合门控投影和上投影，并应用非线性激活函数。
        - 维度映射：将输入特征从隐藏维度映射到中间维度，再映射回隐藏维度。
        - 正则化：在输出前应用 Dropout 以防止过拟合。

    构造函数参数：
        config (TinyMindConfig): 模型的配置对象，需包含以下属性：
            - intermediate_size (int, optional): 中间层维度。如果为 None，将根据 hidden_size 自动计算并向上取整至 64 的倍数。
            - hidden_size (int): 输入和输出的隐藏层维度。
            - dropout (float): Dropout 层的丢弃概率。
            - hidden_act (str): 隐藏层激活函数的名称标识。

    使用限制与副作用：
        - 副作用：当 config.intermediate_size 为 None 时，会自动计算该值并直接修改传入的 config 对象的 intermediate_size 属性。
        - 限制：所有线性层均未使用偏置项 (bias=False)。
        - 限制：依赖全局的 ACT2FN 字典来根据 config.hidden_act 获取实际的激活函数实例。

    代码示例：
        >>> config = TinyMindConfig(hidden_size=512, intermediate_size=None, dropout=0.1, hidden_act="silu")
        >>> ffn = FeedForward(config)
        >>> import torch
        >>> x = torch.randn(2, 10, 512)  # (batch_size, seq_len, hidden_size)
        >>> output = ffn(x)
        >>> print(output.shape)
        # torch.Size([2, 10, 512])
    """
    def __init__(self, config: TinyMindConfig):
        """
        初始化 TinyMind 模型的 MLP (多层感知机) 层。

        如果配置中未指定中间层大小 (`intermediate_size`)，则会根据隐藏层大小自动计算，
        并将其向上取整为 64 的倍数，以优化硬件计算效率。
        随后初始化门控投影、下投影、上投影线性层，以及 Dropout 和激活函数。

        Args:
            config (TinyMindConfig): 模型配置对象，包含以下必要属性：
                - intermediate_size (Optional[int]): MLP 中间层的维度。如果为 None，则自动计算。
                - hidden_size (int): 隐藏层的维度。
                - dropout (float): Dropout 的概率。
                - hidden_act (str): 隐藏层激活函数的名称。
        """
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        前向传播函数。
        
        实现了带有门控机制的前馈神经网络层。首先通过门控投影和上投影对输入进行变换，
        并将门控激活结果与上投影结果进行逐元素相乘，然后将结果通过下投影层，
        最后应用 dropout 操作并返回结果。
        
        Args:
            x (torch.Tensor): 输入张量。
        
        Returns:
            torch.Tensor: 经过门控前馈网络和 dropout 处理后的输出张量。
        """
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))

class TinyMindBlock(nn.Module):
    """
    TinyMind 模型的 Transformer 基础层（Block）。
    
    该类实现了 Pre-Norm 架构的标准 Transformer 层，将输入隐藏状态依次通过
    自注意力机制和前馈神经网络（MLP），并使用残差连接进行特征融合。
    
    核心功能：
        - 自注意力计算：通过内部 Attention 模块实现多头自注意力机制。
        - 前馈网络：通过 FeedForward 模块对特征进行非线性变换。
        - 层归一化：使用 RMSNorm 对自注意力层和 MLP 层的输入进行归一化。
        - KV Cache：支持返回当前层的 Key/Value 缓存，以加速自回归生成。
    
    构造函数参数：
        layer_id (int): 当前层的索引ID，用于区分模型中的不同层。
        config (TinyMindConfig): 模型配置对象，包含如隐藏层大小、注意力头数、
                                 归一化 epsilon 等构建该层所需的超参数。
    
    使用限制与副作用：
        - 该层默认使用标准的 FeedForward 网络，代码中注释了 MOEFeedForward 的分支，
          如果需要使用混合专家模型机制，需手动修改代码取消相关注释。
        - 必须作为 nn.Module 的子类在模型框架中使用，不可独立进行前向传播。
    
    代码示例：
        >>> config = TinyMindConfig(hidden_size=512, num_attention_heads=8, rms_norm_eps=1e-6)
        >>> block = TinyMindBlock(layer_id=0, config=config)
        >>> hidden_states = torch.randn(1, 10, 512)
        >>> pos_emb = (torch.randn(1, 10, 64), torch.randn(1, 10, 64))
        >>> output, present_kv = block(hidden_states, position_embeddings=pos_emb)
    """
    def __init__(self, layer_id: int, config: TinyMindConfig):
        """
        初始化 Transformer 模型层。
        
        Args:
            layer_id (int): 当前层的索引ID。
            config (TinyMindConfig): 模型配置对象，包含模型的各种超参数。
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = FeedForward(config)
            # if not config.use_moe
            # else MOEFeedForward(config))

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        模型前向传播方法。

        Args:
            hidden_states (torch.Tensor): 输入的隐藏状态张量。
            position_embeddings (Tuple[torch.Tensor, torch.Tensor]): 位置嵌入元组，通常包含位置和方向的嵌入。
            past_key_value (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): 过去计算的键值对元组，用于加速推理。默认为 None。
            use_cache (bool, optional): 是否返回键值对以供后续推理使用。默认为 False。
            attention_mask (Optional[torch.Tensor], optional): 注意力掩码张量，用于避免关注填充或未来的标记。默认为 None。

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                - hidden_states (torch.Tensor): 经过自注意力层和MLP层后的输出隐藏状态。
                - present_key_value (Tuple[torch.Tensor, torch.Tensor]): 当前层的键值对，用于后续增量推理。
        """
        res = hidden_states

        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = res + hidden_states

        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value
