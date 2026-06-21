from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对聊天对话数据进行预处理，根据一定概率为没有系统提示的对话添加系统角色提示词。
    
    如果对话数据中包含工具使用信息，则直接保留原始数据不做处理；
    如果对话开头没有系统角色，则根据指定概率随机添加预设的系统提示词。
    
    Args:
        conversations (list[dict]): 聊天对话列表，每个元素是一个包含角色和内容的字典，
                                    例如 [{'role': 'user', 'content': '...'}]。
        add_system_ratio (float, optional): 添加系统提示词的概率，取值范围在0到1之间。默认为0.2。
    
    Returns:
        list[dict]: 预处理后的聊天对话列表。如果触发了添加条件，则在列表头部插入系统提示词字典；
                    否则返回原对话列表。
    """
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): 
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是tinymind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是tinymind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are tinymind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are tinymind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对聊天提示内容进行后处理，根据设定的概率移除特定的思考与输出分隔符。

    该函数检查输入的提示内容中是否包含特定的思考与输出分隔符（"<think>\n\n</think>\n\n"），
    如果包含，则以 (1 - empty_think_ratio) 的概率将其移除。此机制可用于在大多数
    情况下清理特定格式的占位符，同时保留小概率的原始格式以满足特定需求。

    Args:
        prompt_content (str): 待处理的聊天提示内容字符串。
        empty_think_ratio (float, optional): 保留特定分隔符的概率阈值，取值范围为 [0, 1]。
            默认值为 0.05，即有 5% 的概率保留该分隔符，95% 的概率移除。

    Returns:
        str: 经过处理后的提示内容字符串。如果随机数大于 empty_think_ratio，
            则返回移除分隔符后的字符串；否则返回原字符串。
    """
    if (
        "<think>\n\n</think>\n\n" in prompt_content
        and random.random() > empty_think_ratio
    ):
        prompt_content = prompt_content.replace("<think>\n\n</think>\n\n", "")
    return prompt_content

class PretrainDataset(Dataset):
    """
    用于自回归语言模型预训练的数据集类。

    本类负责从 JSON 文件中加载文本数据，将其转换为模型预训练所需的 
    input_ids、labels 和 attention_mask 格式。核心处理流程包括：对文本
    进行分词、拼接 BOS/EOS 特殊标记、对序列进行定长填充（Padding），以及
    将填充位置的标签置为 -100 以便在计算交叉熵损失时自动忽略。

    构造函数参数:
        data_path (str): 训练数据的 JSON 文件路径。数据需包含 "text" 字段。
        tokenizer: 分词器实例，需支持 `bos_token_id`、`eos_token_id` 和 
                   `pad_token_id` 属性，以及标准的 `__call__` 分词方法。
        max_length (int, optional): 序列的最大长度，默认为 512。包含 BOS 和 
                   EOS 在内，超出部分将被截断，不足部分将用 PAD 填充。

    使用限制与注意事项:
        - 传入的 tokenizer 必须已定义 BOS、EOS 和 PAD 标记，否则在获取 
          token_id 时会抛出异常。
        - 数据集会一次性加载到内存中，如果 JSON 文件极大，可能会造成内存溢出。

    代码示例:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("your-model-name")
        >>> dataset = PretrainDataset(data_path="train.json", tokenizer=tokenizer, max_length=1024)
        >>> input_ids, labels, attention_mask = dataset[0]
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # Step 1：tokenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 预留 BOS + EOS 的位置
            truncation=True,
        ).input_ids

        # Step 2：拼接 BOS + token序列 + EOS，构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]

        # Step 3：右侧用 PAD 补齐到 max_length，保证 batch 内等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Step 4：labels 与 input_ids 完全相同，但 PAD 位置置 -100，
        #         CrossEntropyLoss 会自动忽略 -100，不计入 loss
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # ！修正：返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask
