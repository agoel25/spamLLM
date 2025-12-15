# spamLLM

**spamLLM** is a spam detection system that uses generative language models for classification via Bayesian inverse inference. Instead of training a discriminative classifier, it leverages the generative capabilities of decoder-only transformers to compute posterior probabilities over class labels.

The system achieves **87% accuracy** on the Enron email dataset using LoRA fine-tuning with only ~0.1% of the model's parameters trainable.


## Bayesian Inverse Classification

Traditional classifiers model $P(Y|X)$ directly. This approach inverts the problem: given a generative LLM that models $P(X|Y)$, we use Bayes' rule to compute the posterior:

$$
P(Y_\text{label}|X) = \frac{P(X|Y_\text{label}) \cdot P(Y_\text{label})}{\sum_{Y'} P(X|Y') \cdot P(Y')}
$$

For each email, we compute the sequence log-probability under two labelings ("spam" and "ham"), apply a prior (default: 90% ham, 10% spam), and classify via softmax. This exploits the LLM's language understanding without requiring task-specific classification heads.


## Model Architecture

The implementation is a decoder-only transformer following the LLaMA architecture:

1. **Attention with KV Caching**: Multi-headed attention with Grouped Query Attention (GQA) for memory efficiency. Key-value pairs are cached per layer in `DynamicCache`, enabling O(1) incremental token generation instead of O(n) recomputation.

2. **Rotary Position Embeddings (RoPE)**: Position information encoded via rotation matrices applied to queries and keys, enabling better length generalization than absolute positional encodings.

3. **Pre-Norm Architecture**: RMSNorm applied before each sub-layer (attention and MLP) with residual connections, providing more stable training for deep networks.

4. **Parameter-Efficient Fine-Tuning**:
   - **LoRA**: Low-rank adaptation matrices (A, B) injected into attention projections. Only ~50K parameters trained vs ~135M total.
   - **Prefix Tuning**: Learnable key-value prefixes prepended to each layer's KV cache.

```
Token IDs → Embeddings → [RMSNorm → Attention → + → RMSNorm → MLP → +] × N → RMSNorm → LM Head → Log Probs
                              ↑
                         RoPE + KV Cache
```


## Training Methods

| Method | Description | Trainable Params |
|--------|-------------|------------------|
| Zero-Shot | Direct inference with base model | 0 |
| Naive Prompting | Custom prompt prefix at inference | 0 |
| Full Fine-Tune | Update all model parameters | 135M |
| LoRA | Low-rank attention adaptation | ~50K |
| Prefix Tuning | Learnable KV cache prefixes | ~100K |


## Get Started

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation
1. Clone the repository
    ```bash
    git clone https://github.com/agoel25/spamLLM.git
    cd spamLLM
    ```

2. Install dependencies
    ```bash
    uv sync
    ```

3. Configure environment: create a `.env` file with your model settings
    ```
    MODEL_CHECKPOINT=HuggingFaceTB/SmolLM2-135M-Instruct
    MODEL_CACHE_DIR=./cache
    PROJECT_NAME=spamLLM
    ```

### Training
```bash
# Zero-shot evaluation (no training)
bash examples/bayes_inverse_zero_shot.sh

# LoRA fine-tuning (recommended)
bash examples/bayes_inverse_lora.sh

# Full model fine-tuning
bash examples/bayes_inverse_full_finetune.sh
```

### Inference
```bash
# Generate predictions on test set
uv run -m examples.save_prob_example

# Interactive chatbot mode
uv run -m examples.chatbot_example
```


## System Structure

```
spamLLM
│
├── model/
│   ├── llama.py          # Main LlamaModel: embeddings → layers → LM head
│   ├── attention.py      # GQA attention with RoPE and causal masking
│   ├── cache.py          # DynamicCache for KV caching during generation
│   ├── lora.py           # LoRA layer wrapper and model patching
│   └── prefix_llama.py   # Prefix tuning with learnable KV prefixes
│
├── examples/
│   ├── bayes_inverse.py      # Core training/eval loop
│   ├── bayes_inverse_lora.py # LoRA-specific training
│   └── save_prob_example.py  # Inference script for generating predictions
│
└── utils/
    ├── prompt_template.py    # Email → prompt formatting
    └── weight_utils.py       # Model weight loading from HuggingFace
```


## References

1. Vaswani et al., "Attention Is All You Need" (2017): [paper](https://arxiv.org/abs/1706.03762)
2. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021): [paper](https://arxiv.org/abs/2106.09685)
3. Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021): [paper](https://arxiv.org/abs/2101.00190)
4. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021): [paper](https://arxiv.org/abs/2104.09864)
5. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023): [paper](https://arxiv.org/abs/2305.13245)
