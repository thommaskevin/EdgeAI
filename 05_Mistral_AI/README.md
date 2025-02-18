# EdgeAI -Mistral AI _on Raspberry Pi

_From mathematical foundations to edge implementation_

**Social media:**

👨🏽‍💻 Github: [thommaskevin/TinyML](https://github.com/thommaskevin/TinyML)

👷🏾 Linkedin: [Thommas Kevin](https://www.linkedin.com/in/thommas-kevin-ab9810166/)

📽 Youtube: [Thommas Kevin](https://www.youtube.com/channel/UC7uazGXaMIE6MNkHg4ll9oA)

:pencil2:CV Lattes CNPq: [Thommas Kevin Sales Flores](http://lattes.cnpq.br/0630479458408181)

👨🏻‍🏫 Research group: [Conecta.ai](https://conect2ai.dca.ufrn.br/)

![Figure 1](./figures/fig00.png)

## SUMMARY

1 — Introduction

2 — Mistral AI 7B

2.1 — Sliding Window Attention 

2.2 — Rolling Buffer Cache

2.3 — Pre-fill and Chunking

3 — EdgeAI Implementation

---

## 1 - Introduction

Mistral 7B is a high-performance, open-weight large language model developed by Mistral AI. Released in September 2023, it features 7.3 billion parameters and is designed for efficiency, speed, and accessibility. Despite its relatively compact size, Mistral 7B outperforms larger models like Llama 2 13B and even challenges Llama 2 34B in various benchmarks.

Built with advanced transformer optimizations, such as grouped-query attention (GQA) and sliding window attention (SWA), Mistral 7B delivers fast inference and improved memory efficiency. As an open-weight model, it provides researchers and developers with full access to its architecture, making it ideal for customization and deployment across diverse AI applications, including text generation, summarization, chatbots, and code completion.

With its blend of power, efficiency, and transparency, Mistral 7B stands as a benchmark for modern open-source AI models, pushing the boundaries of what smaller yet optimized language models can achieve.

## 2 - Mistral AI 7B

Mistral 7B is based on a transformer architecture. The main parameters of the architecture are summarized in Table.

![Figure 1](./figures/fig01.png)

### 2.1 — Root Mean Square Normalization (RMSNorm)

**RMSNorm** is a normalization technique used in transformer architectures like **Mistral 7B**, which differs from **LayerNorm** by normalizing activations based on the **root mean square (RMS)** instead of mean and variance. This approach improves computational efficiency, making it more suitable for large models.  


Given an input $x$ of dimension $d$, RMSNorm is defined as:

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x)}
$$

where:
- $\gamma$ is a trainable parameter that allows the model to adjust the scale of the normalized values.
- $\text{RMS}(x)$ is the normalization factor, computed as:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}
$$

Unlike **LayerNorm**, which centers the values by subtracting the mean and then dividing by the standard deviation, **RMSNorm ignores the mean**, reducing computational complexity. This means that, in terms of performance, while LayerNorm requires additional summation and division operations to compute the mean and variance, RMSNorm simply calculates the root mean square and uses it for normalization.


Consider an input vector:

$$
x = [2.0, -3.0, 4.0, -1.0]
$$

First, we compute the RMS:

$$
\text{RMS}(x) = \sqrt{\frac{1}{4} (2^2 + (-3)^2 + 4^2 + (-1)^2)} = \sqrt{\frac{30}{4}} = \sqrt{7.5} \approx 2.74
$$

Next, we normalize the values by dividing each element by the RMS:

$$
\hat{x} = \left[ \frac{2.0}{2.74}, \frac{-3.0}{2.74}, \frac{4.0}{2.74}, \frac{-1.0}{2.74} \right] \approx [0.73, -1.09, 1.46, -0.36]
$$

Finally, we apply the scaling factor $\gamma$. Assuming $\gamma = 1.5$, the final normalized values are:

$$
y = 1.5 \times [0.73, -1.09, 1.46, -0.36] \approx [1.10, -1.64, 2.19, -0.55]
$$


In computational terms, **RMSNorm reduces the complexity of normalization calculations** and saves memory, making it more efficient for models, which handle billions of parameters. However, because it does not center the data, it may slightly alter the learning dynamics compared to LayerNorm. Nonetheless, its impact on convergence has been positive in various **efficiency-optimized transformers**, justifying its adoption.


### 2.2 — Self-Attention (GQA, SWA) 

Self-attention is a key mechanism in transformer architectures to efficiently process and capture relationships between tokens. The **Mistral 7B** model incorporates optimizations such as **Grouped Query Attention (GQA)** and **Sliding Window Attention (SWA)** to improve efficiency and scalability when handling long sequences.



Given an input matrix $X \in \mathbb{R}^{n \times d} $, self-attention computes the output as:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

where:
- $Q = XW_Q $ (queries)
- $K = XW_K $ (keys)
- $V = XW_V $ (values)
- $d_k $ is the dimensionality of each key and query
- $W_Q, W_K, W_V $ are trainable projection matrices

Traditional self-attention scales poorly for large sequences due to the **$O(n^2) $** complexity. **GQA** and **SWA** address this issue.

---

#### 2.2.1 — Grouped Query Attention (GQA)

**GQA** reduces memory usage by decreasing the number of **key-value heads**, which optimizes computational efficiency while maintaining strong model performance. Instead of having independent query, key, and value heads for each attention head, GQA **groups multiple queries to share the same key-value representations**.


Instead of having $h$ independent key-value heads, GQA uses a smaller number $g $, where $g < h $. Each group of query heads shares the same key-value representations:

$$
\text{Attention}_{\text{GQA}}(Q, K, V) = \text{softmax} \left( \frac{Q K_g^T}{\sqrt{d_k}} \right) V_g
$$

where:
- $Q $ has $h $ heads.
- $K_g, V_g $ are **shared across multiple query heads**, reducing memory usage.
- The number of key-value heads $g $ is significantly smaller than $h $.

This results in:
- **Lower memory footprint**, as fewer key-value pairs are stored.
- **Improved efficiency**, especially in large-scale models.


Consider a model with **h = 8 query heads** and **g = 2 key-value heads**. Instead of computing attention for 8 separate key-value sets, the model **groups queries into 2 sets**, effectively reducing computation.

---

#### 2.2.2 — Sliding Window Attention (SWA)

**SWA** improves performance by **restricting attention to a local window**, making it more efficient for long sequences. Instead of attending to all tokens in the sequence, each token attends only to a fixed-sized window of surrounding tokens.


For a token at position $i $, attention is computed only over a window of size $w $:

$$
\text{Attention}_{\text{SWA}}(i) = \text{softmax} \left( \frac{Q_i K_{i-w:i+w}^T}{\sqrt{d_k}} \right) V_{i-w:i+w}
$$

where:
- $w $ defines the window size (e.g., 128 tokens).
- $K_{i-w:i+w} $ and $V_{i-w:i+w} $ are **local key-value pairs** instead of global ones.

This reduces complexity from **$O(n^2) $ to $O(nw) $**, where $w $ is much smaller than $n $.


If a sequence has **n = 1024** tokens and a window size of **w = 128**, each token only computes attention over **256 tokens** (128 on the left, 128 on the right), instead of all **1024 tokens**.

This significantly improves:
- **Speed**, as fewer attention weights need to be computed.
- **Memory efficiency**, since the attention matrix size is reduced.

---

#### 2.2.3 — Rotary Position Embeddings (RoPE)

**RoPE** introduces **relative positional information** into the attention mechanism by applying a **rotational transformation** to the queries and keys.
 

Given a token representation $x $ at position $i $, RoPE applies a **complex-valued rotation**:

$$
\text{RoPE}(x, i) = x e^{i \theta_i}
$$

where:
- $\theta_i = \theta_0^i $ is a **frequency-based position encoding**.
- The rotation is applied directly to queries and keys before attention is computed.

This allows:
- **Better generalization to long sequences**.
- **Implicit modeling of relative positions** without needing explicit embeddings.


If two tokens are **64 positions apart**, RoPE naturally encodes this separation in their transformed embeddings, maintaining meaningful positional relationships even for extrapolated sequences.




### 2.3 — Rolling Buffer Cache




### 2.3 — Pre-fill and Chunking





## 3 - EdgeAI Implementation

With this example you can implement the machine learning algorithm in Raspberry Pi.

### 3.0 - Gather the necessary materials

- Raspberry Pi 5 (with a compatible power cable)

- MicroSD card (minimum 64 GB, 126 GB or higher recommended)

- Computer with an SD card reader or USB adapter

- HDMI cable and a monitor/TV

- USB keyboard and mouse (or Bluetooth if supported)

- Internet connection (via Wi-Fi or Ethernet cable)


### 3.1 - Download and install the operating system


Visit [here](https://medium.com/@thommaskevin/edgeai-llama-on-raspberry-pi-4-4dffd65d33ab) to do how download and install the operating system in Raspberry pi 4 or 5.



### 3.2 - Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```


### 3.3 - Run DeepSeek

Link: [deepseek-r1 Models](https://ollama.com/library/deepseek-r1:1.5b)

![Figure 3](./figures/fig22.png)


```bash
ollama run deepseek-r1:{version}
```

We use in this example the 1.5b version

```bash
ollama run deepseek-r1:1.5b
```

![Figure 3](./figures/fig23.png)


### 3.4 - Results for question

The question: explain the LLM models


![Figure 3](./figures/fig24.png)





**References:**

- https://ollama.com/library/deepseek-r1:1.5b

- https://arxiv.org/abs/2501.12948

- https://www.geeksforgeeks.org/deepseek-r1-technical-overview-of-its-architecture-and-innovations/
