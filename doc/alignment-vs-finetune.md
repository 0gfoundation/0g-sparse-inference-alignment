# Alignment and Value Model Roles: Why Not Fine-tune the Base Model Directly?

---

## 1. What is Alignment?

Alignment is the process of making an LLM's outputs conform to human expectations — typically making the model more **Helpful**, more **Harmless**, and more **Honest**.

A pre-trained base model only learns to "generate natural text," not necessarily to "generate text that is helpful to users." Alignment is a layer of directional guidance built on top of base capabilities, steering the model to prefer outputs that align with human values among multiple candidate directions.

---

## 2. The Role of Value Model in SIA

The Value Model is a specially trained small model whose job is not to "generate text" but to "evaluate the quality of text fragments" — specifically, at each token generation step, it estimates "how high the quality of the subsequent response would be if this token were chosen."

The difference from a Reward Model (RM) is:
- Traditional RM: can only score **complete responses**, cannot guide each intermediate step during generation
- Value Model: can score **partial sequences**, supporting token-level intervention

SIA's Value Model is based on Qwen3-4B-Base with LoRA, trained on UltraFeedback (Helpfulness) and WildGuardMix (Harmlessness) data. The training objective is to teach it "which direction better aligns with human preferences."

---

## 3. What are the Advantages over Directly Fine-tuning the Base Model?

### Advantage 1: Much smaller training data requirement (core advantage explicitly confirmed by paper authors)

The paper authors state: **the amount of data needed to train a Value Model is far less than the data needed to fine-tune a base model to achieve the same improvement.**

The reason lies in the fundamentally different training objectives:

- Fine-tuning the base model: requires the model to **change its internal representations**, so that the weights themselves memorize "what constitutes a good response." This requires a large number of comprehensive examples to stably change the behavior of a model with tens of billions of parameters.
- Training a Value Model: the base model's capabilities and knowledge are already present; the Value Model only needs to learn to **distinguish direction within the base model's distribution space** — which token leads toward better outcomes, which toward worse. This is a relatively low-dimensional discriminative task with high signal-to-noise ratio in preference data; a small amount of data is sufficient for effective training.

Analogy: teaching an already skilled chess player to "prioritize safe defense" requires far fewer examples than teaching all chess theory from scratch.

### Advantage 2: Does not affect the base model's general capabilities (no Alignment Tax)

Fine-tuning the base model (especially RLHF-type methods) often causes "Alignment Tax" — the model aligns with the target direction but damages capabilities unrelated to the alignment target, i.e., Catastrophic Forgetting.

SIA's Value Model does not modify the base model weights at all. MMLU regression testing confirms: with SIA enabled, accuracy on knowledge Q&A tasks unrelated to the Value Model's training objective shows no significant change (83.48% vs 82.13%, difference within statistical error). The base model's general capabilities are fully preserved.

### Advantage 3: Modular, plug-and-play

Once a base model is fine-tuned, its behavior is fixed. The Value Model is an independent module:

- Target can be switched at any time: swap in a Value Model trained on safety data, and the same base model becomes a version prioritizing Harmlessness
- Can be hot-swapped online (SIA service's `/reload` endpoint supports replacing the Value Model without restarting vLLM)
- Weight can be adjusted per scenario (`--weight` parameter), even set to 0 to completely disable intervention and fall back to plain vLLM inference

Fine-tuned base models do not have this flexibility — changing the target requires retraining.

### Advantage 4: Weak-to-Strong Generalization

Paper experiments show that a Value Model trained on **Qwen3-4B** can effectively guide **Qwen3-14B**, a model 3.5× larger than it.

This conclusion is significant: the Value Model does not need to be as strong as the LLM it guides; it only needs sufficient "sense of direction" at critical decision points. Using a small model to guide a large one further reduces training cost.

### Advantage 5: Inference-time alignment, more flexible deployment

Fine-tuning is a one-time offline cost, but its result is fixed. SIA's Value Model is an inference-time component, which means:

- The same base model can simultaneously serve users with different alignment objectives (different requests use different Value Models)
- When alignment objectives are updated, only the small Value Model needs to be retrained, without touching the already-deployed large model

---

## 4. Limitations of the Value Model Approach

Objectively, Value Model + inference-time intervention also has limitations:

1. **Increased inference latency**: each intervention requires an additional Value Model forward pass (even after batch forward optimization, intervention steps still take approximately 119 ms vs 15 ms for non-intervention steps). Fine-tuning the base model does not change inference speed.

2. **Cannot correct low-entropy errors**: if the model is very confident about a wrong answer (low entropy), SIA does not trigger and the Value Model never intervenes. Fine-tuning can directly modify the model's "prior knowledge," fundamentally reducing such errors.

3. **Knowledge boundary of the Value Model**: the Value Model can only guide stylistic objectives like "helpful, harmless," and cannot compensate for knowledge gaps in the base model (e.g., insufficient domain expertise). These issues need to be addressed at the base model training stage.

---

## 5. Conclusion

| Dimension | Fine-tune base model | Value Model + SIA |
|------|-------------------|-------------------|
| Training data volume | Large (needs comprehensive coverage) | **Small (high signal-to-noise preference data)** |
| Impact on general capabilities | Alignment Tax risk present | **Does not affect base model** |
| Target flexibility | Fixed, retraining required to change target | **Plug-and-play, hot-swappable** |
| Inference latency | Unchanged | Extra overhead (intervention steps ~119 ms) |
| Low-entropy errors | Correctable via training data | **Cannot correct** |
| Deployment complexity | Low (single model) | Requires maintaining two models |

**Core conclusion**: The fundamental advantage of the Value Model approach is "decoupling the alignment objective from model weights at extremely low training cost." It is not a complete replacement for fine-tuning, but is a better choice than fine-tuning in scenarios that require flexibility, low cost, and preservation of base capabilities.
