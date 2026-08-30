# CS 189/289A - Discussion 12

## Fall 2025

---

> **Note:** Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

## This week's cool AI demo/video

- [Gemini 3 video chat](https://play.google.com/store/apps/details?id=com.google.android.apps.bard&hl=en_US)
- [Google's Cursor clone just launched](https://antigravity.google/download)
- [Sunday came out of stealth](https://www.youtube.com/watch?v=jjOfpsMRhL4)

---

## 1. Supervised Fine-tuning for Vision Language Models

A vision language model (VLM) is a model that integrates visual and textual data, enabling it to understand and process both images and text. By combining a vision encoder and a large language model (LLM), VLMs can perform tasks like generating captions for images, answering questions about an image, and even creating images from text descriptions.

To integrate a Vision Encoder that processes an image and outputs patch embeddings of dimension $D_{\text{vis}}$ and an LLM that expects token embeddings of dimension $D_{\text{txt}}$, we typically define a Projection Layer ($\mathbf{W}_P$) that projects visual embeddings into the text embedding space. Supervised Fine-tuning (SFT) is then performed with a small image-text dataset to align the visual features with the language model's embedding space.

### (a) Projection Dimensions

Suppose the Vision Encoder outputs an embedding $\mathbf{H}_{\text{vis}}$ of size $D_{\text{vis}}=1024$ and the LLM expects embeddings of size $D_{\text{txt}}=4096$. What must be the dimensions of the weight matrix $\mathbf{W}_P$ to project the visual features to the text space if we left-multiply the image embedding by the projection matrix?

**Answer (a):**

Using column-vector notation, the projection is

$$
\mathbf{h}_{\text{txt}}=\mathbf{W}_P\mathbf{h}_{\text{vis}}.
$$

Since $\mathbf{h}_{\text{vis}}\in\mathbb{R}^{1024}$ and the result must lie in $\mathbb{R}^{4096}$, the projection matrix must have dimensions

$$
\boxed{\mathbf{W}_P\in\mathbb{R}^{4096\times1024}}.
$$

### (b) Freezing Strategies

During the initial alignment stage (and often during SFT), it is standard practice to keep the Vision Encoder and the LLM frozen, updating only the projection layer. Give two distinct reasons why we freeze the backbone models.

**Answer (b):**

1. Freezing preserves pretrained capabilities. The vision encoder and LLM were trained on much larger datasets than the alignment dataset. Updating all of their weights on a small image-text dataset can cause catastrophic forgetting and weaken their original visual or language representations.

2. Freezing reduces training cost. Backpropagating through two large backbone models requires substantial memory and computation. Freezing them avoids storing most parameter gradients and optimizer states, so training can focus on the much smaller projection layer.

### (c) Image Token Granularity

The classify token (`[CLS]`) is a special token that represents the entire input sequence for a classification task. We can choose to project only the Vision Encoder's classification token into the LLM embedding space, or we can project the entire sequence of patch tokens. What is the primary trade-off for each of these two approaches?

**Answer (c):**

- Projecting only `[CLS]` adds one visual token to the LLM context, making training and inference cheaper. The single global summary discards much of the image's fine-grained spatial information, so detailed descriptions and grounding are harder.
- Projecting all patch tokens retains local and spatial information and supports more detailed visual reasoning. It may add hundreds of tokens per image, consuming context-window capacity and increasing attention cost and latency.

### (d) The Role of SFT Data

Explain why we cannot simply rely on the pre-training of the individual components and why the SFT stage (using image-text pairs) is necessary even if the embedding dimensions are the same.

**Answer (d):**

Matching dimensions does not make the two representation spaces semantically compatible. A visual feature produced for a cat, for example, is not automatically located where the LLM expects the textual concept "cat" to appear. The untrained projector could therefore map image features to vectors that the LLM interprets as meaningless inputs.

Image-text SFT provides a learning signal that trains the projector to translate visual features into representations the LLM can use for language generation. It aligns the modalities and teaches the combined model how visual evidence should affect its output.

---

## 2. Self-Supervised Learning

### (a)

Fill in the blanks in the following table.

| Method Name | Input Type | Pretext Task | Generative or Discriminative | Loss Function |
| --- | --- | --- | --- | --- |
| Autoencoder | Image | Reconstruct the input image. |  |  |
| Context Encoder | Masked Image | Predict the missing content in the masked region of the input. |  |  |
| Image Rotation | Rotated Image | Predict the rotation angle applied to the input ($0^\circ$, $90^\circ$, $180^\circ$, or $270^\circ$). |  |  |
| SimCLR (Contrastive Learning) | Two Images | Determine whether images are augmentations of the same image or different images. |  |  |

**Answer (a):**

| Method Name | Input Type | Pretext Task | Generative or Discriminative | Loss Function |
| --- | --- | --- | --- | --- |
| Autoencoder | Image | Reconstruct the input image. | Generative | Mean squared error (MSE) |
| Context Encoder | Masked Image | Predict the missing content in the masked region of the input. | Generative | MSE plus adversarial loss |
| Image Rotation | Rotated Image | Predict the applied rotation angle. | Discriminative | Cross-entropy loss |
| SimCLR (Contrastive Learning) | Two augmented views | Pull views of the same image together and push views of different images apart. | Discriminative | Contrastive loss, typically NT-Xent/InfoNCE |

### (b)

When training context encoders, we often use a joint loss function

$$
\mathcal{L}=\mathcal{L}_{\text{rec}}+\lambda\mathcal{L}_{\text{adv}},
$$

where:

- $\mathcal{L}_{\text{rec}}$ is the reconstruction loss (e.g., MSE) applied to the masked region.
- $\mathcal{L}_{\text{adv}}$ is the adversarial loss from a discriminator.
- $\lambda$ is a weighting factor that balances the two terms.

#### (i)

If we trained using only $\mathcal{L}_{\text{rec}}$, what artifact might we observe in the generated missing region? Why?

**Answer (b)(i):**

The reconstructed region will often look blurry or overly smooth. With several plausible completions for the same context, minimizing pixelwise MSE favors the conditional mean of those possibilities. Averaging different textures, edges, or object placements removes high-frequency detail and produces blur.

#### (ii)

Why does adding the adversarial loss $\mathcal{L}_{\text{adv}}$ help fix this issue?

**Answer (b)(ii):**

The discriminator rewards completions that resemble real image regions, including realistic edges and textures. This pressure discourages the generator from returning the pixelwise average of several possibilities and instead favors a sharp, plausible completion. The reconstruction term preserves agreement with the surrounding image, while the adversarial term improves perceptual realism.

---

## 3. Autoregressive vs. Diffusion Models

Autoregressive (AR) models, such as next-word prediction GPT models, and diffusion models are two popular categories of deep generative models. They both achieve high-quality generation by decomposing the process into many steps. In this problem, we explore the design choices that lead to their training efficiency and their distinct generation mechanisms.

### (a) Computational Graph and Training Efficiency

Explain the core design principle that allows AR and diffusion models to be training-efficient despite performing a deep, sequential computation at inference time.

**Answer (a):**

Both model families train with local objectives that avoid backpropagating through the full inference-time generation chain.

For an autoregressive Transformer, teacher forcing supplies the ground-truth prefix at every position. A causal mask lets the model compute next-token losses for all sequence positions in parallel during one forward pass, even though inference must generate tokens one at a time. For a diffusion model, training samples a timestep $t$, adds the corresponding amount of noise directly to a clean example, and trains the network to predict the noise or denoising target at that timestep. It does not run or differentiate through every reverse-diffusion step.

The long sequential chain is therefore required during generation, when gradients are unnecessary, but not during each training update.

### (b) Frequency/Scale Generation in Diffusion Models

Describe the typical pattern of frequency content (or scale of details) that is synthesized as the generative (denoising) process moves from high noise ($t=T$) toward the clean image ($t=0$).

**Answer (b):**

Diffusion generation usually proceeds from coarse structure to fine detail:

- Near $t=T$, heavy noise obscures small details. The denoiser first establishes low-frequency content such as the global layout, large shapes, and broad color regions.
- As $t$ approaches $0$, the noise level falls and the model adds higher-frequency content such as boundaries, textures, and small features.

Thus the main composition appears early, while later denoising steps refine the image without substantially changing its global structure.
