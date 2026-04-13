
这里主要记录关于生成模型的tokenizer，也就是vae后续的迭代

[Semantic latent](../../../reading/interesting/Semantic%20latent.md)



End-to-End Training for Unified Tokenization and Latent Denoising
https://arxiv.org/abs/2603.22283
MIT & Adobe
▷ 背景
√ 目前latent diffusion models (LDMs)的主流框架都是先训练VAE这类tokenizer，然后固定tokenizer训练diffusion model
▷ 方法
√ 这篇文章中，作者提出了一个端到端的训练框架，UNITE，可以联合地训练tokenizer和diffusion model
◇ UNITE的一大创新就是把tokenizer的encoder部分和diffusion合二为一了
◇ 作者指出：tokenization和生成其实可以被看成相同的latent推断问题，只是所依赖的条件输入不同。在 tokenization中，原图片是一个很强的强约束，模型需要将图片映射成对应的latent z_0；而在生成过程中，带噪的latent，z_t，是一个比较弱的条件信号，模型需要基于这个噪声信号逐步还原出z_0
◇ 因此，作者认为，tokenization 和 generation这两个任务只是条件输入不同，换句话说，encoder和diffusion这两个模型可以合二为一，形成一个共享参数的共同的encoder
◇ 作者指出：共享参数使得tokenization 和 generation的梯度能够一起塑造latent space，从而促进形成一种“统一的latent language”
√ UNITIE包含两个主要模块，generative encoder和decoder。其中generative encoder有tokenization和generation两种功能
◇ Tokenization mode（encoder的功能）：给定图片，generative encoder将图片映射到latent。Decoder将基于该latent重建图片，并计算重建相关的loss
◇ Generation mode（diffusion的功能）：给定带噪声的latent和时间t，generative encoder需要能通过diffusion的形式，去噪生成干净的latent
◇ 训练时，作者会将刚才得到的图片latent加噪，和时间t一起送入generative encoder，并计算diffusion的loss
◇ 注意，这里为了避免diffusion的loss导致latent collapse，构造z_t时需要加一个stop gradient操作
▷ 实验
√ 在图像和分子等多种模态上，UNITE 在不使用adversarial loss或DINO等预训练encoder的情况下，取得了接近SoTA的性能
◇ 在 ImageNet 256 × 256 上，Base 和 Large 模型分别达到 2.12 和 1.73 的 FID