Code for a natural language processing task aimed to reproduce the character "Mona" from the video game Genshin Impact in PyTorch, drawing inspiration from [c.ai](https://beta.character.ai/chat?char=Txd-p7aN66ckzfdzpiJz9gLOUPwSe0c3fuPgsZ45gqA)

Original procedure was to follow the same steps that InstructGPT was trained on, which was to fine-tune the GPT-3 model to follow instructions and then use reinforcement learning from human feedback to train a policy for the model

Currently training the model to replicate GPT-2, using the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset

![trainloss-9 6](https://github.com/byronf01/mona_ai/assets/89189391/14b2c10a-baf2-4369-b016-c1dc3d4eed59)

Sources consulted:
 - Original transformer paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
 - Findings on GPT3: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
 - InstructGPT paper: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
 - Addition training details: [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
   
