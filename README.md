# Road to Deep Learning (Road2DL)

## Introduction

This repository references several well-known deep learning/large model repositories such as "Dive into Deep Learning" and "LLMs-from-scratch", and follows the design pattern of the transformers library, aimed at learning the underlying principles of deep learning/large models and common techniques like inference and training.

At the same time, this repository is also a well-organized Python project managed by uv, which can be used to quickly build environments and run with uv.

**Note:** This repository is still under development and may not be fully functional. Although the repository imitates the organization of the transformers library, it only implements the most basic functions, and its adaptability and compatibility are far inferior to the corresponding functions in transformers.

## Repository Structure

```
├── cookbook
├── docs
├── models
├── scripts
├── src/road2dl
│   ├── benchmark
│   ├── data
│   ├── evaluate
│   ├── models
│   ├── optimizer
│   ├── pipelines
│   ├── scratch
│   ├── tokenizer
│   ├── trainer.py
├── tests
├── ...
```

## Reference List

- [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en)
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [transformers](https://github.com/huggingface/transformers)

## TODO List

- Implement a cache class similar to the transformers library to accelerate inference for large language models.

- Implement more attention mechanisms such as GQA, MLA, SWA, etc.

- Implement more well-known transformer-based large language models such as Qwen3, Llama, Deepseek, etc.

- Implement the `GenerationMixin` class required for model inference, and implement the `generate` method.

- ...
