# Watermarking Analysis on Abstractive Summarization

This project applies the LLM watermarking algorithm proposed by
Kirchenbauer et al. (2023) — ["A Watermark for Large Language Models"](https://arxiv.org/abs/2301.10226)
— to an abstractive summarization setting, examining how well the
algorithm performs on short, compressed outputs rather than the
long-form text generation it was originally designed for.

## Setup

- **Model:** facebook/bart-large-xsum
- **Dataset:** XSum (600 test examples)
- **Watermark code:** official implementation from
  [jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)
- **Seeding scheme:** simple_1 (selfhash was found incompatible with
  BART's encoder-decoder architecture)

## Analyses

- Z-score distribution (watermarked vs non-watermarked)
- Threshold sensitivity (detection rate vs false positive rate)
- Delta sensitivity (detection rate vs quality tradeoff)
- Length effect (summary length vs z-score)
- Quality impact (ROUGE, perplexity)

## Key findings

- The recommended `selfhash` seeding scheme is incompatible with
  encoder-decoder architectures like BART — it requires 4 tokens
  of decoder context that aren't available at the start of
  generation. The original `simple_1` scheme was used instead.
- At the default threshold (4.0) and delta (2.0), detection rate on
  XSum summaries is only ~9.8% — short summaries provide too few
  tokens for a strong watermark signal.
- Lowering the threshold to 2.0 recovers ~60.8% detection at a 3.2%
  false positive rate.
- Increasing delta to 5.0 achieves ~91.5% detection but reduces
  ROUGE-1 by ~29%.
- Applying LLM watermarking to abstractive summarization requires
  domain-specific recalibration of both threshold and delta.

## Files

- `watermarking_analysis.ipynb` — full analysis notebook, runnable in Colab