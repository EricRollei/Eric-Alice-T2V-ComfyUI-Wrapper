# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-XX-XX

### Added
- Initial release
- `Eric_AliceLoader` node - loads and caches the Alice T2V 14B MoE pipeline
- `Eric_AliceT2V` node - generates video frames from text prompts
- Vendored alice package approach (no dependency conflicts)
- Support for offload_model and t5_cpu memory optimization flags
- UniPC and DPM++ solver support
- 4-step fast inference via score-regularized consistency distillation

### Notes
- Requires ~28GB VRAM with offload_model=True, t5_cpu=True
- Model weights: [gomirageai/Alice-T2V-14B-MoE](https://huggingface.co/gomirageai/Alice-T2V-14B-MoE)
