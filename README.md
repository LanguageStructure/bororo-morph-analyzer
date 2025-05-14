# Bororo Morphological Analyzer

This repository contains a slot-based morphological analyzer for the Bororo language, built using Conditional Random Fields (CRFs) trained on the Bororo Universal Dependencies Treebank.

## Web address

[Access here](https://boeenomoto.pythonanywhere.com/Morfologia)

## Features

- Per-feature CRF models (Voice, Reflexivity, Tense, etc.)
- Flask web interface for real-time analysis
- Full HTML/CSS-based UI for integration in educational tools
- Compatible with CONLLU-formatted corpora

## Usage

### Train the CRFs

```bash
python train_slot_crfs.py
```

## Citation

If you use this tool, please cite it:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15404849.svg)](https://doi.org/10.5281/zenodo.15404849)

```bibtex
@software{gerardi_bororo_2024,
  author       = {Fabrício Ferraz Gerardi},
  title        = {Bororo Morphological Analyzer},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
