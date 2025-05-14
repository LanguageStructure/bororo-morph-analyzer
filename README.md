# Bororo Morphological Analyzer

This repository contains a slot-based morphological analyzer for the Bororo language, built using Conditional Random Fields (CRFs) trained on Universal Dependencies-style annotated corpora.

## Features

- Per-feature CRF models (Voice, Reflexivity, Tense, etc.)
- Flask web interface for real-time analysis
- Full HTML/CSS-based UI for integration in educational tools
- Compatible with CONLLU-formatted corpora

## Usage

### Train the CRFs

```bash
python train_slot_crfs.py


## Citation

If you use this tool, please cite it using the Zenodo DOI below (to be assigned):

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxxx)
