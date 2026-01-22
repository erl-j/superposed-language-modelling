# SLM: Superposed Language Modelling

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erl-j/superposed-language-modelling/blob/main/examples/tutorial.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2408.02434-b31b1b.svg)](https://arxiv.org/abs/2408.02434)


## Abstract

With the goal of building a system capable of controllable symbolic music loop generation and editing, this paper explores a generalisation of Masked Language Modelling we call Superposed Language Modelling. Rather than input tokens being known or unknown, a Superposed Language Model takes priors over the sequence as input, enabling us to apply various constraints to the generation at inference time. After detailing our approach, we demonstrate our model across various editing tasks in the domain of multi-instrument MIDI loops. We end by highlighting some limitations of the approach and avenues for future work. We provides examples from the SLM across multiple generation and editing tasks.

## Demo

**[Interactive Application Demo](https://www.youtube.com/watch?v=etuF94r-3hM)**  
Watch a demonstration of the interactive application based on Superposed Language Modelling for symbolic music loop editing.

## Paper

**[Steer-by-prior Editing of Symbolic Music Loops](https://arxiv.org/pdf/2408.02434)**  
Presented at MML 2024: 15th International Workshop on Machine Learning and Music

### Citation

```bibtex
@inproceedings{jonason2024steer,
  title={Steer-by-prior Editing of Symbolic Music Loops},
  author={Jonason, Nicolas and Casini, Luca and Sturm, Bob L. T.},
  booktitle={15th International Workshop on Machine Learning and Music (MML 2024)},
  year={2024},
  doi={10.48550/arXiv.2408.02434}
}
```


## Local experimentation (GPU recommended)

First install the package.
```bash
pip install -e .
```

Then open the tutorial.
```
examples/tutorial.ipynb
```