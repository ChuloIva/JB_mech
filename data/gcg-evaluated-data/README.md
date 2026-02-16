---
license: mit

configs:
- config_name: default
  data_files:
  - split: gemma2
    path: "gemma-2-2b-it_eval_data.parquet"
  - split: qwen2.5
    path: "qwen2.5-1.5b-instruct_eval_data.parquet"
  - split: llama3.1
    path: "llama-3.1-8b-instruct_eval_data.parquet"

---

GCG suffixes crafted on [Gemma-2](https://huggingface.co/google/gemma-2-2b-it), [Qwen-2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [Llama-3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), their generated response when appended to harmful instructions (from [AdvBench](https://github.com/llm-attacks/llm-attacks), [StrongReject's custom](https://strong-reject.readthedocs.io/en/latest/)), their evaluation and charecterization.

This dataset was created and utilized in the paper: *Universal Jailbreak Suffixes Are Strong Attention Hijackers* ([paper](https://arxiv.org/abs/2506.12880), [code](https://github.com/matanbt/interp-jailbreak/tree/main)).


**WARNING:** this dataset contains harmful content, and is intended for research purposes only.

## Each row in the dataset describes:
- **Harmful instruction info:**
    - `message_id`: instruction's unique identifier.
    - `message_str`: insturction's string.
    - `message_source`: instruction's source dataset (from AdvBench, or StrongReject's custom subset).
- **Advesarial suffix info:**
    - `suffix_id`: suffix's  unique identifier.
    - `suffix_str`: suffix's string (mostly ~20 tokens long).
    - `message_suffixed`: message concatenated with the suffix.
- **Response & its evaluation:**
    - `response`: response's string.
    - `strongreject_finetuned`: jailbreak score by [StrongReject's grader](https://strong-reject.readthedocs.io/en/latest/) (the higher the more harmful).
    - `response_first_tok`: response's first token.
    - `response_category`: response's category (according to its success and characterization of the response's prefix).
- **Response under prefilling & its evaluation:**
    - `prefilled__response`: response's string (under prefilling).
    - `prefilled__strongreject_finetuned`: jailbreak score (under prefilling).
- **Suffix-specific evaluation:**
    - `univ_score`: universality score of the suffix (i.e., its average jailbreak score across all the instructions).
    - `suffix_rank`: suffix's rank according to its universality score.
- **Suffix's optimization info:**
    - `suffix_objective`: objective used for optimizing the suffix (e.g., GCG's objective is `affirm`).
    - `suffix_optimizer`: optimizer used (e.g., `gcg`).
    - `suffix_obj_indices`: the message ids used for optimizing the suffix.
    - `suffix_category`: suffix category (`intrdm` are taken from an intermediate step of the optimziation, while `reg` are the final suffixes; `init` means a suffix used for initalization).
    - `is_mult_attack`: whether the suffix was optimized on multiple instructions.
    - `is_trained_message`: whether the suffix was optimized on the paired instruction.

## Citation
If you find this work, or any of the research artifacts, useful, please cite our paper as:
```
@article{bentov2025universaljailbreaksuffixesstrong,
      title={{U}niversal {J}ailbreak {S}uffixes {A}re {S}trong {A}ttention {H}ijackers}, 
      author={Matan Ben-Tov and Mor Geva and Mahmood Sharif},
      year={2025},
      eprint={2506.12880},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2506.12880}, 
}
```