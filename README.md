# Beyond Referring Expressions: Scenario Comprehension Visual Grounding
> Authors: Ruozhen He, Nisarg A. Shah, Qihua Dong, Zilin Xiao, Jaywon Koo, Vicente Ordonez

[\[Paper\]](https://arxiv.org/abs/2604.02323) [\[Project Page\]](https://catherine-r-he.github.io/RSC/)

### Requirements
Training uses two environments, because LLaMA-Factory and TRL need different `transformers`/`trl` versions. Evaluation only needs the GRPO environment.
```Shell
# GRPO + evaluation
conda create -n rsc-grpo python=3.10 && conda activate rsc-grpo
pip install -r grpo/requirements.txt

# SFT
conda create -n rsc-sft python=3.10 && conda activate rsc-sft
git clone https://github.com/hiyouga/LLaMA-Factory.git && pip install -e LLaMA-Factory
pip install -r sft/requirements.txt
```

### Data
- RSC annotations are included: `data/rsc_id_test.jsonl` (in-domain, 4,038, COCO vocabulary), `data/rsc_ood_test.jsonl` (out-of-domain, 3,247, LVIS vocabulary), and the SFT and GRPO training files in `data/train/`. See [`data/README.md`](data/README.md) for the format.
- MS-COCO [\[Images\]](https://cocodataset.org/#download). To download only the images RSC uses, run:
```Shell
python data/download_images.py --image-root $COCO_ROOT               # test splits (~700 MB)
python data/download_images.py --image-root $COCO_ROOT --with-train  # + training images (~5.5 GB)
```

### Train
Start from [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and train in three stages, each initialized from the previous checkpoint. See [`sft/README.md`](sft/README.md) and [`grpo/README.md`](grpo/README.md) for details.
```Shell
# SFT (set dataset_dir and media_dir in the yaml, then run from the LLaMA-Factory directory)
llamafactory-cli train /path/to/RSC/sft/qwen25vl_7b_sft.yaml

# GRPO data, stage 1, stage 2 (run from grpo/)
COCO_ROOT=/path/to/COCO scripts/prepare_data.sh
INIT_CHECKPOINT=/path/to/saves/qwen25vl-7b-rsc-sft/checkpoint-2340 scripts/train_stage1.sh
INIT_CHECKPOINT=/path/to/qwen25vl-grpo-bbox-stage1/checkpoint-400 scripts/train_stage2.sh
```

### Evaluation
To evaluate on RSC-ID and RSC-OOD, run the commands below. `MODEL` can be a local checkpoint or a Hugging Face model ID, and results are saved to `eval/results/`. See [`eval/README.md`](eval/README.md) for more options.
```Shell
MODEL=/path/to/checkpoint COCO_ROOT=/path/to/COCO eval/scripts/eval_id.sh
MODEL=/path/to/checkpoint COCO_ROOT=/path/to/COCO eval/scripts/eval_ood.sh
```

### BibTex
```Shell
@article{he2026beyond,
  title={Beyond Referring Expressions: Scenario Comprehension Visual Grounding},
  author={He, Ruozhen and Shah, Nisarg A and Dong, Qihua and Xiao, Zilin and Koo, Jaywon and Ordonez, Vicente},
  journal={arXiv preprint arXiv:2604.02323},
  year={2026}
}
```
