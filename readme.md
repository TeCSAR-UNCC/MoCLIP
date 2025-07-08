# MoCLIP: Motion-Aware Fine-Tuning and Distillation of CLIP for Human Motion Generation  
**(CVPR Workshop on Human Motion Generation 2025)**

📄 [Read the paper on arXiv](https://arxiv.org/abs/2505.10810)

MoCLIP enhances the CLIP encoder for human motion generation by integrating motion-aware features and contrastive alignment strategies. It offers immediate plug-and-play improvement to CLIP-based motion generation pipelines while preserving semantic richness.

---

## 🔧 Setup

### 1. Conda Environment

conda env create -f environment.yml  
conda activate momask  
pip install git+https://github.com/openai/CLIP.git

Tested with Python 3.7.13 and PyTorch 1.7.1.

### 2. Models and Dependencies

**Download Pre-trained Models**

[Todo] - Add download script or link here

**Download Evaluation Models and GloVe**

bash prepare/download_evaluator.sh  
bash prepare/download_glove.sh

### 3. Get Data

You can skip this step if you're only generating motion from your own descriptions.

#### Full Data (Text + Motion)

- **HumanML3D**  
  Follow the instructions from the original HumanML3D repo: https://github.com/EricGuo5513/temos  
  Then copy the result into this repo:

  cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D

- **KIT-ML**  
  Also available from the HumanML3D setup. Place it in:

  ./dataset/KIT-ML

Note: These datasets are maintained by their original authors. Please refer to their repositories for licensing and setup requirements.

---

## 🏋️‍♂️ Training Your Own Model

To train MoCLIP on your machine:

python train_moclip.py --lda 0.4 --dataset_name t2m --gpu_id 0

### 💡 Lambda (`--lda`)

This argument controls the tethering loss, which helps preserve CLIP’s original semantics while adapting to motion tasks.

- Best performing value on HumanML3D: 0.4  
- You may experiment with values like 0.2, 0.4, and 0.6 for different models or datasets.

#### Training Arguments

| Argument         | Description                                  | Default |
|------------------|----------------------------------------------|---------|
| --lda            | Distillation loss weight (lambda)            | 0.6     |
| --dataset_name   | Name of the dataset ('t2m', 'kit', etc.)     | t2m     |
| --gpu_id         | ID of GPU to use                             | 0       |

---

## 🔌 Integration

MoCLIP integrates easily into CLIP-based text-to-motion models like MoMask, BAD, and BAMM.  
See the `examples/` folder for working demos.

### 🧩 Four-Step Integration

1. Create motion encoder  
res_transformer.clip_model.motion = MotionTransformerv1(opt.dataset_name).to(opt.device)

2. Add motion encoder function  
res_transformer.clip_model.encode_motion = encode_motion.__get__(res_transformer.clip_model, type(res_transformer.clip_model))

3. Load MoCLIP weights  
kpt = torch.load(opt.motion_clip, map_location=opt.device)

4. Replace CLIP weights with MoCLIP  
res_transformer.clip_model.load_state_dict(kpt, strict=False)

---

## 📁 Repository Structure

- `train_moclip.py` – Script for training MoCLIP  
- `examples/` – Integration code for BAD, BAMM, and MoMask
- `dataset/` – Expected folder for HumanML3D and KIT-ML datasets

---

## 📣 Citation

@article{maldonado2025moclip,  
  title={MoCLIP: Motion-Aware Fine-Tuning and Distillation of CLIP for Human Motion Generation},  
  author={Maldonado, Gabriel and Danesh Pazho, Armin and Alinezhad Noghre, Ghazal and Katariya, Vinit and Tabkhi, Hamed},  
  journal={arXiv preprint arXiv:2505.10810},  
  year={2025}  
}
