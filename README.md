# LAB3
Lab3 CXR multi-class classification

## Datasets
| Dataset | Normal | Bacterial | Virus | COVID-19 |
|:--|:--:|:--:|:--:|:--:|
| Train | **1072** | **1888** | **1018** | **39** |
| Val | **189** | **333** | **180** | **7** |
| Test | **315** | **556** | **299** | **12** |
---

## 📁 Project Structure
```
LAB3/
│
├── preprocessing.py     # Image preprocessing
├── train.py             # Training pipeline for classification models
└── voting.py            # Voting ensemble of multiple trained models
```
---
## Models in timm
| Model | Architecture in timm | 
|:--|:--:|
| ResNet | `resnet18`,`resnet34`, `resnet50`,`resnet101`,`resnet152` |
| VGGNet | `vgg16`, `vgg19` |
| Vision Transformer | `vit_base_patch16_224`, `vit_small_patch16_224`, `vit_large_patch16_224` |
---
## Code train.py

```
# select model
model_select = '#model#'

model_ft = timm.create_model(model_select, pretrained=True)
in_features = model_ft.get_classifier().in_features
model_ft.reset_classifier(num_classes=n_class)
```
---

