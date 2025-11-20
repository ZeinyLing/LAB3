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
LAB1/
│
├── preprocessing.py     # Image preprocessing (CLAHE, resize to 512×512)
├── train.py             # Training pipeline for classification models
├── inference.py         # Model inference on test dataset
├── voting.py            # Voting ensemble of multiple trained models
├── draw.py              # Draw curves from csvs
│
├── csvs/                # Training and validation logs (acc, F1 per epoch)
├── cm_plot/             # Confusion matrix heatmaps
├── plots/               # Accuracy and F1-score curves
└── pkls/                # Trained model weights (.pkl) -->　In Google Cloud
```
---
## Code tran.py
| Model | Architecture in timm | 
|:--|:--:|
| ResNet | `resnet18, resnet34, resnet50, resnet101, resnet152` |
| VGGNet | `vgg16, vgg19` |
| Vision Transformer | vit_base_patch16_224, vit_small_patch16_224, vit_large_patch16_224` |
---
## Code tran.py

```
# select model
model_select = 'vgg16'

# For vgg efficientnet densenet  resnet
model_ft = timm.create_model(model_select, pretrained=True)

# For 'vit_base_patch16_224'
#model_ft = timm.create_model(model_select, pretrained=True,img_size=512) 

# For resnet model
'''
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, n_class)
'''

# For not resnet model
in_features = model_ft.get_classifier().in_features
model_ft.reset_classifier(num_classes=n_class)
```
---
## Folder Descriptions

| Folder | Description |
|:--|:--|
| `csvs/` | Training and validation logs | 
| `cm_plot/` | Confusion matrix heatmaps | 
| `plots/` | Accuracy & F1-score curves |
| `pkls/` | Model weights (.pkl) | 
---
## Best single model (ResNet34)

✅ Final performance on test set：  
- **Accuracy:** 92.95%  
- **F1-score:** 0.945  
<img src="cm_plots/cm_5_resnet34_ep_20.pkl.png" width="450">

## Voting Ensemble 

✅ Final performance on test set：  
- **voting by ResNet34, ResNet50, ResNet18 , VGG16**
- **Accuracy:** 93.27%  
- **F1-score:** 0.926  
<img src="cm_plots/cm_voted.png" width="450">
