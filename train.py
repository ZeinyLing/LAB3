import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, confusion_matrix
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# === 載入模型並讀取權重 ===
def load_model(pth_path, num_classes=4, use_gpu=True):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    state_dict = torch.load(pth_path, map_location="cuda" if use_gpu else "cpu")
    model.load_state_dict(state_dict)
    
    if use_gpu:
        model = model.cuda()
    model.eval()
    print(f"✅ Model loaded from {pth_path}")
    return model

# === 評估函式 ===
def evaluate_model(model, dataloader, save_path="test_predictions.csv", use_gpu=True):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    all_records = []

    with torch.no_grad():
        for inputs, labels, *extra in dataloader:
            filenames = extra[0] if len(extra) > 0 else [f"sample_{i}" for i in range(len(labels))]

            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            probs_np = probs.cpu().numpy()
            y_prob.extend(probs_np[:, 1] if probs_np.shape[1] > 1 else probs_np[:, 0])

            for fname, label, pred, prob_vec in zip(filenames, labels.cpu().numpy(), preds.cpu().numpy(), probs_np):
                record = {
                    "filename": fname,
                    "true_label": int(label),
                    "pred_label": int(pred),
                }
                for c in range(prob_vec.shape[0]):
                    record[f"prob_class_{c}"] = prob_vec[c]
                all_records.append(record)

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    acc = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    except:
        auroc, auprc = np.nan, np.nan
    cm = confusion_matrix(y_true, y_pred)

    df = pd.DataFrame(all_records)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved prediction probabilities to {save_path}")

    return acc, f1, auroc, auprc, cm

# === 測試資料夾 DataLoader（帶檔名） ===
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        filename = os.path.basename(path)
        return img, label, filename

# === 主程式 ===
if __name__ == "__main__":
    # 路徑設定
    test_dir = "./draw_images/fold4/test"
    model_path = "./resnet50_fold4_best1.pth"
    save_csv = "./test_predictions2.csv"
    use_gpu = torch.cuda.is_available()

    # 測試資料轉換
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = ImageFolderWithPaths(test_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 載入模型與評估
    model = load_model(model_path, num_classes=len(test_dataset.classes), use_gpu=use_gpu)
    acc, f1, auroc, auprc, cm = evaluate_model(model, test_loader, save_path=save_csv, use_gpu=use_gpu)

    print("Accuracy:", acc)
    print("Macro-F1:", f1)
    print("AUROC:", auroc)
    print("AUPRC:", auprc)
    print("Confusion Matrix:\n", cm)
