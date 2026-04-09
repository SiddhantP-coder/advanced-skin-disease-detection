import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import yaml
from tqdm import tqdm
from utils.data_loader import get_dataloaders
from models.simple_model import SkinDiseaseNet

print("🩺 Training Clinical Skin Disease AI...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

train_loader, val_loader = get_dataloaders()

model = SkinDiseaseNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

best_acc = 0
for epoch in range(20):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/20')
    
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        train_pbar.set_postfix({'Acc': f'{100.*train_correct/train_total:.1f}%'})
    
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100.*val_correct/val_total
    scheduler.step()
    
    print(f'Epoch {epoch+1}: Train: {100.*train_correct/train_total:.1f}% | Val: {val_acc:.1f}%')
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

print(f"🎉 Training complete! Best accuracy: {best_acc:.1f}%")
