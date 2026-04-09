import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def create_quick_test_dataset():
    print("🎨 Creating 400 realistic skin lesion images...")
    os.makedirs('data/quick_test/train/images', exist_ok=True)
    os.makedirs('data/quick_test/val/images', exist_ok=True)
    
    np.random.seed(42)
    n_samples = 400
    image_ids = [f"img_{i:04d}" for i in range(n_samples)]
    labels = np.random.choice(7, n_samples, p=[0.60, 0.10, 0.10, 0.05, 0.05, 0.08, 0.02])
    
    class_names = ['nv', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    dx_labels = [class_names[int(label)] for label in labels]
    
    df = pd.DataFrame({'image_id': image_ids, 'label': labels.tolist(), 'dx': dx_labels})
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    colors = [[240,200,180],[200,150,100],[180,140,120],[220,190,160],[100,50,30],[240,200,180],[150,200,220]]
    
    for split, split_df in [('train', train_df), ('val', val_df)]:
        for _, row in split_df.iterrows():
            img = np.full((224, 224, 3), [240, 200, 180], dtype=np.uint8)
            lesion_color = colors[int(row['label'])]
            cv2.ellipse(img, (112, 112), (60, 40), 0, 0, 360, lesion_color, -1)
            cv2.ellipse(img, (112, 112), (70, 50), 15, 0, 360, [0,0,0], 2)
            noise = np.random.normal(0, 8, img.shape).astype(np.uint8)
            img = np.clip(cv2.add(img, noise), 0, 255)
            cv2.imwrite(f"data/quick_test/{split}/images/{row['image_id']}.jpg", 
                       cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        split_df[['image_id', 'label', 'dx']].to_csv(f"data/quick_test/{split}/labels.csv", index=False)
    
    print("✅ Dataset ready! Train:", len(train_df), "Val:", len(val_df))

if __name__ == "__main__":
    create_quick_test_dataset()
