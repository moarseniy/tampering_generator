import torch
from torch.utils.data import DataLoader
from core import ForgeryDataset, get_train_transforms, get_mask_transforms

def setup_training_data(config_path: str, batch_size: int = 8):
    """
    Настройка данных для обучения
    """
    # Трансформации для обучения
    train_transform = get_train_transforms(image_size=(1024, 1024))
    mask_transform = get_mask_transforms(image_size=(1024, 1024))
    
    # Создаем датасет
    train_dataset = ForgeryDataset(
        config_path=config_path,
        transform=train_transform,
        target_transform=mask_transform,
        image_size=(1024, 1024),
        num_samples=50000,  # 50K samples на эпоху
        cache_size=200
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader

# Использование в цикле обучения
def train_model():
    # Настройка данных
    train_loader = setup_training_data("configs/generator_config.yaml", batch_size=8)
    
    # Модель, оптимизатор и т.д.
    model = YourSegmentationModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Цикл обучения
    for epoch in range(100):
        model.train()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.cuda()
            masks = masks.cuda()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

train_model()
