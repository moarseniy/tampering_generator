import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from pathlib import Path
from datetime import datetime

# Наши модули
from core import ForgeryDataset, get_train_transforms, get_mask_transforms
from models import DocumentForgerySegmentor
from losses import ForgerySegmentationLoss

class ForgeryDetectionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.writer = SummaryWriter(log_dir=config['training']['log_dir'])
        
        # Создаем директории
        self._create_directories()
        
        # Инициализация модели и данных
        self.model = self._init_model()
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.train_loader, self.val_loader = self._init_data_loaders()
        
        # Трекеры обучения
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
    def _create_directories(self):
        """Создание необходимых директорий"""
        Path(self.config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    def _init_model(self):
        """Инициализация модели"""
        model = DocumentForgerySegmentor(
            num_classes=self.config['model']['num_classes'],
            pretrained=self.config['model']['pretrained']
        )
        
        # Multi-GPU поддержка
        if torch.cuda.device_count() > 1:
            print(f"🚀 Используется {torch.cuda.device_count()} GPU")
            model = nn.DataParallel(model)
        
        model = model.to(self.device)
        
        # Загрузка чекпоинта если указан
        if self.config['model'].get('checkpoint_path'):
            self._load_checkpoint(model, self.config['model']['checkpoint_path'])
        
        return model
    
    def _init_criterion(self):
        """Инициализация функции потерь"""
        return ForgerySegmentationLoss(
            alpha=self.config['loss']['alpha'],
            beta=self.config['loss']['beta'],
            gamma=self.config['loss']['gamma']
        )
    
    def _init_optimizer(self):
        """Инициализация оптимизатора"""
        if self.config['optimizer']['name'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )
        elif self.config['optimizer']['name'] == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['optimizer']['lr']
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['optimizer']['lr'],
                momentum=0.9
            )
    
    def _init_scheduler(self):
        """Инициализация шедулера"""
        if self.config['scheduler']['name'] == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['scheduler']['T_0'],
                T_mult=self.config['scheduler']['T_mult']
            )
        elif self.config['scheduler']['name'] == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config['scheduler']['patience'],
                factor=0.5
            )
        else:
            return None
    
    def _init_data_loaders(self):
        """Инициализация данных"""
        # Трансформации
        train_transform = get_train_transforms(
            image_size=tuple(self.config['data']['image_size'])
        )
        mask_transform = get_mask_transforms(
            image_size=tuple(self.config['data']['image_size'])
        )
        
        # Train dataset
        train_dataset = ForgeryDataset(
            config_path=self.config['data']['config_path'],
            transform=train_transform,
            target_transform=mask_transform,
            image_size=tuple(self.config['data']['image_size']),
            num_samples=self.config['data']['num_train_samples']
        )
        
        # Validation dataset (меньше samples, без аугментаций)
        val_transform = get_mask_transforms(
            image_size=tuple(self.config['data']['image_size'])
        )
        
        val_dataset = ForgeryDataset(
            config_path=self.config['data']['config_path'],
            transform=val_transform,
            target_transform=val_transform,
            image_size=tuple(self.config['data']['image_size']),
            num_samples=self.config['data']['num_val_samples']
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        print(f"✅ Данные инициализированы:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def _load_checkpoint(self, model, checkpoint_path):
        """Загрузка чекпоинта"""
        if os.path.exists(checkpoint_path):
            print(f"🔄 Загружаем чекпоинт: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'epoch' in checkpoint:
                self.current_epoch = checkpoint['epoch']
            
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            
            print(f"✅ Чекпоинт загружен (epoch {self.current_epoch})")
    
    def train_epoch(self):
        """Одна эпоха обучения"""
        self.model.train()
        epoch_loss = 0
        epoch_seg_loss = 0
        epoch_boundary_loss = 0
        epoch_aux_loss = 0
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss_dict = self.criterion(outputs, masks)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_seg_loss += loss_dict['segmentation'].item()
            epoch_boundary_loss += loss_dict.get('boundary', 0).item()
            epoch_aux_loss += loss_dict.get('auxiliary', 0).item()
            
            # Логирование каждые N батчей
            if batch_idx % self.config['training']['log_interval'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch: {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] | '
                      f'Loss: {loss.item():.4f} | LR: {current_lr:.2e}')
        
        # Средние losses за эпоху
        num_batches = len(self.train_loader)
        avg_loss = epoch_loss / num_batches
        avg_seg_loss = epoch_seg_loss / num_batches
        avg_boundary_loss = epoch_boundary_loss / num_batches
        avg_aux_loss = epoch_aux_loss / num_batches
        
        return {
            'total_loss': avg_loss,
            'seg_loss': avg_seg_loss,
            'boundary_loss': avg_boundary_loss,
            'aux_loss': avg_aux_loss
        }
    
    def validate_epoch(self):
        """Валидация после эпохи"""
        self.model.eval()
        val_loss = 0
        val_seg_loss = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, masks)
                
                val_loss += loss_dict['total'].item()
                val_seg_loss += loss_dict['segmentation'].item()
        
        num_batches = len(self.val_loader)
        avg_val_loss = val_loss / num_batches
        avg_val_seg_loss = val_seg_loss / num_batches
        
        return {
            'val_loss': avg_val_loss,
            'val_seg_loss': avg_val_seg_loss
        }
    
    def save_checkpoint(self, is_best=False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) 
                              else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Регулярное сохранение
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            f'checkpoint_epoch_{self.current_epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Сохранение лучшей модели
        if is_best:
            best_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"🏆 Новая лучшая модель сохранена: {best_path}")
    
    def log_metrics(self, train_metrics, val_metrics):
        """Логирование метрик в TensorBoard"""
        # Train metrics
        self.writer.add_scalar('Loss/train', train_metrics['total_loss'], self.current_epoch)
        self.writer.add_scalar('Loss/train_seg', train_metrics['seg_loss'], self.current_epoch)
        self.writer.add_scalar('Loss/train_boundary', train_metrics['boundary_loss'], self.current_epoch)
        self.writer.add_scalar('Loss/train_aux', train_metrics['aux_loss'], self.current_epoch)
        
        # Val metrics
        self.writer.add_scalar('Loss/val', val_metrics['val_loss'], self.current_epoch)
        self.writer.add_scalar('Loss/val_seg', val_metrics['val_seg_loss'], self.current_epoch)
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('LR', current_lr, self.current_epoch)
    
    def train(self):
        """Основной цикл обучения"""
        print("🚀 Начало обучения модели детекции подделок!")
        print(f"📊 Конфигурация: {self.config['training']}")
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            print(f"\n📍 Эпоха {epoch + 1}/{self.config['training']['epochs']}")
            print("-" * 50)
            
            # Обучение
            epoch_start_time = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - epoch_start_time
            
            # Валидация
            val_start_time = time.time()
            val_metrics = self.validate_epoch()
            val_time = time.time() - val_start_time
            
            # Обновление learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Логирование
            self.log_metrics(train_metrics, val_metrics)
            
            # Вывод статистики
            print(f"✅ Train Loss: {train_metrics['total_loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Time: {train_time + val_time:.1f}s")
            
            print(f"📊 Детали: Seg: {train_metrics['seg_loss']:.4f} | "
                  f"Boundary: {train_metrics['boundary_loss']:.4f} | "
                  f"Aux: {train_metrics['aux_loss']:.4f}")
            
            # Сохранение чекпоинта
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                print(f"🎉 Новый лучший результат: {self.best_val_loss:.4f}")
            
            # Сохраняем каждые N эпох или если это лучшая модель
            if (epoch + 1) % self.config['training']['save_interval'] == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Ранняя остановка
            if self.config['training'].get('early_stopping'):
                if epoch - self.best_epoch > self.config['training']['early_stopping_patience']:
                    print(f"🛑 Ранняя остановка на эпохе {epoch}")
                    break
        
        print(f"\n🎉 Обучение завершено! Лучшая val loss: {self.best_val_loss:.4f}")
        self.writer.close()

def get_training_config():
    """Конфигурация обучения"""
    return {
        'model': {
            'num_classes': 1,
            'pretrained': True,
            'checkpoint_path': None  # путь к чекпоинту для продолжения обучения
        },
        'data': {
            'config_path': 'configs/generator_config.yaml',
            'image_size': [1024, 1024],
            'batch_size': 8,
            'num_train_samples': 50000,
            'num_val_samples': 5000,
            'num_workers': 4,
            'cache_size': 200
        },
        'loss': {
            'alpha': 0.7,
            'beta': 0.3,
            'gamma': 2.0
        },
        'optimizer': {
            'name': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 1e-4
        },
        'scheduler': {
            'name': 'CosineAnnealingWarmRestarts',
            'T_0': 10,
            'T_mult': 2
        },
        'training': {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'epochs': 100,
            'log_interval': 50,
            'save_interval': 5,
            'grad_clip': 1.0,
            'early_stopping': True,
            'early_stopping_patience': 20,
            'checkpoint_dir': './checkpoints',
            'log_dir': f'./logs/forgery_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    }

if __name__ == "__main__":
    # Загружаем конфигурацию
    config = get_training_config()
    
    # Инициализируем тренер
    trainer = ForgeryDetectionTrainer(config)
    
    # Запускаем обучение
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем")
        # Сохраняем чекпоинт при прерывании
        trainer.save_checkpoint()
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        raise