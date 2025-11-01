import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from model import DocumentForgerySegmentor
from core import get_val_transforms

class ForgeryDetector:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device)
        self.model = DocumentForgerySegmentor(num_classes=1, pretrained=False)
        
        # Загрузка весов
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_val_transforms(image_size=(1024, 1024))
    
    def predict(self, image_path, confidence_threshold=0.5):
        """Предсказание для одного изображения"""
        # Загрузка изображения
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Применение трансформаций
        transformed = self.transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.sigmoid(output['segmentation'])
        
        # Постобработка
        prediction = prediction.squeeze().cpu().numpy()
        prediction = cv2.resize(prediction, (original_shape[1], original_shape[0]))
        
        # Бинаризация
        binary_mask = (prediction > confidence_threshold).astype(np.uint8) * 255
        
        return {
            'probability_map': prediction,
            'binary_mask': binary_mask,
            'confidence': np.max(prediction)
        }
    
    def visualize_prediction(self, image_path, output_path=None, confidence_threshold=0.5):
        """Визуализация предсказания"""
        result = self.predict(image_path, confidence_threshold)
        
        # Загрузка оригинального изображения
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Создание overlay
        mask_colored = cv2.applyColorMap(result['binary_mask'], cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
        
        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(result['probability_map'], cmap='hot')
        axes[1].set_title('Probability Map')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Detection Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Результат сохранен: {output_path}")
        
        plt.show()
        
        return result

# Пример использования
if __name__ == "__main__":
    # Инициализация детектора
    detector = ForgeryDetector('checkpoints/best_model.pth')
    
    # Предсказание для одного изображения
    result = detector.visualize_prediction(
        image_path='test_document.jpg',
        output_path='result.png',
        confidence_threshold=0.5
    )
    
    print(f"🎯 Уверенность детекции: {result['confidence']:.3f}")
