"""
YOLO детектор - обёртка для детекции людей
"""

from ultralytics import YOLO
import numpy as np
from pathlib import Path


class PersonDetector:
    def __init__(self, model_path: str = None):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к YOLO модели (если None - использует yolov8n.pt)
        """
        if model_path is None:
            script_dir = Path(__file__).parent.parent
            model_path = script_dir / "data/models/yolov8n.pt"
            if not model_path.exists():
                model_path = 'yolov8n.pt'
        
        self.model = YOLO(str(model_path))
        print(f"Модель загружена: {model_path}")
    
    def detect_people(self, frame, conf_threshold=0.3):
        """
        Детекция людей на кадре
        
        Args:
            frame: numpy array с изображением
            conf_threshold: минимальная уверенность детекции
        
        Returns:
            List of dicts: [{'bbox': (x1,y1,x2,y2), 'confidence': float}, ...]
        """
        # Детектируем только людей (class 0 в COCO)
        results = self.model(frame, classes=[0], conf=conf_threshold, verbose=False)
        
        people = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                people.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'center': (int((x1+x2)/2), int((y1+y2)/2)),
                    'width': int(x2 - x1),
                    'height': int(y2 - y1)
                })
        
        return people