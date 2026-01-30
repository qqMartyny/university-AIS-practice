"""
Инструмент для калибровки столов.
1. Автоматическая детекция столов с помощью YOLO
2. Интерактивное редактирование (добавление/удаление)
"""

import cv2
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path


class TableCalibrator:
    def __init__(self, image_path: str, model_path: str = None):
        self.image_path = image_path
        
        # Автоматически находим модель
        if model_path is None:
            script_dir = Path(__file__).parent.parent
            model_path = script_dir / "data/models/yolov8n.pt"
            if not model_path.exists():
                model_path = 'yolov8n.pt'
        
        self.model = YOLO(str(model_path))
        
        # Загружаем ОРИГИНАЛЬНОЕ изображение
        self.original_image = cv2.imread(image_path)
        
        if self.original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Масштабируем для отображения
        self.frame = self._resize_to_screen(self.original_image)
        self.original_frame = self.frame.copy()
        
        # Вычисляем коэффициенты масштабирования
        self.scale_x = self.original_image.shape[1] / self.frame.shape[1]
        self.scale_y = self.original_image.shape[0] / self.frame.shape[0]
        
        print(f"Оригинальное разрешение: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
        print(f"Отображаемое разрешение: {self.frame.shape[1]}x{self.frame.shape[0]}")
        print(f"Коэффициент масштабирования: {self.scale_x:.3f}x, {self.scale_y:.3f}y")
        
        self.tables = []  # [(x1, y1, x2, y2), ...] в координатах ЭКРАНА
        self.current_box = None
        self.drawing = False
        self.mode = 'auto'
        
    def _resize_to_screen(self, frame, max_width=1920, max_height=1080):
        """Масштабирует изображение под размер экрана"""
        h, w = frame.shape[:2]
        
        # Уменьшаем до 90% от размера экрана
        max_width = int(max_width * 0.9)
        max_height = int(max_height * 0.9)
        
        if w <= max_width and h <= max_height:
            return frame
        
        # Вычисляем масштаб
        scale = min(max_width / w, max_height / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Изображение масштабировано: {w}x{h} → {new_w}x{new_h}")
        
        return resized
        
    def auto_detect_tables(self):
        """Автоматическая детекция столов с помощью YOLO"""
        print("Автоматическая детекция столов...")
        
        # Детектируем на ОТОБРАЖАЕМОМ кадре (для скорости)
        results = self.model(self.frame, classes=[60, 56], conf=0.3)  # table, chair
        
        tables_raw = []
        chairs = []
        
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            if cls == 60:  # dining table
                tables_raw.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'conf': conf
                })
            elif cls == 56:  # chair
                chairs.append((int(x1), int(y1), int(x2), int(y2)))
        
        print(f"Найдено столов: {len(tables_raw)}, стульев: {len(chairs)}")
        
        # Фильтрация: оставляем только столы с стульями рядом
        filtered_tables = []
        for table in tables_raw:
            bbox = table['bbox']
            nearby_chairs = self._count_nearby_chairs(bbox, chairs)
            
            # Оставляем столы с 2+ стульями
            if nearby_chairs >= 2:
                filtered_tables.append(bbox)
        
        # Если не нашли столы, но есть группы стульев - используем их
        if len(filtered_tables) == 0 and len(chairs) > 0:
            print("Столы не найдены, ищем группы стульев...")
            filtered_tables = self._detect_tables_from_chairs(chairs)
        
        self.tables = filtered_tables
        print(f"После фильтрации осталось столов: {len(self.tables)}")
        
        return len(self.tables) > 0
    
    def _count_nearby_chairs(self, table_bbox, chairs, radius=200):
        """Подсчитываем стулья рядом со столом"""
        tx1, ty1, tx2, ty2 = table_bbox
        table_center_x = (tx1 + tx2) / 2
        table_center_y = (ty1 + ty2) / 2
        
        count = 0
        for chair in chairs:
            cx1, cy1, cx2, cy2 = chair
            chair_center_x = (cx1 + cx2) / 2
            chair_center_y = (cy1 + cy2) / 2
            
            distance = np.sqrt(
                (table_center_x - chair_center_x)**2 + 
                (table_center_y - chair_center_y)**2
            )
            
            if distance < radius:
                count += 1
        
        return count
    
    def _detect_tables_from_chairs(self, chairs, min_chairs=3):
        """Находим столы по группам стульев"""
        try:
            from scipy.spatial.distance import cdist
        except ImportError:
            print("Для группировки стульев нужен scipy: pip install scipy")
            return []
        
        if len(chairs) < min_chairs:
            return []
        
        # Центры стульев
        centers = np.array([
            [(x1+x2)/2, (y1+y2)/2] 
            for x1, y1, x2, y2 in chairs
        ])
        
        # Расстояния между стульями
        distances = cdist(centers, centers)
        
        # Находим кластеры (стулья на расстоянии < 300px)
        clusters = []
        used = set()
        
        for i in range(len(chairs)):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            for j in range(len(chairs)):
                if j != i and j not in used and distances[i][j] < 300:
                    cluster.append(j)
                    used.add(j)
            
            if len(cluster) >= min_chairs:
                clusters.append([chairs[idx] for idx in cluster])
        
        # Строим bbox вокруг каждого кластера
        tables = []
        for cluster in clusters:
            x1 = min(c[0] for c in cluster)
            y1 = min(c[1] for c in cluster)
            x2 = max(c[2] for c in cluster)
            y2 = max(c[3] for c in cluster)
            
            # Немного уменьшаем (центр между стульями)
            margin = 30
            tables.append((x1+margin, y1+margin, x2-margin, y2-margin))
        
        return tables
    
    def mouse_callback(self, event, x, y, flags, param):
        """Обработка событий мыши для ручного рисования"""
        
        if self.mode != 'manual':
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box[2] = x
                self.current_box[3] = y
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box[2] = x
            self.current_box[3] = y
            
            # Добавляем новый стол
            x1, y1, x2, y2 = self.current_box
            # Нормализуем координаты (если тянули справа налево)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            if x2 - x1 > 20 and y2 - y1 > 20:  # Минимальный размер
                self.tables.append((x1, y1, x2, y2))
                print(f"Добавлен стол {len(self.tables)}: ({x1}, {y1}, {x2}, {y2}) [экранные координаты]")
            
            self.current_box = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Правая кнопка - удаление ближайшего стола
            self._remove_nearest_table(x, y)
    
    def _remove_nearest_table(self, x, y, threshold=100):
        """Удаляем ближайший стол к точке клика"""
        if not self.tables:
            print("Нет столов для удаления")
            return
        
        min_dist = float('inf')
        min_idx = -1
        
        for i, (x1, y1, x2, y2) in enumerate(self.tables):
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            dist = np.sqrt((center_x - x)**2 + (center_y - y)**2)
            
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        if min_idx >= 0 and min_dist < threshold:
            removed = self.tables.pop(min_idx)
            print(f"Удалён стол {min_idx}: {removed}")
        else:
            print(f"Не найден стол рядом с ({x}, {y}), ближайший на расстоянии {min_dist:.0f}px")
    
    def draw_tables(self):
        """Отрисовка всех столов на кадре"""
        frame = self.original_frame.copy()
        
        # Рисуем существующие столы
        for i, (x1, y1, x2, y2) in enumerate(self.tables):
            color = (0, 255, 0) if self.mode == 'auto' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Подпись
            label = f"Table {i}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Рисуем текущий рисуемый прямоугольник
        if self.current_box and self.drawing:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return frame
    
    def run(self):
        """Основной цикл калибровки"""
        
        # Шаг 1: Автоматическая детекция
        auto_success = self.auto_detect_tables()
        
        if auto_success:
            print("\n=== АВТОМАТИЧЕСКАЯ ДЕТЕКЦИЯ ===")
            print(f"Найдено столов: {len(self.tables)}")
            print("\nУправление:")
            print("  SPACE - принять автоматическую детекцию")
            print("  M - перейти к ручному редактированию")
            print("  Q - выход без сохранения")
        else:
            print("\n=== АВТОМАТИЧЕСКАЯ ДЕТЕКЦИЯ НЕ УДАЛАСЬ ===")
            print("Переход к ручному режиму...")
            self.mode = 'manual'
        
        # Создаём окно
        window_name = 'Table Calibration'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            frame = self.draw_tables()
            
            # Инструкции
            mode_text = "AUTO" if self.mode == 'auto' else "MANUAL"
            cv2.putText(frame, f"Mode: {mode_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if self.mode == 'manual':
                cv2.putText(frame, "LMB: Draw | RMB: Delete | SPACE: Save | Q: Quit",
                           (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Выход без сохранения")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord(' '):
                # Сохранение
                if len(self.tables) > 0:
                    print(f"\nСохранено столов: {len(self.tables)}")
                    cv2.destroyAllWindows()
                    return self.tables
                else:
                    print("Нет столов для сохранения!")
            
            elif key == ord('m'):
                # Переход в ручной режим
                if self.mode == 'auto':
                    self.mode = 'manual'
                    print("\n=== РУЧНОЙ РЕЖИМ ===")
                    print("Управление:")
                    print("  ЛКМ - нарисовать новый стол (зажать и тянуть)")
                    print("  ПКМ - удалить ближайший стол")
                    print("  SPACE - сохранить")
                    print("  Q - выход")
    
    def save_config(self, output_path: str = None):
        """Сохранение конфигурации в JSON с координатами в ОРИГИНАЛЬНОМ масштабе"""
        if output_path is None:
            output_path = Path(__file__).parent.parent / "data/configs/tables_config.json"
        
        # Преобразуем координаты из экранных в оригинальные
        tables_original_coords = []
        for i, (x1, y1, x2, y2) in enumerate(self.tables):
            orig_x1 = int(x1 * self.scale_x)
            orig_y1 = int(y1 * self.scale_y)
            orig_x2 = int(x2 * self.scale_x)
            orig_y2 = int(y2 * self.scale_y)
            
            tables_original_coords.append({
                'id': i,
                'bbox': {
                    'x1': orig_x1,
                    'y1': orig_y1,
                    'x2': orig_x2,
                    'y2': orig_y2
                }
            })
            
            print(f"  Стол {i}: экранные ({x1}, {y1}, {x2}, {y2}) → оригинальные ({orig_x1}, {orig_y1}, {orig_x2}, {orig_y2})")
        
        config = {
            'calibration_image': self.image_path,
            'frame_size': {
                'width': self.original_image.shape[1],
                'height': self.original_image.shape[0]
            },
            'tables': tables_original_coords
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nКонфигурация сохранена: {output_path}")
        
        # Сохраняем изображение с визуализацией (в экранных координатах для просмотра)
        viz_path = output_path.parent / "calibration_result.jpg"
        cv2.imwrite(str(viz_path), self.draw_tables())
        print(f"Визуализация сохранена: {viz_path}")
        
        return str(output_path)


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python calibrator_tool.py <image_path>")
        print("Example: python calibrator_tool.py data/videos/video_frame.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Загрузка изображения: {image_path}")
    calibrator = TableCalibrator(image_path)
    
    tables = calibrator.run()
    
    if tables:
        calibrator.save_config()
        print(f"\n✓ Калибровка завершена успешно!")
        print(f"  Столов: {len(tables)}")
    else:
        print("\n✗ Калибровка отменена")


if __name__ == "__main__":
    main()