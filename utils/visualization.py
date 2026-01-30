"""
Утилиты для визуализации результатов
"""

import cv2
import numpy as np


def draw_table_zones(frame, tables_config, color=(0, 255, 0), thickness=2):
    """Рисуем зоны столов"""
    result = frame.copy()
    
    for table in tables_config['tables']:
        bbox = table['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        label = f"Table {table['id']}"
        cv2.putText(result, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result


def draw_detections(frame, people, occupancy_results, tables_config):
    """
    Рисуем людей и состояние столов
    
    Args:
        frame: исходный кадр
        people: список людей от PersonDetector
        occupancy_results: результаты от OccupancyAnalyzer
        tables_config: конфигурация столов
    """
    result = frame.copy()
    
    # Рисуем людей (зелёные рамки)
    for person in people:
        x1, y1, x2, y2 = person['bbox']
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Уверенность
        conf_text = f"{person['confidence']:.2f}"
        cv2.putText(result, conf_text, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Рисуем столы с цветом в зависимости от занятости
    for table in tables_config['tables']:
        table_id = table['id']
        bbox = table['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        occupied = occupancy_results[table_id]['occupied']
        people_count = occupancy_results[table_id]['people_count']
        
        # Цвет: красный если занят, зелёный если свободен
        color = (0, 0, 255) if occupied else (0, 255, 0)
        
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        
        # Статус
        status = f"Table {table_id}: {'OCCUPIED' if occupied else 'FREE'}"
        if people_count > 0:
            status += f" ({people_count}p)"
        
        cv2.putText(result, status, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result


def save_debug_frame(frame, output_path, table_id, state, frame_num):
    """Сохранить отладочный кадр"""
    cv2.imwrite(str(output_path), frame)
    print(f"  [DEBUG] Сохранён кадр: Table {table_id} - {state} - frame {frame_num}")