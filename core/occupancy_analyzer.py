"""
Анализатор занятости столов
"""

from collections import deque
import numpy as np


class OccupancyAnalyzer:
    def __init__(self, tables_config, occupied_frames=5, free_frames=20):  # 5 и 20!
        """
        Args:
            tables_config: dict с конфигурацией столов из JSON
            occupied_frames: кол-во кадров подряд для определения занятости (5 = 2.5 сек)
            free_frames: кол-во кадров подряд для определения освобождения (20 = 10 сек)
        """
        self.tables = tables_config['tables']
        self.occupied_threshold = occupied_frames
        self.free_threshold = free_frames
        
        # История состояний для каждого стола (последние 20 кадров)
        self.frame_history = {
            table['id']: deque(maxlen=max(occupied_frames, free_frames))
            for table in self.tables
        }
        
        # Текущее состояние каждого стола
        self.current_state = {table['id']: False for table in self.tables}
        
        print(f"Инициализирован анализатор для {len(self.tables)} столов")
        print(f"Правила: {occupied_frames} кадров → занят, {free_frames} кадров → свободен")
    
    def analyze_frame(self, people_detections):
        """
        Анализ занятости столов на текущем кадре
        """
        results = {}
        
        # Сначала определяем какой человек к какому столу относится
        # Один человек может занимать только ОДИН стол
        # Используем ЦЕНТР человека, а не площадь пересечения
        person_to_table = {}  # {person_index: table_id}
        
        for person_idx, person in enumerate(people_detections):
            cx, cy = person['center']  # центр человека
            
            assigned = False
            
            # Проверяем в какую зону стола попадает центр
            for table in self.tables:
                table_id = table['id']
                bbox = table['bbox']
                seating_zone = self._create_seating_zone(bbox)
                
                zx1, zy1, zx2, zy2 = seating_zone
                
                # Если центр человека внутри зоны стола
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    person_to_table[person_idx] = table_id
                    assigned = True
                    break  # Назначили - выходим
            
            # Если центр не попал ни в одну зону - не назначаем никуда
            # (это проходящий мимо человек)
        
        # Теперь для каждого стола считаем людей
        for table in self.tables:
            table_id = table['id']
            
            # Находим людей, назначенных этому столу
            people_at_table = [
                people_detections[idx] 
                for idx, tid in person_to_table.items() 
                if tid == table_id
            ]
            
            has_people = len(people_at_table) > 0
            
            # Добавляем в историю
            self.frame_history[table_id].append(has_people)
            
            # Применяем временной фильтр
            previous_state = self.current_state[table_id]
            new_state = self._apply_temporal_filter(table_id)
            self.current_state[table_id] = new_state
            
            results[table_id] = {
                'occupied': new_state,
                'people_count': len(people_at_table),
                'state_changed': previous_state != new_state,
                'people_in_zone': people_at_table,
                'history': list(self.frame_history[table_id])
            }
        
        return results
    
    def _create_seating_zone(self, table_bbox, margin=30):  # небольшой margin
        """
        Создаём зону стола с небольшим расширением
        """
        return (
            table_bbox['x1'] - margin,
            table_bbox['y1'] - margin,
            table_bbox['x2'] + margin,
            table_bbox['y2'] + margin
        )
    
    def _apply_temporal_filter(self, table_id):
        """
        Применяем правила временной фильтрации
        
        Правила:
        - Занят: если последние N кадров подряд есть люди
        - Свободен: если последние M кадров подряд нет людей
        - Иначе: сохраняем предыдущее состояние (гистерезис)
        """
        history = list(self.frame_history[table_id])
        
        if len(history) == 0:
            return False
        
        # Правило занятости
        if len(history) >= self.occupied_threshold:
            recent = history[-self.occupied_threshold:]
            if all(recent):  # все последние N кадров = True
                return True
        
        # Правило освобождения
        if len(history) >= self.free_threshold:
            recent = history[-self.free_threshold:]
            if not any(recent):  # все последние M кадров = False
                return False
        
        # Гистерезис - сохраняем текущее состояние
        return self.current_state[table_id]
    
    def get_statistics(self):
        """Получить текущую статистику по всем столам"""
        total_tables = len(self.tables)
        occupied_tables = sum(1 for state in self.current_state.values() if state)
        
        return {
            'total_tables': total_tables,
            'occupied': occupied_tables,
            'free': total_tables - occupied_tables,
            'occupancy_rate': occupied_tables / total_tables if total_tables > 0 else 0
        }