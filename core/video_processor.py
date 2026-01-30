"""
Обработчик видео - покадровый анализ занятости столов
"""

import cv2
import json
from pathlib import Path
from collections import defaultdict
from core.detector import PersonDetector
from core.occupancy_analyzer import OccupancyAnalyzer
from utils.visualization import draw_detections, save_debug_frame


class VideoProcessor:
    def __init__(self, video_path: str, tables_config_path: str, frame_interval: int = 10):
        """
        Args:
            video_path: путь к видео файлу
            tables_config_path: путь к tables_config.json
            frame_interval: интервал между обрабатываемыми кадрами (секунды)
        """
        self.video_path = video_path
        self.frame_interval = frame_interval
        
        # Загружаем конфигурацию столов
        with open(tables_config_path) as f:
            self.tables_config = json.load(f)
        
        # Инициализируем компоненты
        self.detector = PersonDetector()
        self.analyzer = OccupancyAnalyzer(self.tables_config)
        
        # Открываем видео
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        print(f"\n=== ВИДЕО ===")
        print(f"Путь: {video_path}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Всего кадров: {self.total_frames}")
        print(f"Длительность: {self.duration:.1f} сек ({self.duration/60:.1f} мин)")
        print(f"Интервал обработки: каждые {frame_interval} сек")
        print(f"Будет обработано кадров: ~{int(self.duration / frame_interval)}")
        
        # Структура для хранения timeline
        self.timeline = defaultdict(list)
        
    def process(self):
        """Обработка всего видео"""
        
        frame_skip = int(self.fps * self.frame_interval)
        frame_num = 0
        processed_frames = 0
        
        print(f"\n=== ОБРАБОТКА ===\n")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Обрабатываем только каждый N-й кадр
            if frame_num % frame_skip == 0:
                timestamp = frame_num / self.fps
                
                print(f"\n--- Кадр {frame_num} (t={timestamp:.1f}s) ---")
                
                # Детекция людей
                people = self.detector.detect_people(frame)
                print(f"Детектировано людей: {len(people)}")
                
                # Анализ занятости
                results = self.analyzer.analyze_frame(people)
                
                # Подробный вывод для каждого стола
                for table_id, info in results.items():
                    history_str = ''.join(['T' if h else 'F' for h in info['history']])
                    status = "ЗАНЯТ" if info['occupied'] else "СВОБОДЕН"
                    changed = " [ИЗМЕНЕНИЕ!]" if info['state_changed'] else ""
                    
                    print(f"  Стол {table_id}: {status} | "
                        f"Людей: {info['people_count']} | "
                        f"История: [{history_str}]{changed}")
                
                # Сохраняем в timeline
                for table_id, info in results.items():
                    self.timeline[table_id].append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'occupied': info['occupied'],
                        'people_count': info['people_count']
                    })
                
                # Убрали сохранение изображений!
                
                processed_frames += 1
            
            frame_num += 1
        
        self.cap.release()
        
        print(f"\n✓ Обработка завершена!")
        print(f"  Обработано кадров: {processed_frames}")
        
        return self._build_results()
    
    def _build_results(self):
        """Построение итоговых результатов"""
        
        results = {
            'video_info': {
                'path': self.video_path,
                'fps': self.fps,
                'duration': self.duration,
                'total_frames': self.total_frames
            },
            'processing_info': {
                'frame_interval': self.frame_interval,
                'frames_processed': len(self.timeline[0]) if 0 in self.timeline else 0
            },
            'timeline': {},
            'aggregated_periods': {},
            'statistics': {}
        }
        
        # Для каждого стола
        for table_id, events in self.timeline.items():
            results['timeline'][table_id] = events
            
            # Агрегируем периоды
            periods = self._aggregate_periods(events)
            results['aggregated_periods'][table_id] = periods
            
            # Статистика по столу
            total_occupied_time = sum(
                p['duration'] for p in periods if p['occupied']
            )
            
            results['statistics'][table_id] = {
                'total_occupied_time': total_occupied_time,
                'occupancy_rate': (total_occupied_time / self.duration) * 100 if self.duration > 0 else 0,
                'occupation_count': sum(1 for p in periods if p['occupied'])
            }
        
        # Общая статистика
        total_tables = len(self.tables_config['tables'])
        avg_occupancy = sum(
            s['occupancy_rate'] for s in results['statistics'].values()
        ) / total_tables if total_tables > 0 else 0
        
        results['overall_statistics'] = {
            'total_tables': total_tables,
            'average_occupancy_rate': avg_occupancy
        }
        
        return results
    
    def _aggregate_periods(self, events):
        """Агрегируем последовательные события в периоды"""
        if not events:
            return []
        
        periods = []
        current_period = None
        
        for event in events:
            if current_period is None:
                # Начало первого периода
                current_period = {
                    'start_time': event['timestamp'],
                    'end_time': event['timestamp'],
                    'occupied': event['occupied'],
                    'start_frame': event['frame'],
                    'end_frame': event['frame']
                }
            elif current_period['occupied'] == event['occupied']:
                # Продолжение текущего периода
                current_period['end_time'] = event['timestamp']
                current_period['end_frame'] = event['frame']
            else:
                # Смена состояния - завершаем период
                current_period['duration'] = current_period['end_time'] - current_period['start_time']
                periods.append(current_period)
                
                # Начинаем новый период
                current_period = {
                    'start_time': event['timestamp'],
                    'end_time': event['timestamp'],
                    'occupied': event['occupied'],
                    'start_frame': event['frame'],
                    'end_frame': event['frame']
                }
        
        # Добавляем последний период
        if current_period:
            current_period['duration'] = current_period['end_time'] - current_period['start_time']
            periods.append(current_period)
        
        return periods
    
    def save_results(self, output_path: str = None):
        """Сохранение результатов в JSON"""
        if output_path is None:
            output_path = Path("output/results") / f"analysis_{Path(self.video_path).stem}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = self._build_results()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Результаты сохранены: {output_path}")
        
        return str(output_path)