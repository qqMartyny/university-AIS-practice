"""
Тест обработки видео
"""

import json
from pathlib import Path
from core.video_processor import VideoProcessor


def main():
    # Пути
    video_path = "data/videos/test_video.mp4"
    config_path = "data/configs/tables_config.json"
    
    print("=== ТЕСТ ОБРАБОТКИ ВИДЕО ===\n")
    
    # Создаём процессор с новыми параметрами
    processor = VideoProcessor(
        video_path=video_path,
        tables_config_path=config_path,
        frame_interval=0.5  # каждые 0.5 секунды (2 кадра в секунду)
    )
    
    # Обрабатываем
    results = processor.process()
    
    # Сохраняем результаты
    results_path = processor.save_results()
    
    # Выводим статистику
    print("\n=== СТАТИСТИКА ===")
    for table_id, stats in results['statistics'].items():
        print(f"\nСтол {table_id}:")
        print(f"  Время занятости: {stats['total_occupied_time']:.1f} сек")
        print(f"  Загруженность: {stats['occupancy_rate']:.1f}%")
        print(f"  Периодов занятости: {stats['occupation_count']}")
    
    print(f"\nСредняя загруженность всех столов: {results['overall_statistics']['average_occupancy_rate']:.1f}%")
    
    print(f"\n✓ Отладочные кадры в: output/debug_frames/")
    print(f"✓ Результаты в: {results_path}")


if __name__ == "__main__":
    main()