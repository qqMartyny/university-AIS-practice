"""
Тест детекции на одном кадре
"""

import cv2
import json
from pathlib import Path
from core.detector import PersonDetector
from core.occupancy_analyzer import OccupancyAnalyzer
from utils.visualization import draw_detections

def main():
    # Загружаем конфигурацию столов
    config_path = Path("data/configs/tables_config.json")
    with open(config_path) as f:
        tables_config = json.load(f)
    
    # Загружаем изображение
    image_path = "data/videos/cafe_with_people.jpg"
    frame = cv2.imread(image_path)
    
    print(f"Загружено изображение: {image_path}")
    print(f"Столов в конфигурации: {len(tables_config['tables'])}")
    
    # Инициализируем компоненты
    detector = PersonDetector()
    analyzer = OccupancyAnalyzer(tables_config)
    
    # Детектируем людей
    print("\n=== ДЕТЕКЦИЯ ЛЮДЕЙ ===")
    people = detector.detect_people(frame)
    print(f"Найдено людей: {len(people)}")
    
    # Анализируем занятость
    print("\n=== АНАЛИЗ ЗАНЯТОСТИ ===")
    results = analyzer.analyze_frame(people)
    
    for table_id, info in results.items():
        status = "ЗАНЯТ" if info['occupied'] else "СВОБОДЕН"
        print(f"  Стол {table_id}: {status} (людей в зоне: {info['people_count']})")
    
    # Статистика
    stats = analyzer.get_statistics()
    print(f"\nОбщая статистика:")
    print(f"  Всего столов: {stats['total_tables']}")
    print(f"  Занято: {stats['occupied']}")
    print(f"  Свободно: {stats['free']}")
    print(f"  Загруженность: {stats['occupancy_rate']:.1%}")
    
    # Визуализация
    result_frame = draw_detections(frame, people, results, tables_config)
    
    # Сохраняем результат
    output_path = Path("output/results/test_detection.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_frame)
    
    print(f"\n✓ Результат сохранён: {output_path}")
    
    # Показываем результат
    cv2.imshow('Detection Result', result_frame)
    print("\nНажмите любую клавишу для выхода...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()