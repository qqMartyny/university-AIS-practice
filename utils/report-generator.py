"""
Генерация графиков занятости столов
"""

import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_occupancy_chart(aggregated_periods, output_path):
    """
    Генерирует график занятости столов
    
    Args:
        aggregated_periods: dict {table_id: [periods]}
        output_path: путь для сохранения PNG
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Группируем периоды по столам
    table_ids = sorted(aggregated_periods.keys())
    
    if not table_ids:
        # Пустой график
        ax.text(0.5, 0.5, 'No data available', 
               ha='center', va='center', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    # Рисуем для каждого стола
    for idx, table_id in enumerate(table_ids):
        periods = aggregated_periods[table_id]
        
        for period in periods:
            start = period['start_time']
            duration = period['duration']
            occupied = period['occupied']
            
            color = '#e74c3c' if occupied else '#2ecc71'  # Красный/Зелёный
            alpha = 0.7
            
            ax.barh(
                idx,
                duration,
                left=start,
                height=0.8,
                color=color,
                alpha=alpha,
                edgecolor='black',
                linewidth=0.5
            )
    
    # Настройка осей
    ax.set_yticks(range(len(table_ids)))
    ax.set_yticklabels([f"Table {tid}" for tid in table_ids])
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Table Occupancy Timeline', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Легенда
    occupied_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='Occupied')
    free_patch = mpatches.Patch(color='#2ecc71', alpha=0.7, label='Free')
    ax.legend(handles=[occupied_patch, free_patch], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранён: {output_path}")
    return output_path