"""
Генерация PDF отчётов
"""

from fpdf import FPDF
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Cafe Table Analytics Report', border=0, ln=True, align='C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_pdf_report(video_filename, duration, statistics, overall_stats, chart_path, output_path):
    """
    Генерирует PDF отчёт
    
    Args:
        video_filename: имя видео файла
        duration: длительность видео (сек)
        statistics: статистика по столам
        overall_stats: общая статистика
        chart_path: путь к PNG графику
        output_path: путь для сохранения PDF
    """
    pdf = PDFReport()
    pdf.add_page()
    
    # Метаданные
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Analysis Information', ln=True)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Video: {video_filename}", ln=True)
    pdf.cell(0, 8, f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)", ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    # Общая статистика
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Overall Statistics', ln=True)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Total Tables: {overall_stats['total_tables']}", ln=True)
    pdf.cell(0, 8, f"Average Occupancy Rate: {overall_stats['average_occupancy_rate']:.1f}%", ln=True)
    pdf.ln(10)
    
    # Статистика по каждому столу
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Table Statistics', ln=True)
    
    pdf.set_font('Arial', '', 11)
    for table_id, stats in sorted(statistics.items()):
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"Table {table_id}:", ln=True)
        
        pdf.set_font('Arial', '', 11)
        pdf.cell(10)  # Отступ
        pdf.cell(0, 7, f"Occupied Time: {stats['total_occupied_time']:.1f}s", ln=True)
        pdf.cell(10)
        pdf.cell(0, 7, f"Occupancy Rate: {stats['occupancy_rate']:.1f}%", ln=True)
        pdf.cell(10)
        pdf.cell(0, 7, f"Number of Occupations: {stats['occupation_count']}", ln=True)
        pdf.ln(3)
    
    # График на новой странице
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Occupancy Timeline', ln=True)
    pdf.ln(5)
    
    # Вставляем график
    pdf.image(chart_path, x=10, w=190)
    
    pdf.output(output_path)
    print(f"PDF отчёт сохранён: {output_path}")
    return output_path