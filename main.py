"""
FastAPI приложение для анализа занятости столов в кафе
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uuid
import json
import shutil
from pathlib import Path
from datetime import datetime

from core.video_processor import VideoProcessor
from utils.report_generator import generate_occupancy_chart
from utils.pdf_generator import generate_pdf_report

# Файл для хранения истории
HISTORY_FILE = Path("data/analysis_history.json")
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_history():
    """Загрузить историю анализов из файла"""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history():
    """Сохранить историю анализов в файл"""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(analysis_results, f, indent=2)

# Загружаем историю при старте
analysis_results = load_history()

app = FastAPI(title="Cafe Table Analytics")

# Папки
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("output")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Templates
templates = Jinja2Templates(directory="templates")

# In-memory хранилище результатов
analysis_results = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "cafe-analytics"}

@app.post("/api/extract-frame")
async def extract_first_frame(video: UploadFile = File(...)):
    """
    Извлечь первый кадр из видео
    """
    import cv2
    
    # Сохраняем видео временно
    temp_video = TEMP_DIR / f"temp_{uuid.uuid4()}.mp4"
    with open(temp_video, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    # Извлекаем первый кадр
    cap = cv2.VideoCapture(str(temp_video))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        temp_video.unlink()
        raise HTTPException(400, "Cannot read video")
    
    # Сохраняем кадр
    frame_path = TEMP_DIR / f"frame_{uuid.uuid4()}.jpg"
    cv2.imwrite(str(frame_path), frame)
    
    # Удаляем видео
    temp_video.unlink()
    
    # Возвращаем кадр
    response = FileResponse(frame_path, media_type="image/jpeg")
    
    return response

@app.post("/api/analyze")
async def analyze_video(
    video: UploadFile = File(...),
    tables_config: str = Form(...)
):
    """
    Запуск анализа видео
    
    Args:
        video: Видео файл
        tables_config: JSON строка с координатами столов
    
    Returns:
        {"analysis_id": "uuid", "status": "processing"}
    """
    try:
        # Генерируем ID анализа
        analysis_id = str(uuid.uuid4())
        
        # Сохраняем видео во временную папку
        video_path = TEMP_DIR / f"{analysis_id}.mp4"
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        # Парсим конфигурацию столов
        try:
            tables_config_dict = json.loads(tables_config)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid tables_config JSON")
        
        # Сохраняем config во временный файл
        config_path = TEMP_DIR / f"{analysis_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(tables_config_dict, f)
        
        # Запускаем обработку
        print(f"\n[{analysis_id}] Начало обработки видео: {video.filename}")
        
        processor = VideoProcessor(
            video_path=str(video_path),
            tables_config_path=str(config_path),
            frame_interval=0.5
        )
        
        results = processor.process()
        
        # Сохраняем результаты в память
        analysis_results[analysis_id] = {
            'id': analysis_id,
            'video_filename': video.filename,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'results': results
        }

        save_history()  # Сохраняем в файл
        
        # Удаляем временные файлы
        video_path.unlink()
        config_path.unlink()
        
        print(f"[{analysis_id}] Обработка завершена")
        
        return {
            'analysis_id': analysis_id,
            'status': 'completed'
        }
        
    except Exception as e:
        print(f"Ошибка обработки: {str(e)}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    
@app.get("/api/history")
async def get_history():
    """
    Получить список всех анализов
    """
    history_list = [
        {
            'id': data['id'],
            'video_filename': data['video_filename'],
            'created_at': data['created_at'],
            'status': data['status'],
            'total_tables': data['results']['overall_statistics']['total_tables'],
            'avg_occupancy': data['results']['overall_statistics']['average_occupancy_rate']
        }
        for data in analysis_results.values()
    ]
    
    # Сортируем по дате (новые первые)
    history_list.sort(key=lambda x: x['created_at'], reverse=True)
    
    return history_list

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Получить результаты анализа
    
    Returns:
        JSON с результатами анализа
    """
    if analysis_id not in analysis_results:
        raise HTTPException(404, "Analysis not found")
    
    return analysis_results[analysis_id]

@app.get("/api/analysis/{analysis_id}/chart")
async def download_chart(analysis_id: str):
    """
    Скачать PNG график занятости
    """
    if analysis_id not in analysis_results:
        raise HTTPException(404, "Analysis not found")
    
    data = analysis_results[analysis_id]['results']
    
    # Генерируем график
    chart_path = OUTPUT_DIR / f"{analysis_id}_chart.png"
    generate_occupancy_chart(data['aggregated_periods'], str(chart_path))
    
    return FileResponse(
        chart_path,
        media_type="image/png",
        filename=f"occupancy_chart_{analysis_id}.png"
    )

@app.get("/api/analysis/{analysis_id}/pdf")
async def download_pdf(analysis_id: str):
    """
    Скачать PDF отчёт
    """
    if analysis_id not in analysis_results:
        raise HTTPException(404, "Analysis not found")
    
    data = analysis_results[analysis_id]
    results = data['results']
    
    # Генерируем график
    chart_path = OUTPUT_DIR / f"{analysis_id}_chart.png"
    generate_occupancy_chart(results['aggregated_periods'], str(chart_path))
    
    # Генерируем PDF
    pdf_path = OUTPUT_DIR / f"{analysis_id}_report.pdf"
    generate_pdf_report(
        video_filename=data['video_filename'],
        duration=results['video_info']['duration'],
        statistics=results['statistics'],
        overall_stats=results['overall_statistics'],
        chart_path=str(chart_path),
        output_path=str(pdf_path)
    )
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"cafe_analytics_report_{analysis_id}.pdf"
    )

@app.get("/api/analysis/{analysis_id}/json")
async def download_json(analysis_id: str):
    """
    Скачать JSON с результатами
    """
    if analysis_id not in analysis_results:
        raise HTTPException(404, "Analysis not found")
    
    data = analysis_results[analysis_id]
    
    # Сохраняем JSON
    json_path = OUTPUT_DIR / f"{analysis_id}_results.json"
    with open(json_path, "w") as f:
        json.dump(data['results'], f, indent=2)
    
    return FileResponse(
        json_path,
        media_type="application/json",
        filename=f"analysis_{analysis_id}.json"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)