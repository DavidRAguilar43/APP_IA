from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import requests

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variable to store loaded data
loaded_data = None

# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class SalesDataResponse(BaseModel):
    data: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int

class StatsResponse(BaseModel):
    stats: Dict[str, Any]

class ChartDataResponse(BaseModel):
    chart_data: Dict[str, Any]

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]

# Load CSV data on startup
@app.on_event("startup")
async def load_csv_data():
    global loaded_data
    try:
        # URL del archivo CSV
        csv_url = "https://customer-assets.emergentagent.com/job_483dfffe-4fb6-4b01-b823-1ba8d1e6664d/artifacts/gzldaovo_datos_dummies_ventas.csv"
        
        # Descargar el archivo
        response = requests.get(csv_url)
        response.raise_for_status()
        
        # Guardar temporalmente
        csv_path = ROOT_DIR / "datos_ventas.csv"
        with open(csv_path, 'wb') as f:
            f.write(response.content)
        
        # Cargar el CSV
        loaded_data = pd.read_csv(csv_path)
        
        # Agregar columna de fecha simulada para análisis temporal
        # Crear fechas incrementales para las ventas
        start_date = pd.to_datetime('2024-01-01')
        date_range = pd.date_range(start=start_date, periods=len(loaded_data), freq='D')
        loaded_data['Fecha'] = date_range
        
        # Calcular ganancia (asumimos un margen del 30%)
        loaded_data['Ganancia'] = loaded_data['Precio_Total'] * 0.30
        
        logger.info(f"Datos cargados exitosamente: {len(loaded_data)} registros")
        logger.info(f"Columnas: {loaded_data.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error cargando datos: {str(e)}")
        raise

# Add your routes to the router
@api_router.get("/")
async def root():
    return {"message": "Dashboard de Ventas API"}

@api_router.get("/sales/data", response_model=SalesDataResponse)
async def get_sales_data(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_order: Optional[str] = "asc",
    category_filter: Optional[str] = None
):
    """
    Obtener datos de ventas con paginación, búsqueda y filtros
    """
    if loaded_data is None:
        raise HTTPException(status_code=500, detail="Datos no cargados")
    
    df = loaded_data.copy()
    
    # Aplicar filtro de categoría
    if category_filter and category_filter != "all":
        df = df[df['Categoría'] == category_filter]
    
    # Aplicar búsqueda
    if search:
        mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
        df = df[mask]
    
    # Aplicar ordenamiento
    if sort_by and sort_by in df.columns:
        ascending = sort_order == "asc"
        df = df.sort_values(by=sort_by, ascending=ascending)
    
    # Calcular paginación
    total = len(df)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    # Convertir a diccionario para respuesta
    df['Fecha'] = df['Fecha'].dt.strftime('%Y-%m-%d')
    paginated_data = df.iloc[start_idx:end_idx].to_dict(orient='records')
    
    return {
        "data": paginated_data,
        "total": total,
        "page": page,
        "page_size": page_size
    }

@api_router.get("/sales/stats", response_model=StatsResponse)
async def get_statistics():
    """
    Calcular estadísticas descriptivas de los datos
    """
    if loaded_data is None:
        raise HTTPException(status_code=500, detail="Datos no cargados")
    
    df = loaded_data
    
    stats = {
        "ventas_totales": float(df['Precio_Total'].sum()),
        "ganancia_total": float(df['Ganancia'].sum()),
        "numero_ventas": len(df),
        "ticket_promedio": float(df['Precio_Total'].mean()),
        
        "precio_total": {
            "media": float(df['Precio_Total'].mean()),
            "mediana": float(df['Precio_Total'].median()),
            "moda": float(df['Precio_Total'].mode()[0]) if not df['Precio_Total'].mode().empty else 0,
            "desviacion_estandar": float(df['Precio_Total'].std()),
            "minimo": float(df['Precio_Total'].min()),
            "maximo": float(df['Precio_Total'].max()),
            "percentil_25": float(df['Precio_Total'].quantile(0.25)),
            "percentil_75": float(df['Precio_Total'].quantile(0.75))
        },
        
        "cantidad": {
            "media": float(df['Cantidad'].mean()),
            "mediana": float(df['Cantidad'].median()),
            "moda": float(df['Cantidad'].mode()[0]) if not df['Cantidad'].mode().empty else 0,
            "desviacion_estandar": float(df['Cantidad'].std()),
            "minimo": int(df['Cantidad'].min()),
            "maximo": int(df['Cantidad'].max())
        },
        
        "precio_unitario": {
            "media": float(df['Precio_Unitario'].mean()),
            "mediana": float(df['Precio_Unitario'].median()),
            "desviacion_estandar": float(df['Precio_Unitario'].std()),
            "minimo": float(df['Precio_Unitario'].min()),
            "maximo": float(df['Precio_Unitario'].max())
        },
        
        "categorias": {
            "total": int(df['Categoría'].nunique()),
            "lista": df['Categoría'].unique().tolist()
        },
        
        "productos": {
            "total": int(df['Producto'].nunique())
        }
    }
    
    return {"stats": stats}

@api_router.get("/sales/charts", response_model=ChartDataResponse)
async def get_chart_data(chart_type: str = Query(..., description="Tipo de gráfica")):
    """
    Generar datos para diferentes tipos de gráficas
    """
    if loaded_data is None:
        raise HTTPException(status_code=500, detail="Datos no cargados")
    
    df = loaded_data
    
    if chart_type == "sales_by_category":
        # Ventas totales por categoría
        data = df.groupby('Categoría')['Precio_Total'].sum().sort_values(ascending=False)
        return {
            "chart_data": {
                "labels": data.index.tolist(),
                "values": data.values.tolist(),
                "title": "Ventas Totales por Categoría"
            }
        }
    
    elif chart_type == "sales_trend":
        # Tendencia de ventas en el tiempo
        df_sorted = df.sort_values('Fecha')
        data = df_sorted.groupby('Fecha')['Precio_Total'].sum()
        return {
            "chart_data": {
                "labels": [str(date) for date in data.index.tolist()],
                "values": data.values.tolist(),
                "title": "Tendencia de Ventas en el Tiempo"
            }
        }
    
    elif chart_type == "price_distribution":
        # Distribución de precios unitarios
        bins = 10
        hist, bin_edges = np.histogram(df['Precio_Unitario'], bins=bins)
        bin_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(len(bin_edges)-1)]
        return {
            "chart_data": {
                "labels": bin_labels,
                "values": hist.tolist(),
                "title": "Distribución de Precios Unitarios"
            }
        }
    
    elif chart_type == "top_products":
        # Top 10 productos más vendidos
        data = df.groupby('Producto')['Precio_Total'].sum().sort_values(ascending=False).head(10)
        return {
            "chart_data": {
                "labels": data.index.tolist(),
                "values": data.values.tolist(),
                "title": "Top 10 Productos Más Vendidos"
            }
        }
    
    elif chart_type == "quantity_vs_total":
        # Relación cantidad vs precio total
        return {
            "chart_data": {
                "data": df[['Cantidad', 'Precio_Total', 'Categoría']].to_dict(orient='records'),
                "title": "Relación Cantidad vs Precio Total"
            }
        }
    
    elif chart_type == "quantity_by_category":
        # Cantidad vendida por categoría
        data = df.groupby('Categoría')['Cantidad'].sum().sort_values(ascending=False)
        return {
            "chart_data": {
                "labels": data.index.tolist(),
                "values": data.values.tolist(),
                "title": "Cantidad Vendida por Categoría"
            }
        }
    
    elif chart_type == "profit_by_category":
        # Ganancia por categoría
        data = df.groupby('Categoría')['Ganancia'].sum().sort_values(ascending=False)
        return {
            "chart_data": {
                "labels": data.index.tolist(),
                "values": data.values.tolist(),
                "title": "Ganancia por Categoría"
            }
        }
    
    else:
        raise HTTPException(status_code=400, detail="Tipo de gráfica no válido")

@api_router.get("/sales/prediction", response_model=PredictionResponse)
async def get_predictions():
    """
    Realizar análisis predictivo usando regresión lineal múltiple
    """
    if loaded_data is None:
        raise HTTPException(status_code=500, detail="Datos no cargados")
    
    df = loaded_data.copy()
    
    # Preparar datos para el modelo
    # Variables predictoras: Cantidad, Precio_Unitario, Categoría (codificada)
    # Variable objetivo: Precio_Total
    
    # Codificar categorías
    le = LabelEncoder()
    df['Categoria_Encoded'] = le.fit_transform(df['Categoría'])
    
    # Seleccionar características
    X = df[['Cantidad', 'Precio_Unitario', 'Categoria_Encoded']]
    y = df['Precio_Total']
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Importancia de características (coeficientes)
    feature_importance = {
        "Cantidad": float(model.coef_[0]),
        "Precio_Unitario": float(model.coef_[1]),
        "Categoría": float(model.coef_[2])
    }
    
    # Preparar datos de predicción vs real para gráfica
    predictions_data = []
    for i, (actual, predicted) in enumerate(zip(y_test.values, y_pred)):
        predictions_data.append({
            "index": i,
            "actual": float(actual),
            "predicted": float(predicted)
        })
    
    return {
        "predictions": predictions_data[:50],  # Limitar a 50 puntos para visualización
        "metrics": {
            "r2_score": float(r2),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae)
        },
        "feature_importance": feature_importance
    }

@api_router.get("/sales/categories")
async def get_categories():
    """
    Obtener lista de categorías únicas
    """
    if loaded_data is None:
        raise HTTPException(status_code=500, detail="Datos no cargados")
    
    categories = loaded_data['Categoría'].unique().tolist()
    return {"categories": ["all"] + categories}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()