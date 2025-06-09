"""
Rutas de la API para datos meteorológicos ERA5
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.era5_data import ERA5DataAccess

era5_bp = Blueprint('era5', __name__)
era5_client = ERA5DataAccess()

@era5_bp.route('/health', methods=['GET'])
def health_check():
    """Verificar el estado de la API"""
    return jsonify({
        'status': 'ok',
        'message': 'API de datos meteorológicos funcionando correctamente',
        'timestamp': datetime.now().isoformat()
    })

@era5_bp.route('/regions', methods=['GET'])
def get_regions():
    """Obtener información sobre las regiones disponibles"""
    return jsonify({
        'caribbean_colombia': {
            'name': 'Región Caribe de Colombia',
            'bounds': {
                'lat_min': era5_client.lat_min,
                'lat_max': era5_client.lat_max,
                'lon_min': era5_client.lon_min,
                'lon_max': era5_client.lon_max
            },
            'description': 'Región costera del Caribe colombiano con alto potencial eólico'
        }
    })

@era5_bp.route('/variables', methods=['GET'])
def get_variables():
    """Obtener lista de variables meteorológicas disponibles"""
    return jsonify({
        'variables': era5_client.variables,
        'wind_levels': ['10m', '100m'],
        'description': 'Variables meteorológicas disponibles en ERA5'
    })

@era5_bp.route('/data', methods=['POST'])
def get_meteorological_data():
    """
    Obtener datos meteorológicos para un rango de fechas y ubicación específica
    
    Body JSON:
    {
        "start_date": "2024-01-01",
        "end_date": "2024-01-31", 
        "lat_min": 10.0,
        "lat_max": 12.0,
        "lon_min": -75.0,
        "lon_max": -73.0,
        "variables": ["u10", "v10", "sp", "t2m"],
        "aggregation": "daily" // "hourly", "daily", "monthly"
    }
    """
    try:
        data = request.get_json()
        
        # Validar parámetros requeridos
        required_fields = ['start_date', 'end_date']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo requerido: {field}'}), 400
        
        # Parámetros por defecto
        start_date = data['start_date']
        end_date = data['end_date']
        variables = data.get('variables', ['u10', 'v10', 'sp', 't2m'])
        aggregation = data.get('aggregation', 'daily')
        
        # Límites geográficos (usar región completa si no se especifica)
        lat_min = data.get('lat_min', era5_client.lat_min)
        lat_max = data.get('lat_max', era5_client.lat_max)
        lon_min = data.get('lon_min', era5_client.lon_min)
        lon_max = data.get('lon_max', era5_client.lon_max)
        
        # Simular datos ERA5
        dataset = era5_client.simulate_era5_data(start_date, end_date, variables)
        
        # Filtrar por región geográfica
        dataset = dataset.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max)
        )
        
        # Agregar datos según el tipo solicitado
        if aggregation == 'daily':
            dataset = dataset.resample(time='D').mean()
        elif aggregation == 'monthly':
            dataset = dataset.resample(time='M').mean()
        
        # Calcular velocidad y dirección del viento si están disponibles
        result_data = {}
        
        for var in variables:
            if var in dataset.data_vars:
                result_data[var] = dataset[var].values.tolist()
        
        # Calcular métricas de viento si están disponibles las componentes
        if 'u10' in dataset.data_vars and 'v10' in dataset.data_vars:
            wind_speed = era5_client.get_wind_speed(dataset['u10'], dataset['v10'])
            wind_direction = era5_client.get_wind_direction(dataset['u10'], dataset['v10'])
            result_data['wind_speed_10m'] = wind_speed.values.tolist()
            result_data['wind_direction_10m'] = wind_direction.values.tolist()
        
        if 'u100' in dataset.data_vars and 'v100' in dataset.data_vars:
            wind_speed_100 = era5_client.get_wind_speed(dataset['u100'], dataset['v100'])
            wind_direction_100 = era5_client.get_wind_direction(dataset['u100'], dataset['v100'])
            result_data['wind_speed_100m'] = wind_speed_100.values.tolist()
            result_data['wind_direction_100m'] = wind_direction_100.values.tolist()
        
        return jsonify({
            'status': 'success',
            'data': result_data,
            'metadata': {
                'start_date': start_date,
                'end_date': end_date,
                'aggregation': aggregation,
                'bounds': {
                    'lat_min': float(dataset.latitude.min()),
                    'lat_max': float(dataset.latitude.max()),
                    'lon_min': float(dataset.longitude.min()),
                    'lon_max': float(dataset.longitude.max())
                },
                'coordinates': {
                    'latitude': dataset.latitude.values.tolist(),
                    'longitude': dataset.longitude.values.tolist(),
                    'time': [t.isoformat() for t in pd.to_datetime(dataset.time.values)]
                },
                'variables': variables
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@era5_bp.route('/point-data', methods=['POST'])
def get_point_data():
    """
    Obtener datos meteorológicos para un punto específico
    
    Body JSON:
    {
        "latitude": 11.0,
        "longitude": -74.0,
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "variables": ["u10", "v10", "sp", "t2m"]
    }
    """
    try:
        data = request.get_json()
        
        # Validar parámetros requeridos
        required_fields = ['latitude', 'longitude', 'start_date', 'end_date']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo requerido: {field}'}), 400
        
        lat = data['latitude']
        lon = data['longitude']
        start_date = data['start_date']
        end_date = data['end_date']
        variables = data.get('variables', ['u10', 'v10', 'sp', 't2m'])
        
        # Simular datos ERA5
        dataset = era5_client.simulate_era5_data(start_date, end_date, variables)
        
        # Seleccionar el punto más cercano
        point_data = dataset.sel(latitude=lat, longitude=lon, method='nearest')
        
        # Convertir a formato JSON
        result_data = {}
        for var in variables:
            if var in point_data.data_vars:
                result_data[var] = point_data[var].values.tolist()
        
        # Calcular métricas de viento
        if 'u10' in point_data.data_vars and 'v10' in point_data.data_vars:
            wind_speed = era5_client.get_wind_speed(point_data['u10'], point_data['v10'])
            wind_direction = era5_client.get_wind_direction(point_data['u10'], point_data['v10'])
            result_data['wind_speed_10m'] = wind_speed.values.tolist()
            result_data['wind_direction_10m'] = wind_direction.values.tolist()
        
        return jsonify({
            'status': 'success',
            'data': result_data,
            'metadata': {
                'latitude': float(point_data.latitude),
                'longitude': float(point_data.longitude),
                'start_date': start_date,
                'end_date': end_date,
                'time': [t.isoformat() for t in pd.to_datetime(point_data.time.values)],
                'variables': variables
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@era5_bp.route('/grid-summary', methods=['POST'])
def get_grid_summary():
    """
    Obtener resumen estadístico de datos en una grilla
    
    Body JSON:
    {
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "lat_min": 10.0,
        "lat_max": 12.0,
        "lon_min": -75.0,
        "lon_max": -73.0,
        "statistic": "mean" // "mean", "max", "min", "std"
    }
    """
    try:
        data = request.get_json()
        
        # Validar parámetros requeridos
        required_fields = ['start_date', 'end_date']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Campo requerido: {field}'}), 400
        
        start_date = data['start_date']
        end_date = data['end_date']
        statistic = data.get('statistic', 'mean')
        
        # Límites geográficos
        lat_min = data.get('lat_min', era5_client.lat_min)
        lat_max = data.get('lat_max', era5_client.lat_max)
        lon_min = data.get('lon_min', era5_client.lon_min)
        lon_max = data.get('lon_max', era5_client.lon_max)
        
        # Simular datos ERA5
        variables = ['u10', 'v10', 'u100', 'v100', 'sp', 't2m']
        dataset = era5_client.simulate_era5_data(start_date, end_date, variables)
        
        # Filtrar por región
        dataset = dataset.sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max)
        )
        
        # Calcular estadística temporal
        if statistic == 'mean':
            summary = dataset.mean(dim='time')
        elif statistic == 'max':
            summary = dataset.max(dim='time')
        elif statistic == 'min':
            summary = dataset.min(dim='time')
        elif statistic == 'std':
            summary = dataset.std(dim='time')
        else:
            return jsonify({'error': 'Estadística no válida'}), 400
        
        # Calcular velocidad del viento
        wind_speed_10m = era5_client.get_wind_speed(summary['u10'], summary['v10'])
        wind_speed_100m = era5_client.get_wind_speed(summary['u100'], summary['v100'])
        
        result_data = {
            'wind_speed_10m': wind_speed_10m.values.tolist(),
            'wind_speed_100m': wind_speed_100m.values.tolist(),
            'temperature': summary['t2m'].values.tolist(),
            'pressure': summary['sp'].values.tolist()
        }
        
        return jsonify({
            'status': 'success',
            'data': result_data,
            'metadata': {
                'statistic': statistic,
                'start_date': start_date,
                'end_date': end_date,
                'coordinates': {
                    'latitude': summary.latitude.values.tolist(),
                    'longitude': summary.longitude.values.tolist()
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

