"""
API completa integrada para análisis de potencial eólico
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import tempfile
import json
from datetime import datetime
import traceback

# Importar módulos desarrollados
from era5_data import ERA5DataAccess
from wind_analytics import WindAnalytics, quick_wind_analysis
from wind_ai_diagnosis import WindPotentialAI, quick_ai_diagnosis
from wind_visualization import WindVisualization, quick_wind_charts
from wind_export import WindDataExporter, quick_export_analysis

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas las rutas

# Instancias globales
era5_access = ERA5DataAccess()
ai_system = WindPotentialAI()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Verificar estado de la API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'era5_data': 'available',
            'wind_analytics': 'available',
            'ai_diagnosis': 'available',
            'visualization': 'available',
            'export': 'available'
        }
    })

@app.route('/api/wind-data', methods=['POST'])
def get_wind_data():
    """
    Obtener datos de viento para una ubicación y período específico
    """
    try:
        data = request.get_json()
        
        # Parámetros requeridos
        lat = data.get('latitude')
        lon = data.get('longitude')
        start_date = data.get('start_date', '2024-01-01')
        end_date = data.get('end_date', '2024-01-31')
        variables = data.get('variables', ['u10', 'v10', 'sp', 't2m'])
        
        if lat is None or lon is None:
            return jsonify({'error': 'Latitud y longitud son requeridas'}), 400
        
        # Obtener datos (simulados por ahora)
        wind_data = era5_access.get_wind_data_for_location(
            lat, lon, start_date, end_date, variables
        )
        
        return jsonify({
            'success': True,
            'data': wind_data,
            'location': {'latitude': lat, 'longitude': lon},
            'period': {'start': start_date, 'end': end_date},
            'variables': variables
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/analyze-wind', methods=['POST'])
def analyze_wind():
    """
    Realizar análisis completo de datos de viento
    """
    try:
        data = request.get_json()
        
        # Obtener datos de viento
        wind_data = data.get('wind_data')
        if not wind_data:
            return jsonify({'error': 'Datos de viento requeridos'}), 400
        
        # Realizar análisis estadístico
        analyzer = WindAnalytics()
        
        # Cargar datos
        wind_speeds = wind_data.get('wind_speed', [])
        wind_directions = wind_data.get('wind_direction')
        timestamps = wind_data.get('timestamps')
        temperature = wind_data.get('temperature')
        pressure = wind_data.get('pressure')
        
        analyzer.load_wind_data(
            wind_speeds=wind_speeds,
            timestamps=timestamps,
            wind_directions=wind_directions,
            temperature=temperature,
            pressure=pressure
        )
        
        # Generar reporte comprensivo
        analysis_results = analyzer.generate_comprehensive_report()
        
        # Diagnóstico con IA
        ai_diagnosis = ai_system.predict_wind_potential(
            analysis_results['statistical_analysis']
        )
        
        return jsonify({
            'success': True,
            'analysis_results': analysis_results,
            'ai_diagnosis': ai_diagnosis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/generate-charts', methods=['POST'])
def generate_charts():
    """
    Generar gráficos estadísticos
    """
    try:
        data = request.get_json()
        
        wind_data = data.get('wind_data')
        analysis_results = data.get('analysis_results')
        chart_types = data.get('chart_types', ['summary_dashboard'])
        
        if not wind_data:
            return jsonify({'error': 'Datos de viento requeridos'}), 400
        
        # Crear visualizador
        viz = WindVisualization(wind_data, analysis_results)
        
        # Generar gráficos solicitados
        charts = {}
        
        for chart_type in chart_types:
            try:
                if chart_type == 'weibull_histogram':
                    chart_data = viz.create_weibull_histogram()
                elif chart_type == 'hourly_boxplot':
                    chart_data = viz.create_hourly_boxplot()
                elif chart_type == 'time_series':
                    chart_data = viz.create_time_series_plot()
                elif chart_type == 'wind_rose':
                    chart_data = viz.create_wind_rose()
                elif chart_type == 'turbulence_analysis':
                    chart_data = viz.create_turbulence_analysis()
                elif chart_type == 'annual_weibull':
                    chart_data = viz.create_annual_weibull_variation()
                elif chart_type == 'summary_dashboard':
                    chart_data = viz.create_summary_dashboard()
                else:
                    continue
                
                if chart_data:
                    charts[chart_type] = chart_data
                    
            except Exception as e:
                print(f"Error generando gráfico {chart_type}: {e}")
                continue
        
        return jsonify({
            'success': True,
            'charts': charts,
            'generated_count': len(charts)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/export-data', methods=['POST'])
def export_data():
    """
    Exportar datos en formato CSV
    """
    try:
        data = request.get_json()
        
        wind_data = data.get('wind_data')
        analysis_results = data.get('analysis_results')
        ai_diagnosis = data.get('ai_diagnosis')
        export_type = data.get('export_type', 'raw_data')
        
        if not wind_data:
            return jsonify({'error': 'Datos de viento requeridos'}), 400
        
        # Crear exportador
        exporter = WindDataExporter(wind_data, analysis_results, ai_diagnosis)
        
        # Crear archivo temporal
        temp_dir = tempfile.mkdtemp()
        
        if export_type == 'raw_data':
            filepath = os.path.join(temp_dir, 'datos_brutos.csv')
            exporter.export_raw_data_csv(filepath)
        elif export_type == 'analysis_results':
            filepath = os.path.join(temp_dir, 'resultados_analisis.csv')
            exporter.export_analysis_results_csv(filepath)
        elif export_type == 'ai_diagnosis':
            filepath = os.path.join(temp_dir, 'diagnostico_ia.json')
            exporter.export_ai_diagnosis_json(filepath)
        else:
            return jsonify({'error': 'Tipo de exportación no válido'}), 400
        
        # Enviar archivo
        return send_file(
            filepath,
            as_attachment=True,
            download_name=os.path.basename(filepath)
        )
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """
    Generar reporte completo en PDF
    """
    try:
        data = request.get_json()
        
        wind_data = data.get('wind_data')
        analysis_results = data.get('analysis_results')
        ai_diagnosis = data.get('ai_diagnosis')
        location_info = data.get('location_info', {})
        
        if not wind_data:
            return jsonify({'error': 'Datos de viento requeridos'}), 400
        
        # Crear exportador
        exporter = WindDataExporter(wind_data, analysis_results, ai_diagnosis)
        
        # Crear directorio temporal
        temp_dir = tempfile.mkdtemp()
        
        # Generar reporte PDF
        pdf_path = exporter.generate_pdf_report(
            location_info=location_info,
            include_charts=True,
            output_dir=temp_dir
        )
        
        if pdf_path and os.path.exists(pdf_path):
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=f'reporte_potencial_eolico_{datetime.now().strftime("%Y%m%d")}.pdf'
            )
        else:
            return jsonify({'error': 'Error generando reporte PDF'}), 500
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/quick-analysis', methods=['POST'])
def quick_analysis():
    """
    Análisis rápido completo para una ubicación
    """
    try:
        data = request.get_json()
        
        # Parámetros de entrada
        lat = data.get('latitude')
        lon = data.get('longitude')
        start_date = data.get('start_date', '2024-01-01')
        end_date = data.get('end_date', '2024-01-31')
        
        if lat is None or lon is None:
            return jsonify({'error': 'Latitud y longitud son requeridas'}), 400
        
        # 1. Obtener datos de viento
        wind_data = era5_access.get_wind_data_for_location(
            lat, lon, start_date, end_date
        )
        
        # 2. Análisis estadístico
        analyzer = WindAnalytics()
        analyzer.load_wind_data(
            wind_speeds=wind_data['wind_speed'],
            wind_directions=wind_data.get('wind_direction'),
            timestamps=wind_data.get('timestamps')
        )
        analysis_results = analyzer.generate_comprehensive_report()
        
        # 3. Diagnóstico con IA
        ai_diagnosis = ai_system.predict_wind_potential(
            analysis_results['statistical_analysis']
        )
        
        # 4. Generar gráfico resumen
        viz = WindVisualization(wind_data, analysis_results['statistical_analysis'])
        dashboard_chart = viz.create_summary_dashboard()
        
        # 5. Preparar respuesta
        response = {
            'success': True,
            'location': {
                'latitude': lat,
                'longitude': lon,
                'coordinates_text': f"{lat:.4f}°N, {lon:.4f}°W"
            },
            'period': {
                'start': start_date,
                'end': end_date
            },
            'wind_data_summary': {
                'total_hours': len(wind_data['wind_speed']),
                'mean_speed': float(np.mean(wind_data['wind_speed'])),
                'max_speed': float(np.max(wind_data['wind_speed'])),
                'data_quality': 'good' if len(wind_data['wind_speed']) > 100 else 'limited'
            },
            'viability_assessment': analysis_results['viability_assessment'],
            'key_metrics': {
                'mean_wind_speed': analysis_results['statistical_analysis']['basic_stats']['mean'],
                'weibull_k': analysis_results['statistical_analysis']['weibull']['k'],
                'weibull_c': analysis_results['statistical_analysis']['weibull']['c'],
                'capacity_factor': analysis_results['statistical_analysis']['capacity_factor']['capacity_factor'],
                'power_density': analysis_results['statistical_analysis']['power_density']['mean_power_density']
            },
            'ai_diagnosis': ai_diagnosis,
            'dashboard_chart': dashboard_chart,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/available-variables', methods=['GET'])
def get_available_variables():
    """
    Obtener variables meteorológicas disponibles
    """
    variables = {
        'u10': {
            'name': 'Componente U del viento a 10m',
            'unit': 'm/s',
            'description': 'Componente este-oeste del viento a 10 metros'
        },
        'v10': {
            'name': 'Componente V del viento a 10m',
            'unit': 'm/s',
            'description': 'Componente norte-sur del viento a 10 metros'
        },
        'u100': {
            'name': 'Componente U del viento a 100m',
            'unit': 'm/s',
            'description': 'Componente este-oeste del viento a 100 metros'
        },
        'v100': {
            'name': 'Componente V del viento a 100m',
            'unit': 'm/s',
            'description': 'Componente norte-sur del viento a 100 metros'
        },
        'sp': {
            'name': 'Presión superficial',
            'unit': 'Pa',
            'description': 'Presión atmosférica a nivel del mar'
        },
        't2m': {
            'name': 'Temperatura a 2m',
            'unit': 'K',
            'description': 'Temperatura del aire a 2 metros'
        }
    }
    
    return jsonify({
        'success': True,
        'variables': variables,
        'default_selection': ['u10', 'v10', 'sp', 't2m']
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    # Entrenar modelo de IA al iniciar
    print("Inicializando sistema de IA...")
    try:
        ai_system.train_models()
        print("Sistema de IA inicializado correctamente")
    except Exception as e:
        print(f"Error inicializando IA: {e}")
    
    print("Iniciando servidor de API...")
    app.run(host='0.0.0.0', port=5002, debug=True)

