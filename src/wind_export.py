"""
Sistema de exportación de datos y reportes para análisis eólico
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import csv
from src.wind_analytics import WindAnalytics
from wind_ai_diagnosis import WindPotentialAI
from wind_visualization import WindVisualization
import warnings
warnings.filterwarnings('ignore')

class WindDataExporter:
    """Clase para exportar datos y generar reportes de análisis eólico"""
    
    def __init__(self, wind_data=None, analysis_results=None, ai_diagnosis=None):
        self.wind_data = wind_data
        self.analysis_results = analysis_results
        self.ai_diagnosis = ai_diagnosis
        self.export_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def set_data(self, wind_data, analysis_results=None, ai_diagnosis=None):
        """Establecer datos para exportación"""
        self.wind_data = wind_data
        self.analysis_results = analysis_results
        self.ai_diagnosis = ai_diagnosis
    
    def export_raw_data_csv(self, filepath=None):
        """
        Exportar datos brutos a CSV
        
        Args:
            filepath (str): Ruta del archivo CSV
            
        Returns:
            str: Ruta del archivo generado
        """
        if self.wind_data is None:
            raise ValueError("No hay datos para exportar")
        
        if filepath is None:
            filepath = f"datos_brutos_{self.export_timestamp}.csv"
        
        # Crear DataFrame con todos los datos disponibles
        data_dict = {}
        
        # Datos principales
        if 'wind_speed' in self.wind_data:
            data_dict['velocidad_viento_ms'] = self.wind_data['wind_speed']
        
        if 'wind_direction' in self.wind_data and self.wind_data['wind_direction'] is not None:
            data_dict['direccion_viento_grados'] = self.wind_data['wind_direction']
        
        if 'temperature' in self.wind_data and self.wind_data['temperature'] is not None:
            data_dict['temperatura_celsius'] = self.wind_data['temperature']
        
        if 'pressure' in self.wind_data and self.wind_data['pressure'] is not None:
            data_dict['presion_hpa'] = self.wind_data['pressure']
        
        # Timestamps
        if 'timestamps' in self.wind_data and self.wind_data['timestamps'] is not None:
            data_dict['fecha_hora'] = self.wind_data['timestamps']
        else:
            # Generar timestamps sintéticos
            n_points = len(self.wind_data['wind_speed'])
            timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
            data_dict['fecha_hora'] = timestamps
        
        # Crear DataFrame
        df = pd.DataFrame(data_dict)
        
        # Calcular métricas derivadas
        if 'velocidad_viento_ms' in df.columns:
            df['densidad_potencia_wm2'] = 0.5 * 1.225 * df['velocidad_viento_ms']**3
            df['energia_especifica_kwh_m2'] = df['densidad_potencia_wm2'] / 1000  # kWh/m²
        
        # Exportar a CSV
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"Datos brutos exportados a: {filepath}")
        return filepath
    
    def export_analysis_results_csv(self, filepath=None):
        """
        Exportar resultados de análisis a CSV
        
        Args:
            filepath (str): Ruta del archivo CSV
            
        Returns:
            str: Ruta del archivo generado
        """
        if self.analysis_results is None:
            raise ValueError("No hay resultados de análisis para exportar")
        
        if filepath is None:
            filepath = f"resultados_analisis_{self.export_timestamp}.csv"
        
        # Crear lista de métricas para CSV
        metrics_data = []
        
        # Estadísticas básicas
        if 'basic_stats' in self.analysis_results:
            stats = self.analysis_results['basic_stats']
            for key, value in stats.items():
                metrics_data.append({
                    'categoria': 'estadisticas_basicas',
                    'metrica': key,
                    'valor': value,
                    'unidad': 'm/s' if 'speed' in key or key in ['mean', 'median', 'std', 'min', 'max'] else 'adimensional'
                })
        
        # Parámetros de Weibull
        if 'weibull' in self.analysis_results:
            weibull = self.analysis_results['weibull']
            weibull_metrics = {
                'parametro_k_forma': (weibull['k'], 'adimensional'),
                'parametro_c_escala': (weibull['c'], 'm/s'),
                'media_teorica': (weibull['mu'], 'm/s'),
                'desviacion_teorica': (weibull['sigma'], 'm/s'),
                'moda': (weibull['v_modo'], 'm/s'),
                'bondad_ajuste_r2': (weibull['r2'], 'adimensional')
            }
            
            for key, (value, unit) in weibull_metrics.items():
                metrics_data.append({
                    'categoria': 'distribucion_weibull',
                    'metrica': key,
                    'valor': value,
                    'unidad': unit
                })
        
        # Turbulencia
        if 'turbulence' in self.analysis_results:
            turb = self.analysis_results['turbulence']
            turbulence_metrics = {
                'intensidad_turbulencia_media': turb.get('mean_ti', 0),
                'intensidad_turbulencia_mediana': turb.get('median_ti', 0),
                'intensidad_turbulencia_p90': turb.get('q90_ti', 0),
                'intensidad_turbulencia_p95': turb.get('q95_ti', 0),
                'intensidad_turbulencia_maxima': turb.get('max_ti', 0)
            }
            
            for key, value in turbulence_metrics.items():
                metrics_data.append({
                    'categoria': 'turbulencia',
                    'metrica': key,
                    'valor': value,
                    'unidad': 'adimensional'
                })
        
        # Densidad de potencia
        if 'power_density' in self.analysis_results:
            power = self.analysis_results['power_density']
            power_metrics = {
                'densidad_potencia_media': (power.get('mean_power_density', 0), 'W/m²'),
                'densidad_potencia_mediana': (power.get('median_power_density', 0), 'W/m²'),
                'densidad_potencia_maxima': (power.get('max_power_density', 0), 'W/m²'),
                'energia_total': (power.get('total_energy_density', 0), 'Wh/m²')
            }
            
            for key, (value, unit) in power_metrics.items():
                metrics_data.append({
                    'categoria': 'densidad_potencia',
                    'metrica': key,
                    'valor': value,
                    'unidad': unit
                })
        
        # Factor de capacidad
        if 'capacity_factor' in self.analysis_results:
            cf = self.analysis_results['capacity_factor']
            cf_metrics = {
                'factor_capacidad': (cf.get('capacity_factor', 0), 'adimensional'),
                'potencia_media': (cf.get('mean_power_output', 0), 'kW'),
                'horas_generacion': (cf.get('hours_generating', 0), 'fracción'),
                'horas_potencia_nominal': (cf.get('hours_rated_power', 0), 'fracción'),
                'energia_anual': (cf.get('annual_energy', 0), 'kWh/año')
            }
            
            for key, (value, unit) in cf_metrics.items():
                metrics_data.append({
                    'categoria': 'factor_capacidad',
                    'metrica': key,
                    'valor': value,
                    'unidad': unit
                })
        
        # Vientos extremos
        if 'extreme_winds' in self.analysis_results and 'extreme_winds' in self.analysis_results['extreme_winds']:
            extreme = self.analysis_results['extreme_winds']['extreme_winds']
            for period, speed in extreme.items():
                metrics_data.append({
                    'categoria': 'vientos_extremos',
                    'metrica': f'velocidad_retorno_{period}',
                    'valor': speed,
                    'unidad': 'm/s'
                })
        
        # Crear DataFrame y exportar
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(filepath, index=False, encoding='utf-8')
        
        print(f"Resultados de análisis exportados a: {filepath}")
        return filepath
    
    def export_ai_diagnosis_json(self, filepath=None):
        """
        Exportar diagnóstico de IA a JSON
        
        Args:
            filepath (str): Ruta del archivo JSON
            
        Returns:
            str: Ruta del archivo generado
        """
        if self.ai_diagnosis is None:
            raise ValueError("No hay diagnóstico de IA para exportar")
        
        if filepath is None:
            filepath = f"diagnostico_ia_{self.export_timestamp}.json"
        
        # Preparar datos para JSON
        diagnosis_data = {
            'timestamp': datetime.now().isoformat(),
            'clasificacion_potencial': self.ai_diagnosis.get('predicted_class', 'no_disponible'),
            'puntuacion_viabilidad': self.ai_diagnosis.get('predicted_score', 0),
            'confianza_prediccion': self.ai_diagnosis.get('confidence', 0),
            'probabilidades_clase': self.ai_diagnosis.get('class_probabilities', {}),
            'diagnostico_detallado': self.ai_diagnosis.get('diagnosis', {})
        }
        
        # Exportar a JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(diagnosis_data, f, indent=2, ensure_ascii=False)
        
        print(f"Diagnóstico de IA exportado a: {filepath}")
        return filepath
    
    def generate_comprehensive_report_markdown(self, location_info=None, filepath=None):
        """
        Generar reporte comprensivo en Markdown
        
        Args:
            location_info (dict): Información de ubicación
            filepath (str): Ruta del archivo Markdown
            
        Returns:
            str: Ruta del archivo generado
        """
        if filepath is None:
            filepath = f"reporte_potencial_eolico_{self.export_timestamp}.md"
        
        # Información de ubicación por defecto
        if location_info is None:
            location_info = {
                'latitud': 10.4,
                'longitud': -75.5,
                'nombre': 'Región Caribe de Colombia',
                'descripcion': 'Análisis de potencial eólico'
            }
        
        # Crear contenido del reporte
        content = f"""# Reporte de Evaluación del Potencial Eólico

## Información General

**Ubicación:** {location_info.get('nombre', 'No especificada')}  
**Coordenadas:** {location_info.get('latitud', 'N/A')}°N, {location_info.get('longitud', 'N/A')}°W  
**Fecha de análisis:** {datetime.now().strftime('%d de %B de %Y')}  
**Hora de generación:** {datetime.now().strftime('%H:%M:%S')}  

## Resumen Ejecutivo

"""
        
        # Agregar diagnóstico de IA si está disponible
        if self.ai_diagnosis:
            diagnosis = self.ai_diagnosis.get('diagnosis', {})
            summary = diagnosis.get('summary', 'No disponible')
            predicted_class = self.ai_diagnosis.get('predicted_class', 'no_clasificado')
            predicted_score = self.ai_diagnosis.get('predicted_score', 0)
            
            content += f"""### Clasificación del Potencial Eólico

**Clasificación:** {predicted_class.upper()}  
**Puntuación de viabilidad:** {predicted_score:.2f}/1.00  
**Resumen:** {summary}

"""
        
        # Estadísticas básicas
        if self.analysis_results and 'basic_stats' in self.analysis_results:
            stats = self.analysis_results['basic_stats']
            content += f"""## Estadísticas del Recurso Eólico

### Estadísticas Básicas

| Métrica | Valor | Unidad |
|---------|-------|--------|
| Velocidad promedio | {stats.get('mean', 0):.2f} | m/s |
| Velocidad mediana | {stats.get('median', 0):.2f} | m/s |
| Desviación estándar | {stats.get('std', 0):.2f} | m/s |
| Velocidad máxima | {stats.get('max', 0):.2f} | m/s |
| Velocidad mínima | {stats.get('min', 0):.2f} | m/s |
| Asimetría | {stats.get('skewness', 0):.2f} | - |
| Curtosis | {stats.get('kurtosis', 0):.2f} | - |
| Total de horas | {stats.get('count', 0)} | horas |

"""
        
        # Distribución de Weibull
        if self.analysis_results and 'weibull' in self.analysis_results:
            weibull = self.analysis_results['weibull']
            content += f"""### Distribución de Weibull

La distribución de Weibull es fundamental para caracterizar el recurso eólico:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| k (forma) | {weibull.get('k', 0):.2f} | Consistencia del viento |
| c (escala) | {weibull.get('c', 0):.2f} m/s | Velocidad característica |
| Media teórica | {weibull.get('mu', 0):.2f} m/s | Velocidad promedio teórica |
| Moda | {weibull.get('v_modo', 0):.2f} m/s | Velocidad más frecuente |
| R² | {weibull.get('r2', 0):.3f} | Bondad de ajuste |

**Interpretación del parámetro k:**
- k < 1.5: Vientos muy variables
- 1.5 ≤ k < 2.0: Vientos moderadamente variables  
- 2.0 ≤ k < 3.0: Vientos consistentes
- k ≥ 3.0: Vientos muy consistentes

"""
        
        # Análisis de turbulencia
        if self.analysis_results and 'turbulence' in self.analysis_results:
            turb = self.analysis_results['turbulence']
            content += f"""### Análisis de Turbulencia

| Métrica | Valor |
|---------|-------|
| Intensidad de turbulencia media | {turb.get('mean_ti', 0):.3f} |
| Intensidad de turbulencia mediana | {turb.get('median_ti', 0):.3f} |
| Percentil 90 | {turb.get('q90_ti', 0):.3f} |
| Percentil 95 | {turb.get('q95_ti', 0):.3f} |
| Máximo | {turb.get('max_ti', 0):.3f} |

**Clasificación IEC:**
- Clase I: TI ≤ 0.12 (Baja turbulencia)
- Clase II: TI ≤ 0.14 (Turbulencia moderada)
- Clase III: TI ≤ 0.16 (Alta turbulencia)

"""
        
        # Densidad de potencia y factor de capacidad
        if self.analysis_results and 'power_density' in self.analysis_results:
            power = self.analysis_results['power_density']
            content += f"""### Densidad de Potencia

| Métrica | Valor | Unidad |
|---------|-------|--------|
| Densidad de potencia media | {power.get('mean_power_density', 0):.0f} | W/m² |
| Densidad de potencia mediana | {power.get('median_power_density', 0):.0f} | W/m² |
| Densidad de potencia máxima | {power.get('max_power_density', 0):.0f} | W/m² |

"""
        
        if self.analysis_results and 'capacity_factor' in self.analysis_results:
            cf = self.analysis_results['capacity_factor']
            content += f"""### Factor de Capacidad

| Métrica | Valor | Unidad |
|---------|-------|--------|
| Factor de capacidad | {cf.get('capacity_factor', 0):.2f} | - |
| Potencia media | {cf.get('mean_power_output', 0):.0f} | kW |
| Horas de generación | {cf.get('hours_generating', 0)*100:.1f} | % del tiempo |
| Horas a potencia nominal | {cf.get('hours_rated_power', 0)*100:.1f} | % del tiempo |
| Energía anual estimada | {cf.get('annual_energy', 0):.0f} | kWh/año |

**Clasificación del factor de capacidad:**
- Excelente: > 0.35
- Bueno: 0.25 - 0.35
- Moderado: 0.15 - 0.25
- Bajo: < 0.15

"""
        
        # Análisis detallado de IA
        if self.ai_diagnosis and 'diagnosis' in self.ai_diagnosis:
            diagnosis = self.ai_diagnosis['diagnosis']
            
            content += f"""## Análisis Detallado con Inteligencia Artificial

### Análisis Técnico

"""
            
            detailed_analysis = diagnosis.get('detailed_analysis', [])
            for analysis in detailed_analysis:
                content += f"- {analysis}\n"
            
            content += f"""
### Recomendaciones

"""
            recommendations = diagnosis.get('recommendations', [])
            for rec in recommendations:
                content += f"- {rec}\n"
            
            # Factores de riesgo
            risk_factors = diagnosis.get('risk_factors', [])
            if risk_factors:
                content += f"""
### Factores de Riesgo

"""
                for risk in risk_factors:
                    content += f"- ⚠️ {risk}\n"
            
            # Oportunidades
            opportunities = diagnosis.get('opportunities', [])
            if opportunities:
                content += f"""
### Oportunidades

"""
                for opp in opportunities:
                    content += f"- ✅ {opp}\n"
        
        # Conclusiones
        content += f"""
## Conclusiones y Próximos Pasos

### Viabilidad del Proyecto

"""
        
        if self.ai_diagnosis:
            predicted_class = self.ai_diagnosis.get('predicted_class', 'no_clasificado')
            predicted_score = self.ai_diagnosis.get('predicted_score', 0)
            
            if predicted_class == 'alto':
                content += """El análisis indica un **alto potencial eólico** en esta ubicación. Se recomienda proceder con:

1. Estudios de factibilidad técnico-económica detallados
2. Mediciones in-situ para validar los datos de reanálisis
3. Evaluación de la infraestructura de conexión eléctrica
4. Análisis de impacto ambiental y social

"""
            elif predicted_class == 'moderado':
                content += """El análisis indica un **potencial eólico moderado** en esta ubicación. Se recomienda:

1. Análisis económico detallado considerando incentivos
2. Evaluación de tecnologías de aerogeneradores optimizadas
3. Estudio de opciones de almacenamiento de energía
4. Análisis de mercado eléctrico local

"""
            else:
                content += """El análisis indica un **potencial eólico limitado** en esta ubicación. Se sugiere:

1. Explorar ubicaciones alternativas en la región
2. Considerar aplicaciones de pequeña escala
3. Evaluar tecnologías de baja velocidad de viento
4. Analizar proyectos híbridos (eólico-solar)

"""
        
        content += f"""
### Limitaciones del Estudio

- Los datos utilizados provienen de modelos de reanálisis ERA5
- Se recomienda validación con mediciones in-situ
- El análisis no incluye consideraciones económicas específicas
- No se evaluaron aspectos ambientales o sociales

### Información Técnica

**Datos utilizados:** ERA5 Reanalysis (ECMWF)  
**Período de análisis:** {self.analysis_results.get('basic_stats', {}).get('count', 0)} horas de datos  
**Metodología:** Análisis estadístico con distribución de Weibull e IA  
**Herramientas:** Python, SciPy, Scikit-learn, Matplotlib  

---

*Reporte generado automáticamente por el Sistema de Evaluación del Potencial Eólico*  
*Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}*
"""
        
        # Escribir archivo
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Reporte en Markdown generado: {filepath}")
        return filepath
    
    def generate_pdf_report(self, location_info=None, include_charts=True, output_dir='reports'):
        """
        Generar reporte completo en PDF
        
        Args:
            location_info (dict): Información de ubicación
            include_charts (bool): Incluir gráficos en el reporte
            output_dir (str): Directorio de salida
            
        Returns:
            str: Ruta del archivo PDF generado
        """
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar reporte en Markdown
        md_filepath = os.path.join(output_dir, f"reporte_{self.export_timestamp}.md")
        self.generate_comprehensive_report_markdown(location_info, md_filepath)
        
        # Generar gráficos si se solicita
        if include_charts and self.wind_data:
            print("Generando gráficos para el reporte...")
            charts_dir = os.path.join(output_dir, 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Crear visualizador
            viz = WindVisualization(self.wind_data, self.analysis_results)
            
            # Generar gráficos principales
            chart_files = []
            try:
                # Dashboard resumen
                dashboard_path = os.path.join(charts_dir, 'dashboard_resumen.png')
                viz.create_summary_dashboard(dashboard_path)
                chart_files.append(dashboard_path)
                
                # Histograma con Weibull
                hist_path = os.path.join(charts_dir, 'histograma_weibull.png')
                viz.create_weibull_histogram(save_path=hist_path)
                chart_files.append(hist_path)
                
                # Boxplot horario
                box_path = os.path.join(charts_dir, 'variacion_horaria.png')
                viz.create_hourly_boxplot(save_path=box_path)
                chart_files.append(box_path)
                
                # Serie temporal
                ts_path = os.path.join(charts_dir, 'serie_temporal.png')
                viz.create_time_series_plot(save_path=ts_path)
                chart_files.append(ts_path)
                
            except Exception as e:
                print(f"Error generando gráficos: {e}")
            
            # Agregar gráficos al Markdown
            if chart_files:
                with open(md_filepath, 'a', encoding='utf-8') as f:
                    f.write("\n\n## Gráficos y Visualizaciones\n\n")
                    
                    for i, chart_path in enumerate(chart_files):
                        if os.path.exists(chart_path):
                            chart_name = os.path.basename(chart_path).replace('_', ' ').replace('.png', '').title()
                            f.write(f"### {chart_name}\n\n")
                            f.write(f"![{chart_name}]({os.path.relpath(chart_path, output_dir)})\n\n")
        
        # Convertir Markdown a PDF
        pdf_filepath = os.path.join(output_dir, f"reporte_potencial_eolico_{self.export_timestamp}.pdf")
        
        try:
            # Usar la utilidad manus-md-to-pdf
            import subprocess
            result = subprocess.run(['manus-md-to-pdf', md_filepath, pdf_filepath], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Reporte PDF generado: {pdf_filepath}")
                return pdf_filepath
            else:
                print(f"Error generando PDF: {result.stderr}")
                return md_filepath  # Devolver Markdown si falla PDF
                
        except Exception as e:
            print(f"Error convirtiendo a PDF: {e}")
            return md_filepath  # Devolver Markdown si falla PDF
    
    def export_complete_package(self, location_info=None, output_dir='export_package'):
        """
        Exportar paquete completo con todos los formatos
        
        Args:
            location_info (dict): Información de ubicación
            output_dir (str): Directorio de salida
            
        Returns:
            dict: Rutas de todos los archivos generados
        """
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        try:
            # Datos brutos CSV
            csv_raw_path = os.path.join(output_dir, f"datos_brutos_{self.export_timestamp}.csv")
            exported_files['datos_brutos_csv'] = self.export_raw_data_csv(csv_raw_path)
        except Exception as e:
            print(f"Error exportando datos brutos: {e}")
        
        try:
            # Resultados de análisis CSV
            csv_analysis_path = os.path.join(output_dir, f"resultados_analisis_{self.export_timestamp}.csv")
            exported_files['analisis_csv'] = self.export_analysis_results_csv(csv_analysis_path)
        except Exception as e:
            print(f"Error exportando análisis: {e}")
        
        try:
            # Diagnóstico IA JSON
            json_path = os.path.join(output_dir, f"diagnostico_ia_{self.export_timestamp}.json")
            exported_files['diagnostico_json'] = self.export_ai_diagnosis_json(json_path)
        except Exception as e:
            print(f"Error exportando diagnóstico IA: {e}")
        
        try:
            # Reporte PDF completo
            exported_files['reporte_pdf'] = self.generate_pdf_report(location_info, True, output_dir)
        except Exception as e:
            print(f"Error generando reporte PDF: {e}")
        
        # Crear archivo de resumen
        summary_path = os.path.join(output_dir, 'resumen_exportacion.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Resumen de Exportación - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write("Archivos generados:\n\n")
            
            for file_type, filepath in exported_files.items():
                if filepath and os.path.exists(filepath):
                    file_size = os.path.getsize(filepath) / 1024  # KB
                    f.write(f"- {file_type}: {os.path.basename(filepath)} ({file_size:.1f} KB)\n")
            
            f.write(f"\nTotal de archivos: {len(exported_files)}\n")
            f.write(f"Directorio: {os.path.abspath(output_dir)}\n")
        
        exported_files['resumen'] = summary_path
        
        print(f"\nPaquete completo exportado en: {output_dir}")
        print(f"Total de archivos generados: {len(exported_files)}")
        
        return exported_files

# Función de utilidad para exportación rápida
def quick_export_analysis(wind_data, analysis_results=None, ai_diagnosis=None, 
                         location_info=None, export_format='all'):
    """
    Exportar análisis rápidamente
    
    Args:
        wind_data (dict): Datos de viento
        analysis_results (dict): Resultados de análisis
        ai_diagnosis (dict): Diagnóstico de IA
        location_info (dict): Información de ubicación
        export_format (str): Formato de exportación ('csv', 'pdf', 'all')
        
    Returns:
        dict: Archivos generados
    """
    exporter = WindDataExporter(wind_data, analysis_results, ai_diagnosis)
    
    if export_format == 'all':
        return exporter.export_complete_package(location_info)
    elif export_format == 'csv':
        files = {}
        files['datos_csv'] = exporter.export_raw_data_csv()
        if analysis_results:
            files['analisis_csv'] = exporter.export_analysis_results_csv()
        return files
    elif export_format == 'pdf':
        return {'reporte_pdf': exporter.generate_pdf_report(location_info)}
    else:
        raise ValueError("Formato no válido. Use 'csv', 'pdf' o 'all'")

if __name__ == "__main__":
    # Ejemplo de uso
    print("=== SISTEMA DE EXPORTACIÓN DE DATOS EÓLICOS ===")
    
    # Simular datos de ejemplo
    np.random.seed(42)
    n_hours = 1000
    wind_speeds = np.random.weibull(2.0, n_hours) * 8.0 + 2.0
    
    wind_data = {
        'wind_speed': wind_speeds,
        'wind_direction': np.random.uniform(0, 360, n_hours),
        'temperature': np.random.normal(25, 5, n_hours),
        'pressure': np.random.normal(1013, 10, n_hours)
    }
    
    # Simular resultados de análisis
    analysis_results = {
        'basic_stats': {
            'mean': np.mean(wind_speeds),
            'median': np.median(wind_speeds),
            'std': np.std(wind_speeds),
            'max': np.max(wind_speeds),
            'min': np.min(wind_speeds),
            'count': len(wind_speeds),
            'skewness': 0.5,
            'kurtosis': 0.2
        },
        'weibull': {
            'k': 2.2,
            'c': 9.5,
            'mu': 8.4,
            'sigma': 4.1,
            'v_modo': 7.8,
            'r2': 0.95
        },
        'capacity_factor': {
            'capacity_factor': 0.32,
            'mean_power_output': 640,
            'annual_energy': 5600
        },
        'power_density': {
            'mean_power_density': 450,
            'median_power_density': 380,
            'max_power_density': 2500
        }
    }
    
    # Simular diagnóstico IA
    ai_diagnosis = {
        'predicted_class': 'alto',
        'predicted_score': 0.75,
        'confidence': 0.92,
        'diagnosis': {
            'summary': '✅ Alto potencial eólico - Recomendado para desarrollo',
            'detailed_analysis': [
                'Excelente recurso eólico con velocidad promedio de 9.0 m/s',
                'Distribución de vientos consistente (k=2.20)',
                'Buen factor de capacidad (0.32)'
            ],
            'recommendations': [
                'Proceder con estudios de factibilidad detallados',
                'Considerar aerogeneradores de gran escala'
            ]
        }
    }
    
    # Información de ubicación
    location_info = {
        'latitud': 10.4,
        'longitud': -75.5,
        'nombre': 'Cartagena, Colombia',
        'descripcion': 'Análisis de potencial eólico en la costa Caribe'
    }
    
    # Crear exportador
    exporter = WindDataExporter(wind_data, analysis_results, ai_diagnosis)
    
    # Exportar paquete completo
    print("Generando paquete completo de exportación...")
    exported_files = exporter.export_complete_package(location_info, 'ejemplo_exportacion')
    
    print("\nArchivos generados:")
    for file_type, filepath in exported_files.items():
        if filepath and os.path.exists(filepath):
            print(f"  {file_type}: {filepath}")
    
    print("\n¡Exportación completada exitosamente!")

