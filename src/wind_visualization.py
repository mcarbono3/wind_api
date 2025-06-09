"""
Módulo para generación de visualizaciones y gráficos estadísticos para análisis eólico
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import gamma
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import base64
import io
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WindVisualization:
    """Clase para generar visualizaciones de datos eólicos"""
    
    def __init__(self, wind_data=None, analysis_results=None):
        self.wind_data = wind_data
        self.analysis_results = analysis_results
        self.figures = {}
        
    def set_data(self, wind_data, analysis_results=None):
        """Establecer datos de viento y resultados de análisis"""
        self.wind_data = wind_data
        self.analysis_results = analysis_results
    
    def create_weibull_histogram(self, bins=30, save_path=None):
        """
        Crear histograma con ajuste de distribución de Weibull
        
        Args:
            bins (int): Número de bins para el histograma
            save_path (str): Ruta para guardar la figura
            
        Returns:
            str: Ruta de la imagen generada o datos base64
        """
        if self.wind_data is None:
            return None
        
        wind_speeds = self.wind_data.get('wind_speed', [])
        if len(wind_speeds) == 0:
            return None
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histograma
        n, bins_edges, patches = ax.hist(wind_speeds, bins=bins, density=True, 
                                        alpha=0.7, color='skyblue', 
                                        edgecolor='black', linewidth=0.5)
        
        # Ajuste de Weibull si está disponible
        if self.analysis_results and 'weibull' in self.analysis_results:
            weibull_params = self.analysis_results['weibull']
            k, c = weibull_params['k'], weibull_params['c']
            
            # Generar curva teórica
            x = np.linspace(0, max(wind_speeds), 200)
            weibull_pdf = (k/c) * (x/c)**(k-1) * np.exp(-(x/c)**k)
            
            ax.plot(x, weibull_pdf, 'r-', linewidth=2, 
                   label=f'Weibull (k={k:.2f}, c={c:.2f})')
            
            # Añadir estadísticas
            ax.axvline(weibull_params['mu'], color='red', linestyle='--', 
                      label=f'Media: {weibull_params["mu"]:.2f} m/s')
            if weibull_params['v_modo'] > 0:
                ax.axvline(weibull_params['v_modo'], color='orange', linestyle='--', 
                          label=f'Moda: {weibull_params["v_modo"]:.2f} m/s')
        
        # Configurar gráfico
        ax.set_xlabel('Velocidad del Viento (m/s)', fontsize=12)
        ax.set_ylabel('Densidad de Probabilidad', fontsize=12)
        ax.set_title('Distribución de Velocidades del Viento con Ajuste de Weibull', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar o convertir a base64
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
    
    def create_hourly_boxplot(self, save_path=None):
        """
        Crear boxplot de velocidades del viento por hora del día
        
        Args:
            save_path (str): Ruta para guardar la figura
            
        Returns:
            str: Ruta de la imagen generada o datos base64
        """
        if self.wind_data is None or 'timestamps' not in self.wind_data:
            # Simular datos horarios si no hay timestamps
            wind_speeds = self.wind_data.get('wind_speed', [])
            hours = np.tile(np.arange(24), len(wind_speeds) // 24 + 1)[:len(wind_speeds)]
        else:
            # Usar timestamps reales
            timestamps = pd.to_datetime(self.wind_data['timestamps'])
            wind_speeds = self.wind_data['wind_speed']
            hours = timestamps.hour
        
        # Crear DataFrame
        df = pd.DataFrame({
            'wind_speed': wind_speeds,
            'hour': hours
        })
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Boxplot
        box_plot = ax.boxplot([df[df['hour'] == h]['wind_speed'].values 
                              for h in range(24)],
                             labels=range(24),
                             patch_artist=True,
                             showfliers=True)
        
        # Colorear cajas
        colors = plt.cm.viridis(np.linspace(0, 1, 24))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Configurar gráfico
        ax.set_xlabel('Hora del Día', fontsize=12)
        ax.set_ylabel('Velocidad del Viento (m/s)', fontsize=12)
        ax.set_title('Variación Horaria de la Velocidad del Viento', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Añadir línea de promedio
        hourly_means = [df[df['hour'] == h]['wind_speed'].mean() for h in range(24)]
        ax.plot(range(1, 25), hourly_means, 'ro-', linewidth=2, 
               markersize=4, label='Promedio horario')
        ax.legend()
        
        plt.tight_layout()
        
        # Guardar o convertir a base64
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
    
    def create_time_series_plot(self, variables=['wind_speed'], save_path=None):
        """
        Crear gráfico de evolución temporal de variables
        
        Args:
            variables (list): Variables a graficar
            save_path (str): Ruta para guardar la figura
            
        Returns:
            str: Ruta de la imagen generada o datos base64
        """
        if self.wind_data is None:
            return None
        
        # Crear timestamps si no existen
        if 'timestamps' not in self.wind_data or self.wind_data['timestamps'] is None:
            n_points = len(self.wind_data.get('wind_speed', []))
            timestamps = pd.date_range(start='2024-01-01', periods=n_points, freq='H')
        else:
            timestamps = pd.to_datetime(self.wind_data['timestamps'])
        
        # Crear figura con subplots
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4*n_vars), sharex=True)
        if n_vars == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, var in enumerate(variables):
            if var in self.wind_data and self.wind_data[var] is not None:
                data = self.wind_data[var]
                
                # Gráfico principal
                axes[i].plot(timestamps, data, color=colors[i % len(colors)], 
                           linewidth=1, alpha=0.8)
                
                # Media móvil
                if len(data) > 24:
                    rolling_mean = pd.Series(data).rolling(window=24, center=True).mean()
                    axes[i].plot(timestamps, rolling_mean, color='red', 
                               linewidth=2, label='Media móvil (24h)')
                
                # Configurar subplot
                ylabel = self._get_variable_label(var)
                axes[i].set_ylabel(ylabel, fontsize=11)
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                
                # Estadísticas en el título
                mean_val = np.mean(data)
                std_val = np.std(data)
                axes[i].set_title(f'{ylabel} - Media: {mean_val:.2f}, Desv. Est.: {std_val:.2f}',
                                fontsize=12)
        
        # Configurar eje x
        axes[-1].set_xlabel('Tiempo', fontsize=12)
        plt.suptitle('Evolución Temporal de Variables Meteorológicas', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Guardar o convertir a base64
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
    
    def create_wind_rose(self, save_path=None):
        """
        Crear rosa de vientos
        
        Args:
            save_path (str): Ruta para guardar la figura
            
        Returns:
            str: Ruta de la imagen generada o datos base64
        """
        if (self.wind_data is None or 
            'wind_direction' not in self.wind_data or 
            self.wind_data['wind_direction'] is None):
            return None
        
        wind_speeds = self.wind_data['wind_speed']
        wind_directions = self.wind_data['wind_direction']
        
        # Crear bins direccionales y de velocidad
        dir_bins = np.arange(0, 361, 22.5)  # 16 sectores
        speed_bins = [0, 3, 6, 9, 12, 15, float('inf')]
        speed_labels = ['0-3', '3-6', '6-9', '9-12', '12-15', '>15']
        speed_colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
        
        # Calcular frecuencias
        dir_centers = (dir_bins[:-1] + dir_bins[1:]) / 2
        frequencies = np.zeros((len(dir_centers), len(speed_bins)-1))
        
        for i, dir_center in enumerate(dir_centers):
            # Máscara direccional
            dir_mask = ((wind_directions >= dir_bins[i]) & 
                       (wind_directions < dir_bins[i+1]))
            
            for j, (speed_min, speed_max) in enumerate(zip(speed_bins[:-1], speed_bins[1:])):
                # Máscara de velocidad
                if speed_max == float('inf'):
                    speed_mask = wind_speeds >= speed_min
                else:
                    speed_mask = (wind_speeds >= speed_min) & (wind_speeds < speed_max)
                
                # Frecuencia combinada
                combined_mask = dir_mask & speed_mask
                frequencies[i, j] = np.sum(combined_mask) / len(wind_speeds) * 100
        
        # Crear gráfico polar
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Convertir direcciones a radianes
        theta = np.radians(dir_centers)
        
        # Gráfico de barras apiladas
        bottom = np.zeros(len(dir_centers))
        for j in range(len(speed_bins)-1):
            ax.bar(theta, frequencies[:, j], width=np.radians(22.5), 
                  bottom=bottom, label=f'{speed_labels[j]} m/s',
                  color=speed_colors[j], alpha=0.8)
            bottom += frequencies[:, j]
        
        # Configurar gráfico
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_title('Rosa de Vientos\n(Frecuencia por Dirección y Velocidad)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, np.max(np.sum(frequencies, axis=1)) * 1.1)
        ax.set_ylabel('Frecuencia (%)', fontsize=11)
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
        
        plt.tight_layout()
        
        # Guardar o convertir a base64
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
    
    def create_turbulence_analysis(self, save_path=None):
        """
        Crear gráfico de análisis de turbulencia
        
        Args:
            save_path (str): Ruta para guardar la figura
            
        Returns:
            str: Ruta de la imagen generada o datos base64
        """
        if (self.analysis_results is None or 
            'turbulence' not in self.analysis_results):
            return None
        
        turbulence_data = self.analysis_results['turbulence']
        ti_values = turbulence_data.get('ti_values', [])
        mean_speeds = turbulence_data.get('mean_speeds', [])
        
        if len(ti_values) == 0:
            return None
        
        # Crear figura con subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: TI vs Velocidad del viento
        ax1.scatter(mean_speeds, ti_values, alpha=0.6, s=30, color='blue')
        
        # Líneas de referencia IEC
        speed_range = np.linspace(min(mean_speeds), max(mean_speeds), 100)
        iec_class_a = 0.16  # Clase A
        iec_class_b = 0.14  # Clase B
        iec_class_c = 0.12  # Clase C
        
        ax1.axhline(y=iec_class_a, color='red', linestyle='--', 
                   label='IEC Clase A (0.16)')
        ax1.axhline(y=iec_class_b, color='orange', linestyle='--', 
                   label='IEC Clase B (0.14)')
        ax1.axhline(y=iec_class_c, color='green', linestyle='--', 
                   label='IEC Clase C (0.12)')
        
        ax1.set_xlabel('Velocidad del Viento (m/s)', fontsize=11)
        ax1.set_ylabel('Índice de Turbulencia', fontsize=11)
        ax1.set_title('Intensidad de Turbulencia vs Velocidad del Viento', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Histograma de TI
        ax2.hist(ti_values, bins=20, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        
        # Estadísticas
        mean_ti = np.mean(ti_values)
        median_ti = np.median(ti_values)
        p90_ti = np.percentile(ti_values, 90)
        
        ax2.axvline(mean_ti, color='red', linestyle='-', 
                   label=f'Media: {mean_ti:.3f}')
        ax2.axvline(median_ti, color='orange', linestyle='-', 
                   label=f'Mediana: {median_ti:.3f}')
        ax2.axvline(p90_ti, color='purple', linestyle='-', 
                   label=f'P90: {p90_ti:.3f}')
        
        ax2.set_xlabel('Índice de Turbulencia', fontsize=11)
        ax2.set_ylabel('Densidad de Probabilidad', fontsize=11)
        ax2.set_title('Distribución del Índice de Turbulencia', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar o convertir a base64
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
    
    def create_annual_weibull_variation(self, save_path=None):
        """
        Crear gráfico de variación anual de parámetros de Weibull
        
        Args:
            save_path (str): Ruta para guardar la figura
            
        Returns:
            str: Ruta de la imagen generada o datos base64
        """
        if self.wind_data is None:
            return None
        
        wind_speeds = self.wind_data['wind_speed']
        
        # Simular datos mensuales si no hay timestamps
        if 'timestamps' not in self.wind_data or self.wind_data['timestamps'] is None:
            # Dividir datos en 12 meses
            n_per_month = len(wind_speeds) // 12
            monthly_data = [wind_speeds[i*n_per_month:(i+1)*n_per_month] 
                           for i in range(12)]
        else:
            timestamps = pd.to_datetime(self.wind_data['timestamps'])
            df = pd.DataFrame({'wind_speed': wind_speeds, 'timestamp': timestamps})
            df.set_index('timestamp', inplace=True)
            monthly_data = [df[df.index.month == month]['wind_speed'].values 
                           for month in range(1, 13)]
        
        # Calcular parámetros de Weibull mensuales
        months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        k_values = []
        c_values = []
        mean_values = []
        
        for month_data in monthly_data:
            if len(month_data) > 10:  # Suficientes datos
                # Ajustar Weibull
                try:
                    shape, loc, scale = stats.weibull_min.fit(month_data[month_data > 0], floc=0)
                    k_values.append(shape)
                    c_values.append(scale)
                    mean_values.append(np.mean(month_data))
                except:
                    k_values.append(np.nan)
                    c_values.append(np.nan)
                    mean_values.append(np.nan)
            else:
                k_values.append(np.nan)
                c_values.append(np.nan)
                mean_values.append(np.nan)
        
        # Crear figura
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        x = np.arange(12)
        
        # Subplot 1: Parámetro k
        ax1.plot(x, k_values, 'bo-', linewidth=2, markersize=6)
        ax1.set_ylabel('Parámetro k (Forma)', fontsize=11)
        ax1.set_title('Variación Anual de Parámetros de Weibull', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
        
        # Subplot 2: Parámetro c
        ax2.plot(x, c_values, 'ro-', linewidth=2, markersize=6)
        ax2.set_ylabel('Parámetro c (Escala) [m/s]', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        
        # Subplot 3: Velocidad media
        ax3.plot(x, mean_values, 'go-', linewidth=2, markersize=6)
        ax3.set_ylabel('Velocidad Media [m/s]', fontsize=11)
        ax3.set_xlabel('Mes', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(bottom=0)
        
        # Configurar eje x
        ax3.set_xticks(x)
        ax3.set_xticklabels(months)
        
        plt.tight_layout()
        
        # Guardar o convertir a base64
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
    
    def create_summary_dashboard(self, save_path=None):
        """
        Crear dashboard resumen con múltiples gráficos
        
        Args:
            save_path (str): Ruta para guardar la figura
            
        Returns:
            str: Ruta de la imagen generada o datos base64
        """
        if self.wind_data is None:
            return None
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        wind_speeds = self.wind_data['wind_speed']
        
        # 1. Histograma con Weibull
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(wind_speeds, bins=20, density=True, alpha=0.7, color='skyblue')
        if self.analysis_results and 'weibull' in self.analysis_results:
            weibull = self.analysis_results['weibull']
            x = np.linspace(0, max(wind_speeds), 100)
            k, c = weibull['k'], weibull['c']
            pdf = (k/c) * (x/c)**(k-1) * np.exp(-(x/c)**k)
            ax1.plot(x, pdf, 'r-', linewidth=2)
        ax1.set_title('Distribución de Velocidades', fontsize=10)
        ax1.set_xlabel('Velocidad (m/s)', fontsize=9)
        
        # 2. Serie temporal
        ax2 = fig.add_subplot(gs[0, 1:])
        if len(wind_speeds) > 100:
            sample_indices = np.linspace(0, len(wind_speeds)-1, 100, dtype=int)
            ax2.plot(sample_indices, wind_speeds[sample_indices], 'b-', alpha=0.7)
        else:
            ax2.plot(wind_speeds, 'b-', alpha=0.7)
        ax2.set_title('Serie Temporal de Velocidad del Viento', fontsize=10)
        ax2.set_ylabel('Velocidad (m/s)', fontsize=9)
        
        # 3. Estadísticas básicas (texto)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        if self.analysis_results and 'basic_stats' in self.analysis_results:
            stats_text = self._format_basic_stats(self.analysis_results['basic_stats'])
        else:
            stats_text = f"""Estadísticas Básicas:
Media: {np.mean(wind_speeds):.2f} m/s
Desv. Est.: {np.std(wind_speeds):.2f} m/s
Máximo: {np.max(wind_speeds):.2f} m/s
Mínimo: {np.min(wind_speeds):.2f} m/s"""
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        # 4. Boxplot por velocidad
        ax4 = fig.add_subplot(gs[1, 1])
        speed_ranges = ['0-3', '3-6', '6-9', '9-12', '>12']
        speed_data = []
        for i, (low, high) in enumerate([(0,3), (3,6), (6,9), (9,12), (12,100)]):
            mask = (wind_speeds >= low) & (wind_speeds < high)
            speed_data.append(wind_speeds[mask])
        
        ax4.boxplot([data for data in speed_data if len(data) > 0], 
                   labels=[speed_ranges[i] for i, data in enumerate(speed_data) if len(data) > 0])
        ax4.set_title('Distribución por Rangos', fontsize=10)
        ax4.set_xlabel('Rango (m/s)', fontsize=9)
        
        # 5. Métricas de viabilidad
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        if self.analysis_results:
            viability_text = self._format_viability_metrics()
        else:
            viability_text = "Métricas de viabilidad\nno disponibles"
        ax5.text(0.1, 0.9, viability_text, transform=ax5.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
        
        # 6. Gráfico de densidad de potencia
        ax6 = fig.add_subplot(gs[2, :])
        power_density = 0.5 * 1.225 * wind_speeds**3  # W/m²
        ax6.scatter(wind_speeds, power_density, alpha=0.5, s=10)
        ax6.set_xlabel('Velocidad del Viento (m/s)', fontsize=9)
        ax6.set_ylabel('Densidad de Potencia (W/m²)', fontsize=9)
        ax6.set_title('Densidad de Potencia vs Velocidad del Viento', fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Dashboard de Análisis Eólico', fontsize=16, fontweight='bold')
        
        # Guardar o convertir a base64
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{image_base64}"
    
    def _get_variable_label(self, variable):
        """Obtener etiqueta legible para variable"""
        labels = {
            'wind_speed': 'Velocidad del Viento (m/s)',
            'wind_direction': 'Dirección del Viento (°)',
            'temperature': 'Temperatura (°C)',
            'pressure': 'Presión (hPa)',
            'u10': 'Componente U del Viento 10m (m/s)',
            'v10': 'Componente V del Viento 10m (m/s)',
            'u100': 'Componente U del Viento 100m (m/s)',
            'v100': 'Componente V del Viento 100m (m/s)'
        }
        return labels.get(variable, variable)
    
    def _format_basic_stats(self, stats):
        """Formatear estadísticas básicas para mostrar"""
        return f"""Estadísticas Básicas:
Media: {stats['mean']:.2f} m/s
Mediana: {stats['median']:.2f} m/s
Desv. Est.: {stats['std']:.2f} m/s
Máximo: {stats['max']:.2f} m/s
Mínimo: {stats['min']:.2f} m/s
Asimetría: {stats['skewness']:.2f}
Curtosis: {stats['kurtosis']:.2f}"""
    
    def _format_viability_metrics(self):
        """Formatear métricas de viabilidad"""
        text = "Métricas de Viabilidad:\n"
        
        if 'capacity_factor' in self.analysis_results:
            cf = self.analysis_results['capacity_factor']['capacity_factor']
            text += f"Factor de Capacidad: {cf:.2f}\n"
        
        if 'power_density' in self.analysis_results:
            pd_val = self.analysis_results['power_density']['mean_power_density']
            text += f"Densidad de Potencia: {pd_val:.0f} W/m²\n"
        
        if 'weibull' in self.analysis_results:
            weibull = self.analysis_results['weibull']
            text += f"Weibull k: {weibull['k']:.2f}\n"
            text += f"Weibull c: {weibull['c']:.2f} m/s\n"
        
        return text
    
    def generate_all_charts(self, output_dir='charts'):
        """
        Generar todos los gráficos disponibles
        
        Args:
            output_dir (str): Directorio de salida
            
        Returns:
            dict: Rutas de todos los gráficos generados
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        charts = {}
        
        # Generar cada tipo de gráfico
        chart_methods = [
            ('weibull_histogram', self.create_weibull_histogram),
            ('hourly_boxplot', self.create_hourly_boxplot),
            ('time_series', self.create_time_series_plot),
            ('wind_rose', self.create_wind_rose),
            ('turbulence_analysis', self.create_turbulence_analysis),
            ('annual_weibull', self.create_annual_weibull_variation),
            ('summary_dashboard', self.create_summary_dashboard)
        ]
        
        for chart_name, method in chart_methods:
            try:
                save_path = os.path.join(output_dir, f'{chart_name}.png')
                result = method(save_path=save_path)
                if result:
                    charts[chart_name] = result
                    print(f"Gráfico generado: {chart_name}")
            except Exception as e:
                print(f"Error generando {chart_name}: {e}")
        
        return charts

# Función de utilidad para generar gráficos rápidamente
def quick_wind_charts(wind_data, analysis_results=None, output_dir='charts'):
    """
    Generar gráficos de análisis eólico rápidamente
    
    Args:
        wind_data (dict): Datos de viento
        analysis_results (dict): Resultados de análisis
        output_dir (str): Directorio de salida
        
    Returns:
        dict: Rutas de gráficos generados
    """
    viz = WindVisualization(wind_data, analysis_results)
    return viz.generate_all_charts(output_dir)

if __name__ == "__main__":
    # Ejemplo de uso
    print("=== GENERADOR DE VISUALIZACIONES EÓLICAS ===")
    
    # Simular datos de viento
    np.random.seed(42)
    n_hours = 8760  # Un año
    wind_speeds = np.random.weibull(2.0, n_hours) * 8.0 + 2.0
    wind_directions = np.random.uniform(0, 360, n_hours)
    timestamps = pd.date_range('2024-01-01', periods=n_hours, freq='H')
    
    wind_data = {
        'wind_speed': wind_speeds,
        'wind_direction': wind_directions,
        'timestamps': timestamps
    }
    
    # Simular resultados de análisis
    analysis_results = {
        'basic_stats': {
            'mean': np.mean(wind_speeds),
            'median': np.median(wind_speeds),
            'std': np.std(wind_speeds),
            'max': np.max(wind_speeds),
            'min': np.min(wind_speeds),
            'skewness': 0.5,
            'kurtosis': 0.2
        },
        'weibull': {
            'k': 2.2,
            'c': 9.5,
            'mu': 8.4
        },
        'capacity_factor': {
            'capacity_factor': 0.32
        },
        'power_density': {
            'mean_power_density': 450
        }
    }
    
    # Crear visualizador
    viz = WindVisualization(wind_data, analysis_results)
    
    # Generar gráfico de resumen
    print("Generando dashboard de resumen...")
    dashboard = viz.create_summary_dashboard('dashboard_ejemplo.png')
    
    if dashboard:
        print("Dashboard generado exitosamente!")
    else:
        print("Error generando dashboard")

