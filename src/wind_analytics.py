"""
Módulo para análisis estadístico y métricas eólicas avanzadas
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class WindAnalytics:
    """Clase para análisis estadístico avanzado de datos eólicos"""
    
    def __init__(self):
        self.wind_data = None
        self.analysis_results = {}
    
    def load_wind_data(self, wind_speeds, timestamps=None, wind_directions=None, 
                      temperature=None, pressure=None):
        """
        Cargar datos de viento para análisis
        
        Args:
            wind_speeds (array): Velocidades del viento (m/s)
            timestamps (array): Marcas de tiempo
            wind_directions (array): Direcciones del viento (grados)
            temperature (array): Temperatura (°C)
            pressure (array): Presión atmosférica (hPa)
        """
        self.wind_data = {
            'wind_speed': np.array(wind_speeds),
            'timestamps': timestamps,
            'wind_direction': wind_directions,
            'temperature': temperature,
            'pressure': pressure
        }
        
        # Limpiar datos (remover valores negativos o NaN)
        valid_mask = (self.wind_data['wind_speed'] >= 0) & \
                    (~np.isnan(self.wind_data['wind_speed']))
        
        for key in self.wind_data:
            if self.wind_data[key] is not None:
                self.wind_data[key] = np.array(self.wind_data[key])[valid_mask]
    
    def calculate_basic_statistics(self):
        """Calcular estadísticas básicas del viento"""
        wind_speeds = self.wind_data['wind_speed']
        
        stats_dict = {
            'mean': np.mean(wind_speeds),
            'median': np.median(wind_speeds),
            'std': np.std(wind_speeds),
            'min': np.min(wind_speeds),
            'max': np.max(wind_speeds),
            'q25': np.percentile(wind_speeds, 25),
            'q75': np.percentile(wind_speeds, 75),
            'skewness': stats.skew(wind_speeds),
            'kurtosis': stats.kurtosis(wind_speeds),
            'count': len(wind_speeds)
        }
        
        self.analysis_results['basic_stats'] = stats_dict
        return stats_dict
    
    def fit_weibull_distribution(self, method='mle'):
        """
        Ajustar distribución de Weibull a los datos de viento
        
        Args:
            method (str): Método de ajuste ('mle', 'moments', 'lsq')
            
        Returns:
            dict: Parámetros de Weibull (k, c, μ, σ, v_modo)
        """
        wind_speeds = self.wind_data['wind_speed']
        wind_speeds = wind_speeds[wind_speeds > 0]  # Weibull requiere valores positivos
        
        if method == 'mle':
            # Método de máxima verosimilitud
            shape, loc, scale = stats.weibull_min.fit(wind_speeds, floc=0)
            k = shape  # Parámetro de forma
            c = scale  # Parámetro de escala
            
        elif method == 'moments':
            # Método de momentos
            mean_ws = np.mean(wind_speeds)
            std_ws = np.std(wind_speeds)
            
            # Estimación inicial usando método de momentos
            cv = std_ws / mean_ws  # Coeficiente de variación
            
            # Aproximación para k usando coeficiente de variación
            if cv < 0.2:
                k = 1 / (cv**1.086)
            else:
                k = (-0.7 + 1.4 * cv**(-1.4))**(-1)
            
            # Calcular c usando la media
            c = mean_ws / stats.gamma(1 + 1/k)
            
        elif method == 'lsq':
            # Método de mínimos cuadrados
            def weibull_cdf(x, k, c):
                return 1 - np.exp(-(x/c)**k)
            
            # Datos empíricos
            sorted_data = np.sort(wind_speeds)
            n = len(sorted_data)
            empirical_cdf = np.arange(1, n+1) / (n+1)
            
            # Función objetivo
            def objective(params):
                k, c = params
                if k <= 0 or c <= 0:
                    return np.inf
                theoretical_cdf = weibull_cdf(sorted_data, k, c)
                return np.sum((empirical_cdf - theoretical_cdf)**2)
            
            # Optimización
            result = minimize(objective, [2.0, np.mean(wind_speeds)], 
                            method='Nelder-Mead')
            k, c = result.x
        
        # Calcular parámetros adicionales
        mu = c * gamma(1 + 1/k)  # Media teórica
        sigma = c * np.sqrt(gamma(1 + 2/k) - (gamma(1 + 1/k))**2)  # Desviación estándar
        v_modo = c * ((k-1)/k)**(1/k) if k > 1 else 0  # Moda
        
        # Calcular bondad de ajuste
        theoretical_cdf = 1 - np.exp(-(np.sort(wind_speeds)/c)**k)
        empirical_cdf = np.arange(1, len(wind_speeds)+1) / (len(wind_speeds)+1)
        r2 = r2_score(empirical_cdf, theoretical_cdf)
        
        weibull_params = {
            'k': k,  # Parámetro de forma
            'c': c,  # Parámetro de escala
            'mu': mu,  # Media
            'sigma': sigma,  # Desviación estándar
            'v_modo': v_modo,  # Moda
            'r2': r2,  # Bondad de ajuste
            'method': method
        }
        
        self.analysis_results['weibull'] = weibull_params
        return weibull_params
    
    def calculate_turbulence_intensity(self, averaging_period=10):
        """
        Calcular índice de turbulencia (TI)
        
        Args:
            averaging_period (int): Período de promediado en minutos
            
        Returns:
            dict: Estadísticas de turbulencia
        """
        wind_speeds = self.wind_data['wind_speed']
        
        # Si no hay timestamps, asumir datos horarios
        if self.wind_data['timestamps'] is None:
            # Simular períodos de 10 minutos (6 mediciones por hora)
            n_periods = len(wind_speeds) // 6
            periods = np.array_split(wind_speeds[:n_periods*6], n_periods)
        else:
            # Agrupar por período de promediado
            df = pd.DataFrame({
                'wind_speed': wind_speeds,
                'timestamp': self.wind_data['timestamps']
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Resamplear a períodos de promediado
            resampled = df.resample(f'{averaging_period}T')
            periods = [group['wind_speed'].values for name, group in resampled if len(group) > 0]
        
        # Calcular TI para cada período
        ti_values = []
        mean_speeds = []
        
        for period in periods:
            if len(period) > 1:
                mean_speed = np.mean(period)
                std_speed = np.std(period)
                
                if mean_speed > 0:
                    ti = std_speed / mean_speed
                    ti_values.append(ti)
                    mean_speeds.append(mean_speed)
        
        if len(ti_values) == 0:
            return {'error': 'No se pudieron calcular índices de turbulencia'}
        
        ti_stats = {
            'mean_ti': np.mean(ti_values),
            'median_ti': np.median(ti_values),
            'std_ti': np.std(ti_values),
            'max_ti': np.max(ti_values),
            'min_ti': np.min(ti_values),
            'q90_ti': np.percentile(ti_values, 90),
            'q95_ti': np.percentile(ti_values, 95),
            'ti_values': ti_values,
            'mean_speeds': mean_speeds,
            'averaging_period': averaging_period
        }
        
        self.analysis_results['turbulence'] = ti_stats
        return ti_stats
    
    def calculate_wind_power_density(self, air_density=1.225):
        """
        Calcular densidad de potencia eólica
        
        Args:
            air_density (float): Densidad del aire (kg/m³)
            
        Returns:
            dict: Estadísticas de densidad de potencia
        """
        wind_speeds = self.wind_data['wind_speed']
        
        # Densidad de potencia: P = 0.5 * ρ * v³
        power_density = 0.5 * air_density * wind_speeds**3
        
        # Estadísticas
        power_stats = {
            'mean_power_density': np.mean(power_density),
            'median_power_density': np.median(power_density),
            'std_power_density': np.std(power_density),
            'max_power_density': np.max(power_density),
            'total_energy_density': np.sum(power_density),  # Energía total
            'air_density': air_density
        }
        
        # Densidad de potencia usando Weibull si está disponible
        if 'weibull' in self.analysis_results:
            weibull = self.analysis_results['weibull']
            k, c = weibull['k'], weibull['c']
            
            # Densidad de potencia teórica usando Weibull
            theoretical_power = 0.5 * air_density * c**3 * gamma(1 + 3/k)
            power_stats['theoretical_power_density'] = theoretical_power
        
        self.analysis_results['power_density'] = power_stats
        return power_stats
    
    def calculate_wind_rose_statistics(self, direction_bins=16):
        """
        Calcular estadísticas de rosa de vientos
        
        Args:
            direction_bins (int): Número de sectores direccionales
            
        Returns:
            dict: Estadísticas direccionales
        """
        if self.wind_data['wind_direction'] is None:
            return {'error': 'No hay datos de dirección del viento'}
        
        wind_speeds = self.wind_data['wind_speed']
        wind_directions = self.wind_data['wind_direction']
        
        # Crear bins direccionales
        bin_edges = np.linspace(0, 360, direction_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Asignar direcciones a bins
        direction_indices = np.digitize(wind_directions, bin_edges) - 1
        direction_indices = np.clip(direction_indices, 0, direction_bins - 1)
        
        # Calcular estadísticas por sector
        sector_stats = []
        for i in range(direction_bins):
            mask = direction_indices == i
            sector_winds = wind_speeds[mask]
            
            if len(sector_winds) > 0:
                stats_dict = {
                    'direction': bin_centers[i],
                    'frequency': len(sector_winds) / len(wind_speeds),
                    'mean_speed': np.mean(sector_winds),
                    'max_speed': np.max(sector_winds),
                    'power_contribution': np.sum(sector_winds**3) / np.sum(wind_speeds**3)
                }
            else:
                stats_dict = {
                    'direction': bin_centers[i],
                    'frequency': 0,
                    'mean_speed': 0,
                    'max_speed': 0,
                    'power_contribution': 0
                }
            
            sector_stats.append(stats_dict)
        
        # Dirección predominante
        frequencies = [s['frequency'] for s in sector_stats]
        predominant_direction = bin_centers[np.argmax(frequencies)]
        
        wind_rose_stats = {
            'sector_stats': sector_stats,
            'predominant_direction': predominant_direction,
            'direction_bins': direction_bins,
            'bin_centers': bin_centers.tolist()
        }
        
        self.analysis_results['wind_rose'] = wind_rose_stats
        return wind_rose_stats
    
    def calculate_capacity_factor(self, rated_power=2000, cut_in=3, rated_speed=12, cut_out=25):
        """
        Calcular factor de capacidad para un aerogenerador típico
        
        Args:
            rated_power (float): Potencia nominal (kW)
            cut_in (float): Velocidad de arranque (m/s)
            rated_speed (float): Velocidad nominal (m/s)
            cut_out (float): Velocidad de parada (m/s)
            
        Returns:
            dict: Estadísticas de factor de capacidad
        """
        wind_speeds = self.wind_data['wind_speed']
        
        # Curva de potencia simplificada
        power_output = np.zeros_like(wind_speeds)
        
        # Región 1: v < cut_in
        mask1 = wind_speeds < cut_in
        power_output[mask1] = 0
        
        # Región 2: cut_in <= v < rated_speed (cúbica)
        mask2 = (wind_speeds >= cut_in) & (wind_speeds < rated_speed)
        power_output[mask2] = rated_power * ((wind_speeds[mask2] - cut_in) / (rated_speed - cut_in))**3
        
        # Región 3: rated_speed <= v < cut_out (constante)
        mask3 = (wind_speeds >= rated_speed) & (wind_speeds < cut_out)
        power_output[mask3] = rated_power
        
        # Región 4: v >= cut_out
        mask4 = wind_speeds >= cut_out
        power_output[mask4] = 0
        
        # Calcular factor de capacidad
        capacity_factor = np.mean(power_output) / rated_power
        
        # Estadísticas adicionales
        cf_stats = {
            'capacity_factor': capacity_factor,
            'mean_power_output': np.mean(power_output),
            'max_power_output': np.max(power_output),
            'hours_generating': np.sum(power_output > 0) / len(power_output),
            'hours_rated_power': np.sum(power_output == rated_power) / len(power_output),
            'annual_energy': np.sum(power_output) * 8760 / len(power_output),  # kWh/año
            'turbine_specs': {
                'rated_power': rated_power,
                'cut_in': cut_in,
                'rated_speed': rated_speed,
                'cut_out': cut_out
            }
        }
        
        self.analysis_results['capacity_factor'] = cf_stats
        return cf_stats
    
    def calculate_extreme_wind_analysis(self, return_periods=[1, 10, 50, 100]):
        """
        Análisis de vientos extremos usando distribución de Gumbel
        
        Args:
            return_periods (list): Períodos de retorno en años
            
        Returns:
            dict: Velocidades extremas para diferentes períodos de retorno
        """
        wind_speeds = self.wind_data['wind_speed']
        
        # Obtener máximos anuales (simulados)
        # Si no hay timestamps, simular datos anuales
        n_years = max(1, len(wind_speeds) // (365 * 24))
        annual_maxima = []
        
        for i in range(n_years):
            start_idx = i * (len(wind_speeds) // n_years)
            end_idx = (i + 1) * (len(wind_speeds) // n_years)
            annual_max = np.max(wind_speeds[start_idx:end_idx])
            annual_maxima.append(annual_max)
        
        if len(annual_maxima) < 3:
            # Si no hay suficientes años, usar máximos mensuales
            n_months = max(3, len(wind_speeds) // (30 * 24))
            monthly_maxima = []
            for i in range(n_months):
                start_idx = i * (len(wind_speeds) // n_months)
                end_idx = (i + 1) * (len(wind_speeds) // n_months)
                monthly_max = np.max(wind_speeds[start_idx:end_idx])
                monthly_maxima.append(monthly_max)
            annual_maxima = monthly_maxima
        
        # Ajustar distribución de Gumbel
        try:
            loc, scale = stats.gumbel_r.fit(annual_maxima)
            
            # Calcular velocidades extremas para diferentes períodos de retorno
            extreme_winds = {}
            for period in return_periods:
                # Probabilidad de no excedencia
                prob = 1 - 1/period
                extreme_wind = stats.gumbel_r.ppf(prob, loc=loc, scale=scale)
                extreme_winds[f'{period}_year'] = extreme_wind
            
            extreme_stats = {
                'extreme_winds': extreme_winds,
                'gumbel_params': {'location': loc, 'scale': scale},
                'annual_maxima': annual_maxima,
                'n_years_data': len(annual_maxima)
            }
            
        except Exception as e:
            extreme_stats = {
                'error': f'No se pudo realizar análisis de extremos: {str(e)}',
                'annual_maxima': annual_maxima
            }
        
        self.analysis_results['extreme_winds'] = extreme_stats
        return extreme_stats
    
    def generate_comprehensive_report(self):
        """
        Generar reporte comprensivo de todas las métricas calculadas
        
        Returns:
            dict: Reporte completo con todas las métricas
        """
        # Calcular todas las métricas si no se han calculado
        if 'basic_stats' not in self.analysis_results:
            self.calculate_basic_statistics()
        
        if 'weibull' not in self.analysis_results:
            self.fit_weibull_distribution()
        
        if 'turbulence' not in self.analysis_results:
            self.calculate_turbulence_intensity()
        
        if 'power_density' not in self.analysis_results:
            self.calculate_wind_power_density()
        
        if 'capacity_factor' not in self.analysis_results:
            self.calculate_capacity_factor()
        
        if 'extreme_winds' not in self.analysis_results:
            self.calculate_extreme_wind_analysis()
        
        # Calcular métricas adicionales de viabilidad
        basic_stats = self.analysis_results['basic_stats']
        weibull = self.analysis_results['weibull']
        
        # Criterios de viabilidad eólica
        viability_criteria = {
            'mean_wind_speed': basic_stats['mean'],
            'weibull_scale': weibull['c'],
            'weibull_shape': weibull['k'],
            'capacity_factor': self.analysis_results['capacity_factor']['capacity_factor'],
            'power_density': self.analysis_results['power_density']['mean_power_density']
        }
        
        # Clasificación de viabilidad
        if (viability_criteria['mean_wind_speed'] > 7.0 and 
            viability_criteria['capacity_factor'] > 0.25 and
            viability_criteria['power_density'] > 300):
            viability_class = 'alto'
            viability_score = 0.8 + min(0.2, (viability_criteria['mean_wind_speed'] - 7) * 0.05)
        elif (viability_criteria['mean_wind_speed'] > 5.0 and 
              viability_criteria['capacity_factor'] > 0.15 and
              viability_criteria['power_density'] > 150):
            viability_class = 'moderado'
            viability_score = 0.4 + min(0.4, (viability_criteria['mean_wind_speed'] - 5) * 0.1)
        else:
            viability_class = 'bajo'
            viability_score = min(0.4, viability_criteria['mean_wind_speed'] * 0.08)
        
        # Reporte comprensivo
        comprehensive_report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_hours': basic_stats['count'],
                'data_quality': 'good' if basic_stats['count'] > 100 else 'limited'
            },
            'viability_assessment': {
                'class': viability_class,
                'score': viability_score,
                'criteria': viability_criteria
            },
            'statistical_analysis': self.analysis_results,
            'recommendations': self._generate_recommendations(viability_class, viability_criteria)
        }
        
        return comprehensive_report
    
    def _generate_recommendations(self, viability_class, criteria):
        """Generar recomendaciones basadas en el análisis"""
        recommendations = []
        
        if viability_class == 'alto':
            recommendations.append("Excelente recurso eólico. Se recomienda continuar con estudios de factibilidad detallados.")
            recommendations.append("Considerar aerogeneradores de gran escala (>2 MW).")
            
        elif viability_class == 'moderado':
            recommendations.append("Recurso eólico moderado. Evaluar viabilidad económica cuidadosamente.")
            recommendations.append("Considerar aerogeneradores de mediana escala (1-2 MW).")
            
        else:
            recommendations.append("Recurso eólico limitado. No se recomienda para proyectos comerciales.")
            recommendations.append("Podría ser viable para aplicaciones de pequeña escala o autoconsumo.")
        
        # Recomendaciones específicas
        if criteria['mean_wind_speed'] < 4:
            recommendations.append("Velocidad promedio muy baja. Considerar otras ubicaciones.")
        
        if 'turbulence' in self.analysis_results:
            mean_ti = self.analysis_results['turbulence']['mean_ti']
            if mean_ti > 0.2:
                recommendations.append("Alta turbulencia detectada. Evaluar impacto en vida útil de equipos.")
        
        return recommendations

# Función de utilidad para análisis rápido
def quick_wind_analysis(wind_speeds, **kwargs):
    """
    Realizar análisis rápido de datos de viento
    
    Args:
        wind_speeds (array): Velocidades del viento
        **kwargs: Argumentos adicionales para WindAnalytics
        
    Returns:
        dict: Reporte comprensivo
    """
    analyzer = WindAnalytics()
    analyzer.load_wind_data(wind_speeds, **kwargs)
    return analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    # Ejemplo de uso
    np.random.seed(42)
    
    # Simular datos de viento
    n_hours = 8760  # Un año de datos horarios
    wind_speeds = np.random.weibull(2.0, n_hours) * 8.0 + 2.0  # Distribución Weibull
    wind_directions = np.random.uniform(0, 360, n_hours)
    
    # Crear analizador
    analyzer = WindAnalytics()
    analyzer.load_wind_data(wind_speeds, wind_directions=wind_directions)
    
    # Generar reporte
    report = analyzer.generate_comprehensive_report()
    
    print("=== REPORTE DE ANÁLISIS EÓLICO ===")
    print(f"Clase de viabilidad: {report['viability_assessment']['class'].upper()}")
    print(f"Puntuación: {report['viability_assessment']['score']:.2f}")
    print(f"Velocidad promedio: {report['statistical_analysis']['basic_stats']['mean']:.2f} m/s")
    print(f"Factor de capacidad: {report['statistical_analysis']['capacity_factor']['capacity_factor']:.2f}")
    print(f"Parámetros Weibull: k={report['statistical_analysis']['weibull']['k']:.2f}, c={report['statistical_analysis']['weibull']['c']:.2f}")

