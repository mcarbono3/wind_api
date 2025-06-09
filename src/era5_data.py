"""
Módulo para acceso y simulación de datos ERA5 para la región Caribe de Colombia
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import os

class ERA5DataAccess:
    """Clase para acceder a datos ERA5 del ECMWF"""
    
    def __init__(self):
        # Coordenadas de la región Caribe de Colombia
        self.lat_min = 8.0   # Latitud mínima
        self.lat_max = 15.0  # Latitud máxima
        self.lon_min = -82.0 # Longitud mínima
        self.lon_max = -70.0 # Longitud máxima
        
        # Variables meteorológicas disponibles
        self.variables = {
            'u10': '10m u-component of wind',
            'v10': '10m v-component of wind', 
            'u100': '100m u-component of wind',
            'v100': '100m v-component of wind',
            'sp': 'Surface pressure',
            't2m': '2m temperature',
            'msl': 'Mean sea level pressure'
        }
    
    def simulate_era5_data(self, start_date, end_date, variables=None):
        """
        Simula datos ERA5 para desarrollo y pruebas
        
        Args:
            start_date (str): Fecha de inicio (YYYY-MM-DD)
            end_date (str): Fecha de fin (YYYY-MM-DD)
            variables (list): Lista de variables a simular
            
        Returns:
            xarray.Dataset: Dataset con datos simulados
        """
        if variables is None:
            variables = list(self.variables.keys())
        
        # Crear grilla espacial
        lats = np.arange(self.lat_min, self.lat_max + 0.25, 0.25)
        lons = np.arange(self.lon_min, self.lon_max + 0.25, 0.25)
        
        # Crear serie temporal horaria
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        times = pd.date_range(start, end, freq='H')
        
        # Crear dataset vacío
        data_vars = {}
        
        for var in variables:
            if var in ['u10', 'u100']:
                # Componente u del viento (este-oeste)
                # Simular vientos alisios del este
                base_wind = -5.0 + 3.0 * np.sin(2 * np.pi * np.arange(len(times)) / (24 * 365))
                data = np.random.normal(base_wind[:, np.newaxis, np.newaxis], 2.0, 
                                      (len(times), len(lats), len(lons)))
                
            elif var in ['v10', 'v100']:
                # Componente v del viento (norte-sur)
                base_wind = 2.0 + 1.5 * np.sin(2 * np.pi * np.arange(len(times)) / (24 * 365))
                data = np.random.normal(base_wind[:, np.newaxis, np.newaxis], 1.5,
                                      (len(times), len(lats), len(lons)))
                
            elif var == 'sp':
                # Presión superficial (Pa)
                base_pressure = 101325.0
                data = np.random.normal(base_pressure, 500.0,
                                      (len(times), len(lats), len(lons)))
                
            elif var == 't2m':
                # Temperatura a 2m (K)
                base_temp = 298.15  # ~25°C
                seasonal_var = 3.0 * np.sin(2 * np.pi * np.arange(len(times)) / (24 * 365))
                daily_var = 5.0 * np.sin(2 * np.pi * np.arange(len(times)) / 24)
                temp_pattern = base_temp + seasonal_var + daily_var
                data = np.random.normal(temp_pattern[:, np.newaxis, np.newaxis], 2.0,
                                      (len(times), len(lats), len(lons)))
                
            elif var == 'msl':
                # Presión a nivel del mar (Pa)
                base_pressure = 101325.0
                data = np.random.normal(base_pressure, 300.0,
                                      (len(times), len(lats), len(lons)))
            
            data_vars[var] = (['time', 'latitude', 'longitude'], data)
        
        # Crear dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                'time': times,
                'latitude': lats,
                'longitude': lons
            }
        )
        
        # Añadir atributos
        ds.attrs['title'] = 'Simulated ERA5 data for Caribbean Colombia'
        ds.attrs['source'] = 'Simulated for development purposes'
        
        return ds
    
    def download_era5_data(self, start_date, end_date, variables=None, area=None):
        """
        Descarga datos reales de ERA5 (requiere credenciales CDS)
        
        Args:
            start_date (str): Fecha de inicio
            end_date (str): Fecha de fin
            variables (list): Variables a descargar
            area (list): [lat_max, lon_min, lat_min, lon_max]
            
        Returns:
            str: Ruta del archivo descargado
        """
        try:
            import cdsapi
            
            c = cdsapi.Client()
            
            if area is None:
                area = [self.lat_max, self.lon_min, self.lat_min, self.lon_max]
            
            if variables is None:
                variables = ['10m_u_component_of_wind', '10m_v_component_of_wind',
                           '100m_u_component_of_wind', '100m_v_component_of_wind',
                           'surface_pressure', '2m_temperature']
            
            filename = f'era5_data_{start_date}_{end_date}.nc'
            
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': variables,
                    'date': f'{start_date}/{end_date}',
                    'time': [f'{i:02d}:00' for i in range(24)],
                    'area': area,
                    'format': 'netcdf',
                },
                filename
            )
            
            return filename
            
        except Exception as e:
            print(f"Error descargando datos ERA5: {e}")
            print("Usando datos simulados para desarrollo...")
            return None
    
    def get_wind_speed(self, u_component, v_component):
        """
        Calcula la velocidad del viento a partir de componentes u y v
        
        Args:
            u_component (array): Componente u del viento
            v_component (array): Componente v del viento
            
        Returns:
            array: Velocidad del viento
        """
        return np.sqrt(u_component**2 + v_component**2)
    
    def get_wind_direction(self, u_component, v_component):
        """
        Calcula la dirección del viento a partir de componentes u y v
        
        Args:
            u_component (array): Componente u del viento
            v_component (array): Componente v del viento
            
        Returns:
            array: Dirección del viento en grados
        """
        direction = np.arctan2(-u_component, -v_component) * 180 / np.pi
        direction = (direction + 360) % 360
        return direction

if __name__ == "__main__":
    # Prueba del módulo
    era5 = ERA5DataAccess()
    
    # Simular datos para una semana
    data = era5.simulate_era5_data('2024-01-01', '2024-01-07')
    
    print("Dataset simulado creado:")
    print(data)
    
    # Calcular velocidad del viento
    wind_speed = era5.get_wind_speed(data['u10'], data['v10'])
    print(f"\nVelocidad del viento promedio: {wind_speed.mean().values:.2f} m/s")

