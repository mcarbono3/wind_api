"""
Sistema de diagnóstico automático con IA para evaluación del potencial eólico
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class WindPotentialAI:
    """Sistema de IA para diagnóstico automático del potencial eólico"""
    
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'mean_wind_speed', 'std_wind_speed', 'max_wind_speed',
            'weibull_k', 'weibull_c', 'weibull_mu',
            'turbulence_intensity', 'power_density',
            'capacity_factor', 'wind_speed_variability'
        ]
        self.is_trained = False
        
    def generate_training_data(self, n_samples=1000):
        """
        Generar datos de entrenamiento sintéticos basados en conocimiento experto
        
        Args:
            n_samples (int): Número de muestras a generar
            
        Returns:
            tuple: (X, y_class, y_score) - características, clases, puntuaciones
        """
        np.random.seed(42)
        
        # Generar características sintéticas
        data = []
        labels_class = []
        labels_score = []
        
        for i in range(n_samples):
            # Generar diferentes tipos de sitios eólicos
            site_type = np.random.choice(['excellent', 'good', 'moderate', 'poor'], 
                                       p=[0.15, 0.25, 0.35, 0.25])
            
            if site_type == 'excellent':
                # Sitios excelentes (offshore, montañas ventosas)
                mean_wind = np.random.normal(9.5, 1.5)
                std_wind = np.random.normal(3.0, 0.5)
                max_wind = mean_wind + np.random.normal(15, 3)
                weibull_k = np.random.normal(2.2, 0.3)
                weibull_c = mean_wind / 0.887  # Aproximación
                turbulence = np.random.normal(0.12, 0.03)
                label_class = 'alto'
                label_score = np.random.uniform(0.8, 1.0)
                
            elif site_type == 'good':
                # Sitios buenos (costas, colinas)
                mean_wind = np.random.normal(7.5, 1.0)
                std_wind = np.random.normal(2.5, 0.4)
                max_wind = mean_wind + np.random.normal(12, 2)
                weibull_k = np.random.normal(2.0, 0.3)
                weibull_c = mean_wind / 0.887
                turbulence = np.random.normal(0.15, 0.04)
                label_class = 'alto'
                label_score = np.random.uniform(0.6, 0.8)
                
            elif site_type == 'moderate':
                # Sitios moderados (llanuras, áreas rurales)
                mean_wind = np.random.normal(5.5, 1.0)
                std_wind = np.random.normal(2.0, 0.3)
                max_wind = mean_wind + np.random.normal(10, 2)
                weibull_k = np.random.normal(1.8, 0.3)
                weibull_c = mean_wind / 0.887
                turbulence = np.random.normal(0.18, 0.05)
                label_class = 'moderado'
                label_score = np.random.uniform(0.3, 0.6)
                
            else:  # poor
                # Sitios pobres (áreas urbanas, valles protegidos)
                mean_wind = np.random.normal(3.5, 0.8)
                std_wind = np.random.normal(1.5, 0.3)
                max_wind = mean_wind + np.random.normal(8, 1.5)
                weibull_k = np.random.normal(1.5, 0.3)
                weibull_c = mean_wind / 0.887
                turbulence = np.random.normal(0.25, 0.08)
                label_class = 'bajo'
                label_score = np.random.uniform(0.0, 0.3)
            
            # Asegurar valores físicamente realistas
            mean_wind = max(0.5, mean_wind)
            std_wind = max(0.1, std_wind)
            max_wind = max(mean_wind + 1, max_wind)
            weibull_k = max(0.5, min(5.0, weibull_k))
            weibull_c = max(0.5, weibull_c)
            turbulence = max(0.05, min(0.5, turbulence))
            
            # Calcular métricas derivadas
            weibull_mu = weibull_c * np.exp(np.log(np.pi/2) / weibull_k)  # Aproximación
            power_density = 0.5 * 1.225 * mean_wind**3  # W/m²
            
            # Factor de capacidad simplificado
            if mean_wind < 3:
                capacity_factor = 0
            elif mean_wind < 12:
                capacity_factor = ((mean_wind - 3) / 9)**3 * 0.45
            else:
                capacity_factor = 0.45
            
            # Variabilidad del viento
            wind_variability = std_wind / mean_wind
            
            # Crear vector de características
            features = [
                mean_wind, std_wind, max_wind,
                weibull_k, weibull_c, weibull_mu,
                turbulence, power_density,
                capacity_factor, wind_variability
            ]
            
            data.append(features)
            labels_class.append(label_class)
            labels_score.append(label_score)
        
        X = np.array(data)
        y_class = np.array(labels_class)
        y_score = np.array(labels_score)
        
        return X, y_class, y_score
    
    def train_models(self, X=None, y_class=None, y_score=None):
        """
        Entrenar los modelos de clasificación y regresión
        
        Args:
            X (array): Características (si None, genera datos sintéticos)
            y_class (array): Etiquetas de clase
            y_score (array): Puntuaciones continuas
        """
        if X is None:
            print("Generando datos de entrenamiento sintéticos...")
            X, y_class, y_score = self.generate_training_data(1500)
        
        # Dividir datos
        X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = \
            train_test_split(X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class)
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar clasificador (para categorías: alto, moderado, bajo)
        print("Entrenando clasificador...")
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.classifier.fit(X_train_scaled, y_class_train)
        
        # Evaluar clasificador
        y_pred_class = self.classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_class_test, y_pred_class)
        print(f"Precisión del clasificador: {accuracy:.3f}")
        
        # Entrenar regresor (para puntuación continua)
        print("Entrenando regresor...")
        self.regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.regressor.fit(X_train_scaled, y_score_train)
        
        # Evaluar regresor
        y_pred_score = self.regressor.predict(X_test_scaled)
        mse = np.mean((y_score_test - y_pred_score)**2)
        print(f"Error cuadrático medio del regresor: {mse:.4f}")
        
        self.is_trained = True
        print("Modelos entrenados exitosamente!")
        
        # Mostrar importancia de características
        self._show_feature_importance()
    
    def _show_feature_importance(self):
        """Mostrar la importancia de las características"""
        if self.classifier is not None:
            importance = self.classifier.feature_importances_
            feature_importance = list(zip(self.feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("\nImportancia de características (Clasificador):")
            for feature, imp in feature_importance[:5]:
                print(f"  {feature}: {imp:.3f}")
    
    def predict_wind_potential(self, wind_metrics):
        """
        Predecir el potencial eólico usando los modelos entrenados
        
        Args:
            wind_metrics (dict): Métricas del viento calculadas
            
        Returns:
            dict: Predicción con clase, puntuación y confianza
        """
        if not self.is_trained:
            print("Entrenando modelos...")
            self.train_models()
        
        # Extraer características del diccionario de métricas
        features = self._extract_features(wind_metrics)
        
        if features is None:
            return {
                'error': 'No se pudieron extraer características suficientes',
                'required_metrics': self.feature_names
            }
        
        # Escalar características
        features_scaled = self.scaler.transform([features])
        
        # Predicción de clase
        predicted_class = self.classifier.predict(features_scaled)[0]
        class_probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Predicción de puntuación
        predicted_score = self.regressor.predict(features_scaled)[0]
        predicted_score = max(0.0, min(1.0, predicted_score))  # Limitar entre 0 y 1
        
        # Calcular confianza
        max_prob = np.max(class_probabilities)
        confidence = max_prob
        
        # Generar diagnóstico detallado
        diagnosis = self._generate_detailed_diagnosis(
            wind_metrics, predicted_class, predicted_score, confidence
        )
        
        return {
            'predicted_class': predicted_class,
            'predicted_score': predicted_score,
            'confidence': confidence,
            'class_probabilities': {
                'alto': class_probabilities[0] if len(class_probabilities) > 0 else 0,
                'bajo': class_probabilities[1] if len(class_probabilities) > 1 else 0,
                'moderado': class_probabilities[2] if len(class_probabilities) > 2 else 0
            },
            'diagnosis': diagnosis
        }
    
    def _extract_features(self, wind_metrics):
        """Extraer características del diccionario de métricas"""
        try:
            # Estadísticas básicas
            basic_stats = wind_metrics.get('basic_stats', {})
            mean_wind = basic_stats.get('mean', 0)
            std_wind = basic_stats.get('std', 0)
            max_wind = basic_stats.get('max', 0)
            
            # Parámetros de Weibull
            weibull = wind_metrics.get('weibull', {})
            weibull_k = weibull.get('k', 1.5)
            weibull_c = weibull.get('c', mean_wind / 0.887 if mean_wind > 0 else 1.0)
            weibull_mu = weibull.get('mu', mean_wind)
            
            # Turbulencia
            turbulence = wind_metrics.get('turbulence', {})
            turbulence_intensity = turbulence.get('mean_ti', 0.2)
            
            # Densidad de potencia
            power_density_data = wind_metrics.get('power_density', {})
            power_density = power_density_data.get('mean_power_density', 0)
            
            # Factor de capacidad
            capacity_data = wind_metrics.get('capacity_factor', {})
            capacity_factor = capacity_data.get('capacity_factor', 0)
            
            # Variabilidad del viento
            wind_variability = std_wind / mean_wind if mean_wind > 0 else 1.0
            
            features = [
                mean_wind, std_wind, max_wind,
                weibull_k, weibull_c, weibull_mu,
                turbulence_intensity, power_density,
                capacity_factor, wind_variability
            ]
            
            # Verificar que todas las características son números válidos
            if any(not isinstance(f, (int, float)) or np.isnan(f) for f in features):
                return None
            
            return features
            
        except Exception as e:
            print(f"Error extrayendo características: {e}")
            return None
    
    def _generate_detailed_diagnosis(self, wind_metrics, predicted_class, predicted_score, confidence):
        """Generar diagnóstico detallado basado en las métricas"""
        
        # Extraer métricas clave
        basic_stats = wind_metrics.get('basic_stats', {})
        weibull = wind_metrics.get('weibull', {})
        turbulence = wind_metrics.get('turbulence', {})
        capacity_data = wind_metrics.get('capacity_factor', {})
        
        mean_wind = basic_stats.get('mean', 0)
        weibull_k = weibull.get('k', 0)
        weibull_c = weibull.get('c', 0)
        mean_ti = turbulence.get('mean_ti', 0)
        capacity_factor = capacity_data.get('capacity_factor', 0)
        
        # Generar diagnóstico
        diagnosis = {
            'summary': self._get_summary_message(predicted_class, predicted_score),
            'detailed_analysis': [],
            'recommendations': [],
            'risk_factors': [],
            'opportunities': []
        }
        
        # Análisis detallado
        if mean_wind > 8:
            diagnosis['detailed_analysis'].append(
                f"Excelente recurso eólico con velocidad promedio de {mean_wind:.1f} m/s"
            )
        elif mean_wind > 6:
            diagnosis['detailed_analysis'].append(
                f"Buen recurso eólico con velocidad promedio de {mean_wind:.1f} m/s"
            )
        elif mean_wind > 4:
            diagnosis['detailed_analysis'].append(
                f"Recurso eólico moderado con velocidad promedio de {mean_wind:.1f} m/s"
            )
        else:
            diagnosis['detailed_analysis'].append(
                f"Recurso eólico limitado con velocidad promedio de {mean_wind:.1f} m/s"
            )
        
        # Análisis de Weibull
        if weibull_k > 2.5:
            diagnosis['detailed_analysis'].append(
                f"Distribución de vientos muy consistente (k={weibull_k:.2f})"
            )
        elif weibull_k > 2.0:
            diagnosis['detailed_analysis'].append(
                f"Distribución de vientos consistente (k={weibull_k:.2f})"
            )
        else:
            diagnosis['detailed_analysis'].append(
                f"Distribución de vientos variable (k={weibull_k:.2f})"
            )
        
        # Análisis de turbulencia
        if mean_ti < 0.15:
            diagnosis['detailed_analysis'].append(
                f"Baja turbulencia ({mean_ti:.2f}) - favorable para aerogeneradores"
            )
        elif mean_ti < 0.20:
            diagnosis['detailed_analysis'].append(
                f"Turbulencia moderada ({mean_ti:.2f}) - aceptable para la mayoría de aerogeneradores"
            )
        else:
            diagnosis['detailed_analysis'].append(
                f"Alta turbulencia ({mean_ti:.2f}) - puede afectar la vida útil de equipos"
            )
            diagnosis['risk_factors'].append("Alta turbulencia puede reducir vida útil de aerogeneradores")
        
        # Análisis de factor de capacidad
        if capacity_factor > 0.35:
            diagnosis['detailed_analysis'].append(
                f"Excelente factor de capacidad ({capacity_factor:.2f})"
            )
            diagnosis['opportunities'].append("Alto factor de capacidad permite excelente retorno de inversión")
        elif capacity_factor > 0.25:
            diagnosis['detailed_analysis'].append(
                f"Buen factor de capacidad ({capacity_factor:.2f})"
            )
        elif capacity_factor > 0.15:
            diagnosis['detailed_analysis'].append(
                f"Factor de capacidad moderado ({capacity_factor:.2f})"
            )
        else:
            diagnosis['detailed_analysis'].append(
                f"Factor de capacidad bajo ({capacity_factor:.2f})"
            )
            diagnosis['risk_factors'].append("Bajo factor de capacidad puede afectar viabilidad económica")
        
        # Recomendaciones basadas en la clase predicha
        if predicted_class == 'alto':
            diagnosis['recommendations'].extend([
                "Proceder con estudios de factibilidad detallados",
                "Considerar aerogeneradores de gran escala (>2 MW)",
                "Evaluar conexión a red eléctrica",
                "Realizar mediciones in-situ para validar datos"
            ])
            
        elif predicted_class == 'moderado':
            diagnosis['recommendations'].extend([
                "Evaluar viabilidad económica cuidadosamente",
                "Considerar aerogeneradores de mediana escala (1-2 MW)",
                "Analizar incentivos gubernamentales disponibles",
                "Estudiar opciones de almacenamiento de energía"
            ])
            
        else:  # bajo
            diagnosis['recommendations'].extend([
                "No recomendado para proyectos comerciales de gran escala",
                "Evaluar aplicaciones de pequeña escala o autoconsumo",
                "Considerar ubicaciones alternativas",
                "Explorar tecnologías de baja velocidad de viento"
            ])
        
        # Oportunidades adicionales
        if mean_wind > 7 and mean_ti < 0.18:
            diagnosis['opportunities'].append("Condiciones ideales para tecnología eólica moderna")
        
        if weibull_k > 2.2:
            diagnosis['opportunities'].append("Vientos consistentes permiten predicción energética confiable")
        
        return diagnosis
    
    def _get_summary_message(self, predicted_class, predicted_score):
        """Generar mensaje de resumen"""
        if predicted_class == 'alto':
            if predicted_score > 0.8:
                return "✅ Excelente potencial eólico - Altamente recomendado para desarrollo"
            else:
                return "✅ Alto potencial eólico - Recomendado para desarrollo"
        elif predicted_class == 'moderado':
            return "⚠️ Potencial eólico moderado - Requiere análisis económico detallado"
        else:
            return "❌ Potencial eólico limitado - No recomendado para proyectos comerciales"
    
    def save_models(self, filepath_prefix='wind_ai_models'):
        """Guardar los modelos entrenados"""
        if self.is_trained:
            joblib.dump(self.classifier, f'{filepath_prefix}_classifier.pkl')
            joblib.dump(self.regressor, f'{filepath_prefix}_regressor.pkl')
            joblib.dump(self.scaler, f'{filepath_prefix}_scaler.pkl')
            print(f"Modelos guardados con prefijo: {filepath_prefix}")
        else:
            print("Los modelos no han sido entrenados aún")
    
    def load_models(self, filepath_prefix='wind_ai_models'):
        """Cargar modelos previamente entrenados"""
        try:
            self.classifier = joblib.load(f'{filepath_prefix}_classifier.pkl')
            self.regressor = joblib.load(f'{filepath_prefix}_regressor.pkl')
            self.scaler = joblib.load(f'{filepath_prefix}_scaler.pkl')
            self.is_trained = True
            print(f"Modelos cargados desde: {filepath_prefix}")
        except FileNotFoundError:
            print("No se encontraron modelos guardados. Entrenando nuevos modelos...")
            self.train_models()

# Función de utilidad para diagnóstico rápido
def quick_ai_diagnosis(wind_metrics):
    """
    Realizar diagnóstico rápido con IA
    
    Args:
        wind_metrics (dict): Métricas del viento
        
    Returns:
        dict: Diagnóstico completo
    """
    ai_system = WindPotentialAI()
    return ai_system.predict_wind_potential(wind_metrics)

if __name__ == "__main__":
    # Ejemplo de uso
    print("=== SISTEMA DE DIAGNÓSTICO IA PARA POTENCIAL EÓLICO ===")
    
    # Crear sistema de IA
    ai_system = WindPotentialAI()
    
    # Entrenar modelos
    ai_system.train_models()
    
    # Ejemplo de métricas de viento
    example_metrics = {
        'basic_stats': {
            'mean': 8.5,
            'std': 3.2,
            'max': 18.5,
            'min': 0.5
        },
        'weibull': {
            'k': 2.3,
            'c': 9.6,
            'mu': 8.5
        },
        'turbulence': {
            'mean_ti': 0.16
        },
        'power_density': {
            'mean_power_density': 450
        },
        'capacity_factor': {
            'capacity_factor': 0.32
        }
    }
    
    # Realizar diagnóstico
    diagnosis = ai_system.predict_wind_potential(example_metrics)
    
    print(f"\nClase predicha: {diagnosis['predicted_class'].upper()}")
    print(f"Puntuación: {diagnosis['predicted_score']:.2f}")
    print(f"Confianza: {diagnosis['confidence']:.2f}")
    print(f"\nResumen: {diagnosis['diagnosis']['summary']}")
    
    print("\nAnálisis detallado:")
    for analysis in diagnosis['diagnosis']['detailed_analysis']:
        print(f"  • {analysis}")
    
    print("\nRecomendaciones:")
    for rec in diagnosis['diagnosis']['recommendations']:
        print(f"  • {rec}")

