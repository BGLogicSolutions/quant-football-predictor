import json
import os

def calcular_gestion_riesgo(probabilidad_modelo, cuota_mercado):
    """
    Aplica el algoritmo del Criterio de Kelly Fraccionario para determinar
    el tamaño exacto de la apuesta (stake) protegiendo la banca.
    """
    # Cargar configuraciones de riesgo
    ruta_config = os.path.join(os.path.dirname(__file__), '../config/settings.json')
    with open(ruta_config, 'r') as f:
        config = json.load(f)
        
    bankroll = config['bankroll_inicial']
    f_kelly = config['fraccion_kelly']
    min_edge = config['minima_ventaja_requerida'] / 100
    max_stake = config['maximo_stake_permitido_por_partido'] / 100

    prob_mercado = 1 / cuota_mercado
    ventaja = probabilidad_modelo - prob_mercado

    # Si no hay ventaja matemática o no supera el umbral mínimo, no se apuesta
    if ventaja < min_edge:
        return 0.0, 0.0, ventaja * 100

    # Fórmula estándar del Criterio de Kelly
    # kelly_bruto = (p * b - q) / b -> Equivalente a: (p * cuota - 1) / (cuota - 1)
    kelly_bruto = (probabilidad_modelo * cuota_mercado - 1) / (cuota_mercado - 1)
    
    # Aplicar fracción de seguridad (Fractional Kelly) para mitigar varianza
    kelly_sugerido = kelly_bruto * f_kelly

    # Aplicar límites máximos de exposición por partido (Control de pérdidas de fondos)
    if kelly_sugerido > max_stake:
        kelly_sugerido = max_stake
        
    if kelly_sugerido < 0:
        kelly_sugerido = 0.0

    dinero_a_apostar = bankroll * kelly_sugerido
    porcentaje_banca = kelly_sugerido * 100

    return round(dinero_a_apostar, 2), round(porcentaje_banca, 2), round(ventaja * 100, 2)
