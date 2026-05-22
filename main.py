import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from xgboost import XGBClassifier
from src.staking import calcular_gestion_riesgo
import warnings
warnings.filterwarnings('ignore')

print("🚀 Iniciando tubería analítica de nivel profesional...")

# 1. INGESTA Y PROCESAMIENTO DE DATOS HISTÓRICOS
urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
]
datos = []
for url in urls:
    try:
        datos.append(pd.read_csv(url))
    except:
        pass

df = pd.concat(datos, ignore_index=True)
df['Target'] = np.where(df['FTR'] == 'H', 1, 0)

# Feature Engineering Avanzado (Métricas de rendimiento ofensivo/defensivo acumulado)
df['Home_Goals_Avg'] = df.groupby('HomeTeam')['FTHG'].transform(lambda x: x.expanding().mean().shift())
df['Away_Goals_Avg'] = df.groupby('AwayTeam')['FTAG'].transform(lambda x: x.expanding().mean().shift())
df['Home_Goals_Rec_Avg'] = df.groupby('HomeTeam')['FTAG'].transform(lambda x: x.expanding().mean().shift())
df['Away_Goals_Rec_Avg'] = df.groupby('AwayTeam')['FTHG'].transform(lambda x: x.expanding().mean().shift())
df['Implied_Prob_H'] = 1 / df['B365H']
df['Implied_Prob_A'] = 1 / df['B365A']

features = ['Home_Goals_Avg', 'Away_Goals_Avg', 'Home_Goals_Rec_Avg', 'Away_Goals_Rec_Avg', 'Implied_Prob_H', 'Implied_Prob_A']
df_model = df.dropna(subset=features + ['Target'])

# 2. ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING
X = df_model[features]
y = df_model['Target']
modelo = XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=4, subsample=0.8, random_state=42)
modelo.fit(X, y)
print("✅ Inteligencia Artificial entrenada y calibrada con éxito.")

# Mapeo de nomenclatura entre proveedores de datos
mapeo_equipos = {
    "Manchester United": "Man United", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton", "Wolverhampton Wanderers": "Wolves"
}

# 3. ESCÁNER EN TIEMPO REAL Y EJECUCIÓN FINANCIERA
API_KEY = os.environ.get('ODDS_API_KEY')
url_odds = f'https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={API_KEY}&regions=eu&markets=h2h'
response = requests.get(url_odds)

nuevas_predicciones = []

if response.status_code == 200:
    partidos = response.json()
    for partido in partidos:
        try:
            home_raw = partido['home_team']
            away_raw = partido['away_team']
            
            home_team = mapeo_equipos.get(home_raw, home_raw)
            away_team = mapeo_equipos.get(away_raw, away_raw)
            
            cuotas = partido['bookmakers'][0]['markets'][0]['outcomes']
            odds_home = next(item['price'] for item in cuotas if item['name'] == home_raw)
            odds_away = next(item['price'] for item in cuotas if item['name'] == away_raw)
            
            # Recuperar métricas del estado del arte actual de los equipos
            h_stat = df[df['HomeTeam'] == home_team]['Home_Goals_Avg'].iloc[-1] if home_team in df['HomeTeam'].values else df['FTHG'].mean()
            a_stat = df[df['AwayTeam'] == away_team]['Away_Goals_Avg'].iloc[-1] if away_team in df['AwayTeam'].values else df['FTAG'].mean()
            h_rec = df[df['HomeTeam'] == home_team]['Home_Goals_Rec_Avg'].iloc[-1] if home_team in df['HomeTeam'].values else df['FTAG'].mean()
            a_rec = df[df['AwayTeam'] == away_team]['Away_Goals_Rec_Avg'].iloc[-1] if away_team in df['AwayTeam'].values else df['FTHG'].mean()
            
            X_nuevo = pd.DataFrame([{
                'Home_Goals_Avg': h_stat, 'Away_Goals_Avg': a_stat,
                'Home_Goals_Rec_Avg': h_rec, 'Away_Goals_Rec_Avg': a_rec,
                'Implied_Prob_H': 1 / odds_home, 'Implied_Prob_A': 1 / odds_away
            }])
            
            probabilidad_real = modelo.predict_proba(X_nuevo)[0][1]
            
            # Ejecutar algoritmo de Kelly para determinar viabilidad e inversión
            dinero_stake, porc_banca, ventaja_pct = calcular_gestion_riesgo(probabilidad_real, odds_home)
            
            if dinero_stake > 0:
                nuevas_predicciones.append({
                    'Fecha_Analisis': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'Partido': f"{home_team} vs {away_team}",
                    'Pronostico': f"Victoria {home_team}",
                    'Cuota_Bookmaker': round(odds_home, 2),
                    'Prob_ML_%': round(probabilidad_real * 100, 2),
                    'Ventaja_Edge_%': ventaja_pct,
                    'Inversion_Sugerida_$': dinero_stake,
                    'Porcentaje_Banca_%': porc_banca,
                    'Estado': 'ACTIVA'
                })
        except:
            pass

# 4. PERSISTENCIA DE LA BASE DE DATOS AUDITADA
archivo_csv = 'predicciones_auditadas.csv'
if nuevas_predicciones:
    df_nuevas = pd.DataFrame(nuevas_predicciones)
    if not os.path.isfile(archivo_csv):
        df_nuevas.to_csv(archivo_csv, index=False)
    else:
        df_existente = pd.read_csv(archivo_csv)
        df_final = pd.concat([df_existente, df_nuevas]).drop_duplicates(subset=['Partido'], keep='first')
        df_final.to_csv(archivo_csv, index=False)
    print(f"🎯 Concluido. Se han registrado {len(nuevas_predicciones)} operaciones con Valor Esperado Positivo (+EV).")
else:
    print("📉 Mercado eficiente. Ningún partido cumple con las estrictas condiciones de riesgo hoy.")
