import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from xgboost import XGBClassifier
from src.staking import calcular_gestion_riesgo
import warnings
warnings.filterwarnings('ignore')

print("🚀 Escaneando el mercado global en busca del TOP 10...")

# 1. INGESTA HISTÓRICA (Base de conocimiento global)
urls = [
    "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", "https://www.football-data.co.uk/mmz4281/2526/I1.csv"
]
datos = [pd.read_csv(u) for u in urls if requests.head(u).status_code == 200]
df = pd.concat(datos, ignore_index=True)

# Feature Engineering (xG Proxy)
df['H_xG_c'] = df.groupby('HomeTeam')['HS'].transform(lambda x: x.expanding().mean().shift())
df['A_xG_c'] = df.groupby('AwayTeam')['AS'].transform(lambda x: x.expanding().mean().shift())
X = df.dropna(subset=['H_xG_c', 'A_xG_c', 'FTR'])[['H_xG_c', 'A_xG_c']]
y = np.where(df.dropna(subset=['H_xG_c', 'A_xG_c', 'FTR'])['FTR'] == 'H', 1, 0)

modelo = XGBClassifier().fit(X, y)

# 2. ESCÁNER GLOBAL (Solicitud única a la API)
API_KEY = os.environ.get('ODDS_API_KEY')
# Usamos 'soccer_soccer' para obtener todas las ligas disponibles de una vez
url = f'https://api.the-odds-api.com/v4/sports/soccer_soccer/odds/?apiKey={API_KEY}&regions=eu&markets=h2h&oddsFormat=decimal'
response = requests.get(url)

oportunidades = []

if response.status_code == 200:
    for partido in response.json():
        try:
            # Calcular probabilidad con modelo
            prob = modelo.predict_proba([[np.mean(df['H_xG_c']), np.mean(df['A_xG_c'])]])[0][1]
            
            for bookmaker in partido['bookmakers']:
                for outcome in bookmaker['markets'][0]['outcomes']:
                    if outcome['name'] == partido['home_team']:
                        cuota = outcome['price']
                        stake, porc, edge = calcular_gestion_riesgo(prob, cuota)
                        
                        if edge > 0: # Si hay valor matemático
                            oportunidades.append({
                                'Partido': f"{partido['home_team']} vs {partido['away_team']}",
                                'Liga': partido['sport_key'],
                                'Cuota': cuota,
                                'Ventaja_Edge_%': edge,
                                'Inversion_$': stake,
                                'Pronostico': 'Local'
                            })
        except: continue

# 3. FILTRADO FINAL (EL TOP 10 MÁS RENTABLE)
if oportunidades:
    df_top = pd.DataFrame(oportunidades)
    # Ordenar por Edge y cortar los 10 primeros
    df_top = df_top.sort_values(by='Ventaja_Edge_%', ascending=False).head(10)
    df_top.to_csv('predicciones_auditadas.csv', index=False)
    print("✅ Top 10 Global guardado.")
else:
    print("📉 Mercado eficiente. No hay oportunidades top hoy.")
