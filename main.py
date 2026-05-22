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

print("🤖 Iniciando Escáner Diario Top 10 Value Bets...")

# ==========================================
# 1. INGESTA Y BUCLE DE RETROALIMENTACIÓN
# ==========================================
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
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

archivo_csv = 'predicciones_auditadas.csv'
if os.path.isfile(archivo_csv):
    df_audit = pd.read_csv(archivo_csv)
    if 'Estado' in df_audit.columns and not df_audit[df_audit['Estado'] == 'PENDIENTE'].empty:
        for idx, fila in df_audit[df_audit['Estado'] == 'PENDIENTE'].iterrows():
            partido_real = df[(df['HomeTeam'] == fila['Local']) & (df['AwayTeam'] == fila['Visitante'])]
            if not partido_real.empty:
                res = partido_real.iloc[-1]
                marcador = f"{int(res['FTHG'])}-{int(res['FTAG'])}"
                hit = (fila['Pronostico'] == "Local" and res['FTR'] == 'H') or \
                      (fila['Pronostico'] == "Ambos Anotan" and int(res['FTHG']) > 0 and int(res['FTAG']) > 0)
                status_final = "GANADO" if hit else "PERDIDO"
                inversion = float(fila['Inversion_Sugerida_$'])
                retorno = round((float(fila['Cuota_Bookmaker']) - 1) * inversion, 2) if hit else -inversion
                
                df_audit.at[idx, 'Resultado_Real'] = marcador
                df_audit.at[idx, 'Estado'] = status_final
                df_audit.at[idx, 'Rendimiento_Neto_$'] = retorno
        df_audit.to_csv(archivo_csv, index=False)

# ==========================================
# 2. FEATURE ENGINEERING Y ENTRENAMIENTO IA
# ==========================================
df['Home_xG_Proxy'] = (df['HS'] * 0.05) + (df['HST'] * 0.25)
df['Away_xG_Proxy'] = (df['AS'] * 0.05) + (df['AST'] * 0.25)
df['H_xG_Creado'] = df.groupby('HomeTeam')['Home_xG_Proxy'].transform(lambda x: x.expanding().mean().shift())
df['A_xG_Creado'] = df.groupby('AwayTeam')['Away_xG_Proxy'].transform(lambda x: x.expanding().mean().shift())
df['H_xG_Concedido'] = df.groupby('HomeTeam')['Away_xG_Proxy'].transform(lambda x: x.expanding().mean().shift())
df['A_xG_Concedido'] = df.groupby('AwayTeam')['Home_xG_Proxy'].transform(lambda x: x.expanding().mean().shift())

features = ['H_xG_Creado', 'A_xG_Creado', 'H_xG_Concedido', 'A_xG_Concedido']
df_model = df.dropna(subset=features + ['FTR', 'FTHG', 'FTAG'])

X = df_model[features]
modelo_h2h = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
modelo_h2h.fit(X, np.where(df_model['FTR'] == 'H', 1, 0))

modelo_btts = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
modelo_btts.fit(X, np.where((df_model['FTHG'] > 0) & (df_model['FTAG'] > 0), 1, 0))

# ==========================================
# 3. ESCÁNER DE MERCADO EN VIVO
# ==========================================
API_KEY = os.environ.get('ODDS_API_KEY')
url_odds = f'https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={API_KEY}&regions=eu&markets=h2h,btts'
response = requests.get(url_odds)

mapeo_equipos = {
    "Manchester United": "Man United", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton", "Wolverhampton Wanderers": "Wolves"
}

todas_las_oportunidades = []

if response.status_code == 200:
    for partido in response.json():
        try:
            home_raw = partido['home_team']
            away_raw = partido['away_team']
            home_team = mapeo_equipos.get(home_raw, home_raw)
            away_team = mapeo_equipos.get(away_raw, away_raw)
            bookmaker = partido['bookmakers'][0]
            mercado_h2h = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
            mercado_btts = next((m for m in bookmaker['markets'] if m['key'] == 'btts'), None)
            
            h_xg_c = df[df['HomeTeam'] == home_team]['H_xG_Creado'].iloc[-1] if home_team in df['HomeTeam'].values else df['Home_xG_Proxy'].mean()
            a_xg_c = df[df['AwayTeam'] == away_team]['A_xG_Creado'].iloc[-1] if away_team in df['AwayTeam'].values else df['Away_xG_Proxy'].mean()
            h_xg_r = df[df['HomeTeam'] == home_team]['H_xG_Concedido'].iloc[-1] if home_team in df['HomeTeam'].values else df['Away_xG_Proxy'].mean()
            a_xg_r = df[df['AwayTeam'] == away_team]['A_xG_Concedido'].iloc[-1] if away_team in df['AwayTeam'].values else df['Home_xG_Proxy'].mean()
            
            X_nuevo = pd.DataFrame([{'H_xG_Creado': h_xg_c, 'A_xG_Creado': a_xg_c, 'H_xG_Concedido': h_xg_r, 'A_xG_Concedido': a_xg_r}])
            
            if mercado_h2h:
                odds_home = next(item['price'] for item in mercado_h2h['outcomes'] if item['name'] == home_raw)
                prob_h2h = modelo_h2h.predict_proba(X_nuevo)[0][1]
                dinero_h2h, porc_h2h, edge_h2h = calcular_gestion_riesgo(prob_h2h, odds_home)
                if dinero_h2h > 0:
                    todas_las_oportunidades.append({
                        'Fecha_Analisis': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Partido': f"{home_team} vs {away_team}", 'Local': home_team, 'Visitante': away_team,
                        'Pronostico': "Local", 'Cuota_Bookmaker': round(odds_home, 2),
                        'Prob_ML_%': round(prob_h2h * 100, 2), 'Ventaja_Edge_%': edge_h2h,
                        'Inversion_Sugerida_$': dinero_h2h, 'Porcentaje_Banca_%': porc_h2h,
                        'Resultado_Real': "PENDIENTE", 'Estado': "PENDIENTE", 'Rendimiento_Neto_$': 0.0
                    })
            
            if mercado_btts:
                odds_btts_si = next(item['price'] for item in mercado_btts['outcomes'] if item['name'] == 'Yes')
                prob_btts = modelo_btts.predict_proba(X_nuevo)[0][1]
                dinero_btts, porc_btts, edge_btts = calcular_gestion_riesgo(prob_btts, odds_btts_si)
                if dinero_btts > 0:
                    todas_las_oportunidades.append({
                        'Fecha_Analisis': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Partido': f"{home_team} vs {away_team}", 'Local': home_team, 'Visitante': away_team,
                        'Pronostico': "Ambos Anotan", 'Cuota_Bookmaker': round(odds_btts_si, 2),
                        'Prob_ML_%': round(prob_btts * 100, 2), 'Ventaja_Edge_%': edge_btts,
                        'Inversion_Sugerida_$': dinero_btts, 'Porcentaje_Banca_%': porc_btts,
                        'Resultado_Real': "PENDIENTE", 'Estado': "PENDIENTE", 'Rendimiento_Neto_$': 0.0
                    })
        except:
            pass

# ==========================================
# 4. EXTRACCIÓN DEL TOP 10 MÁS RENTABLE
# ==========================================
if todas_las_oportunidades:
    # Ordenar la lista de mayor a menor ventaja matemática (Edge)
    todas_las_oportunidades = sorted(todas_las_oportunidades, key=lambda x: x['Ventaja_Edge_%'], reverse=True)
    
    # Extraer estrictamente el Top 10
    top_10_diario = todas_las_oportunidades[:10]
    
    df_nuevas = pd.DataFrame(top_10_diario)
    if not os.path.isfile(archivo_csv):
        df_nuevas.to_csv(archivo_csv, index=False)
    else:
        df_existente = pd.read_csv(archivo_csv)
        # Evitar duplicados si el partido ya estaba guardado días anteriores
        df_final = pd.concat([df_existente, df_nuevas]).drop_duplicates(subset=['Partido', 'Pronostico'], keep='first')
        df_final.to_csv(archivo_csv, index=False)
    print(f"🎯 Escáner diario completado. Guardadas las {len(top_10_diario)} oportunidades con mayor Edge.")
else:
    print("📉 Mercado eficiente. Ningún partido superó el umbral de Kelly hoy.")
