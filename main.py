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

print("🤖 Iniciando Tubería Cuantitativa Autolimpiable y Multimercado...")

# ==========================================
# 1. INGESTA DE HISTÓRICOS Y FILTRADO DE DATOS
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

# ==========================================
# 2. BUCLE DE RETROALIMENTACIÓN (SELF-LEARNING AUDIT)
# ==========================================
archivo_csv = 'predicciones_auditadas.csv'

if os.path.isfile(archivo_csv):
    print("🔄 Analizando predicciones pasadas para auditar resultados...")
    df_audit = pd.read_csv(archivo_csv)
    
    if 'Estado' in df_audit.columns and not df_audit[df_audit['Estado'] == 'PENDIENTE'].empty:
        for idx, fila in df_audit[df_audit['Estado'] == 'PENDIENTE'].iterrows():
            # Buscar el partido en el histórico descargado
            partido_real = df[(df['HomeTeam'] == fila['Local']) & (df['AwayTeam'] == fila['Visitante'])]
            
            if not partido_real.empty:
                resultado_f = partido_real.iloc[-1]
                ftr = resultado_f['FTR']
                fthg = int(resultado_f['FTHG'])
                ftag = int(resultado_f['FTAG'])
                
                marcador = f"{fthg}-{ftag}"
                hit = False
                
                # Validar según el tipo de mercado apostado
                if fila['Pronostico'] == "Local" and ftr == 'H':
                    hit = True
                elif fila['Pronostico'] == "Ambos Anotan" and fthg > 0 and ftag > 0:
                    hit = True
                
                status_final = "GANADO" if hit else "PERDIDO"
                cuota = float(fila['Cuota_Bookmaker'])
                inversion = float(fila['Inversion_Sugerida_$'])
                retorno = round((cuota - 1) * inversion, 2) if hit else -inversion
                
                # Escribir el aprendizaje en la base de datos interna
                df_audit.at[idx, 'Resultado_Real'] = marcador
                df_audit.at[idx, 'Estado'] = status_final
                df_audit.at[idx, 'Rendimiento_Neto_$'] = retorno
                print(f"   📈 Partido Auditado: {fila['Partido']} | Pronóstico: {fila['Pronostico']} | Resultado: {marcador} -> {status_final}")
        
        df_audit.to_csv(archivo_csv, index=False)
    print("✅ Auditoría interna finalizada y guardada.")

# ==========================================
# 3. FEATURE ENGINEERING: MODELADO DE xG PROXY
# ==========================================
# Al no contar con xG de pago, calculamos un Proxy matemático de Goles Esperados 
# basado en volumen de tiros totales (Shots) y precisión matemática (Shots on Target).
df['Home_xG_Proxy'] = (df['HS'] * 0.05) + (df['HST'] * 0.25)
df['Away_xG_Proxy'] = (df['AS'] * 0.05) + (df['AST'] * 0.25)

# Variables de rendimiento acumulado desplazadas (shift) para evitar Data Leakage
df['H_xG_Creado'] = df.groupby('HomeTeam')['Home_xG_Proxy'].transform(lambda x: x.expanding().mean().shift())
df['A_xG_Creado'] = df.groupby('AwayTeam')['Away_xG_Proxy'].transform(lambda x: x.expanding().mean().shift())
df['H_xG_Concedido'] = df.groupby('HomeTeam')['Away_xG_Proxy'].transform(lambda x: x.expanding().mean().shift())
df['A_xG_Concedido'] = df.groupby('AwayTeam')['Home_xG_Proxy'].transform(lambda x: x.expanding().mean().shift())

features = ['H_xG_Creado', 'A_xG_Creado', 'H_xG_Concedido', 'A_xG_Concedido']
df_model = df.dropna(subset=features + ['FTR', 'FTHG', 'FTAG'])

# ==========================================
# 4. ENTRENAMIENTO DE INTELIGENCIAS ARTIFICIALES DUALES
# ==========================================
X = df_model[features]

# Target 1: Victoria Local (H2H)
y_h2h = np.where(df_model['FTR'] == 'H', 1, 0)
modelo_h2h = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
modelo_h2h.fit(X, y_h2h)

# Target 2: Ambos Anotan (BTTS)
y_btts = np.where((df_model['FTHG'] > 0) & (df_model['FTAG'] > 0), 1, 0)
modelo_btts = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
modelo_btts.fit(X, y_btts)

print("✅ Modelos de Machine Learning (H2H + BTTS + xG) calibrados.")

# ==========================================
# 5. ESCÁNER MULTIMERCADO EN VIVO (THE ODDS API)
# ==========================================
API_KEY = os.environ.get('ODDS_API_KEY')
# Solicitamos de manera simultánea cuotas de Ganador (h2h) y Ambos Anotan (btts)
url_odds = f'https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={API_KEY}&regions=eu&markets=h2h,btts'
response = requests.get(url_odds)

mapeo_equipos = {
    "Manchester United": "Man United", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton", "Wolverhampton Wanderers": "Wolves"
}

nuevas_predicciones = []

if response.status_code == 200:
    partidos = response.json()
    for partido in partidos:
        try:
            home_raw = partido['home_team']
            away_raw = partido['away_team']
            home_team = mapeo_equipos.get(home_raw, home_raw)
            away_team = mapeo_equipos.get(away_raw, away_raw)
            
            bookmaker = partido['bookmakers'][0]
            
            # Extraer cuotas de manera segura de las estructuras anidadas
            mercado_h2h = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
            mercado_btts = next((m for m in bookmaker['markets'] if m['key'] == 'btts'), None)
            
            # Obtener métricas xG actuales para la predicción
            h_xg_c = df[df['HomeTeam'] == home_team]['H_xG_Creado'].iloc[-1] if home_team in df['HomeTeam'].values else df['Home_xG_Proxy'].mean()
            a_xg_c = df[df['AwayTeam'] == away_team]['A_xG_Creado'].iloc[-1] if away_team in df['AwayTeam'].values else df['Away_xG_Proxy'].mean()
            h_xg_r = df[df['HomeTeam'] == home_team]['H_xG_Concedido'].iloc[-1] if home_team in df['HomeTeam'].values else df['Away_xG_Proxy'].mean()
            a_xg_r = df[df['AwayTeam'] == away_team]['A_xG_Concedido'].iloc[-1] if away_team in df['AwayTeam'].values else df['Home_xG_Proxy'].mean()
            
            X_nuevo = pd.DataFrame([{
                'H_xG_Creado': h_xg_c, 'A_xG_Creado': a_xg_c,
                'H_xG_Concedido': h_xg_r, 'A_xG_Concedido': a_xg_r
            }])
            
            # --- EVALUACIÓN MERCADO LOCAL ---
            if mercado_h2h:
                odds_home = next(item['price'] for item in mercado_h2h['outcomes'] if item['name'] == home_raw)
                prob_h2h = modelo_h2h.predict_proba(X_nuevo)[0][1]
                dinero_h2h, porc_h2h, edge_h2h = calcular_gestion_riesgo(prob_h2h, odds_home)
                
                if dinero_h2h > 0:
                    nuevas_predicciones.append({
                        'Fecha_Analisis': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Partido': f"{home_team} vs {away_team}", 'Local': home_team, 'Visitante': away_team,
                        'Pronostico': "Local", 'Cuota_Bookmaker': round(odds_home, 2),
                        'Prob_ML_%': round(prob_h2h * 100, 2), 'Ventaja_Edge_%': edge_h2h,
                        'Inversion_Sugerida_$': dinero_h2h, 'Porcentaje_Banca_%': porc_h2h,
                        'Resultado_Real': "PENDIENTE", 'Estado': "PENDIENTE", 'Rendimiento_Neto_$': 0.0
                    })
            
            # --- EVALUACIÓN MERCADO AMBOS ANOTAN ---
            if mercado_btts:
                odds_btts_si = next(item['price'] for item in mercado_btts['outcomes'] if item['name'] == 'Yes')
                prob_btts = modelo_btts.predict_proba(X_nuevo)[0][1]
                dinero_btts, porc_btts, edge_btts = calcular_gestion_riesgo(prob_btts, odds_btts_si)
                
                if dinero_btts > 0:
                    nuevas_predicciones.append({
                        'Fecha_Analisis': datetime.now().strftime("%Y-%m-%d %H:%M"),
                        'Partido': f"{home_team} vs {away_team}", 'Local': home_team, 'Visitante': away_team,
                        'Pronostico': "Ambos Anotan", 'Cuota_Bookmaker': round(odds_btts_si, 2),
                        'Prob_ML_%': round(prob_btts * 100, 2), 'Ventaja_Edge_%': edge_btts,
                        'Inversion_Sugerida_$': dinero_btts, 'Porcentaje_Banca_%': porc_btts,
                        'Resultado_Real': "PENDIENTE", 'Estado': "PENDIENTE", 'Rendimiento_Neto_$': 0.0
                    })
        except Exception as e:
            pass

# ==========================================
# 6. UNIFICACIÓN Y ALMACENAMIENTO DE CONTROL
# ==========================================
if nuevas_predicciones:
    df_nuevas = pd.DataFrame(nuevas_predicciones)
    if not os.path.isfile(archivo_csv):
        df_nuevas.to_csv(archivo_csv, index=False)
    else:
        df_existente = pd.read_csv(archivo_csv)
        # Unir y eliminar duplicados basándose en la combinación única de Partido y Pronóstico
        df_final = pd.concat([df_existente, df_nuevas]).drop_duplicates(subset=['Partido', 'Pronostico'], keep='first')
        df_final.to_csv(archivo_csv, index=False)
    print(f"📊 Carteras actualizadas. Se encontraron {len(nuevas_predicciones)} Value Bets blindadas por xG y Kelly.")
else:
    print("✨ Análisis cerrado. Los mercados de Goles y H2H no presentan ineficiencias explotables.")
