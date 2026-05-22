import os
import pandas as pd
def generar_reporte_financiero():
    archivo_csv = 'predicciones_auditadas.csv'
    reporte_md = 'REPORTE_RENDIMIENTO.md'
    
    print("📊 Iniciando auditoría mensual del portafolio cuantitativo...")
    
    if not os.path.isfile(archivo_csv):
        print("⚠️ No se encontró el archivo de predicciones históricas. Saltando reporte.")
        return
    df = pd.read_csv(archivo_csv)
    
    # Filtrar solo registros que ya fueron auditados (excluir PENDIENTES)
    df_terminados = df[df['Estado'].isin(['GANADO', 'PERDIDO'])]
    
    if df_terminados.empty:
        with open(reporte_md, 'w', encoding='utf-8') as f:
            f.write("# 📊 Reporte de Rendimiento Cuantitativo\n\n")
            f.write("⚠️ Todavía no hay suficientes partidos terminados y auditados para generar métricas estadísticas.\n")
        return
    # 1. CÁLCULO DE MÉTRICAS CLAVE
    total_apuestas = len(df_terminados)
    apuestas_ganadas = len(df_terminados[df_terminados['Estado'] == 'GANADO'])
    
    win_rate = (apuestas_ganadas / total_apuestas) * 100 if total_apuestas > 0 else 0.0
    pnl_total = df_terminados['Rendimiento_Neto_$'].sum()
    total_invertido = df_terminados['Inversion_Sugerida_$'].sum()
    
    # El Yield representa el porcentaje de beneficio real por cada dólar que pusiste en riesgo
    yield_porcentaje = (pnl_total / total_invertido) * 100 if total_invertbed > 0 else 0.0
    # Desglose estadístico por mercado analizado
    btts_df = df_terminados[df_terminados['Pronostico'] == 'Ambos Anotan']
    h2h_df = df_terminados[df_terminados['Pronostico'] == 'Local']
    
    def metricas_por_mercado(sub_df):
        if sub_df.empty: return "0 / 0 (0.0%)", 0.0
        ganados = len(sub_df[sub_df['Estado'] == 'GANADO'])
        totales = len(sub_df)
        wr = (ganados / totales) * 100
        pnl = sub_df['Rendimiento_Neto_$'].sum()
        return f"{ganados} / {totales} ({wr:.1f}%)", round(pnl, 2)
    btts_stats, btts_pnl = metricas_por_mercado(btts_df)
    h2h_stats, h2h_pnl = metricas_por_mercado(h2h_df)
    # 2. CONSTRUCCIÓN ESTRUCTURAL DEL INFORME EN MARKDOWN
    contenido = f"""# 📊 Reporte Ejecutivo de Rendimiento Cuantitativo
*Actualizado automáticamente de forma mensual por el sistema de auditoría interna.*
---
## 📈 Resumen Financiero General

| Métrica | Valor General |
| :--- | :--- |
| **Total de Operaciones Realizadas** | {total_apuestas} partidos |
| **Tasa de Acierto Global (Win Rate)** | **{win_rate:.2f}%** |
| **Beneficio / Pérdida Acumulado (P&L)** | **{pnl_total:+.2f} USD** |
| **Volumen Total Capitalizado** | {total_invertido:.2f} USD |
| **Retorno Neto sobre Inversión (Yield)** | **{yield_porcentaje:+.2f}%** |

---
## 🎯 Desglose de Precisión por Mercado de Apuestas

| Mercado | Tasa de Acierto (Ganadas/Totales) | Retorno Financiero (P&L) |
| :--- | :--- | :--- |
| **H2H (Victorias Locales)** | {h2h_stats} | {h2h_pnl:+.2f} USD |
| **BTTS (Ambos Anotan)** | {btts_stats} | {btts_pnl:+.2f} USD |

---
## 📑 Historial de las Últimas 10 Operaciones Ejecutadas

| Fecha | Partido | Mercado | Cuota | Prob. IA | Inversión | Resultado | P&L |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | <br> """ <br> # Tomar las últimas 10 apuestas y agregarlas a la tabla del reporte <br> ultimas_apuestas = df_terminados.tail(10).iloc[::-1] <br> for _, fila in ultimas_apuestas.iterrows(): <br> contenido += f"| {fila['Fecha_Analisis']} | {fila['Partido']} | {fila['Pronostico']} | {fila['Cuota_Bookmaker']} | {fila['Prob_ML_%']}% | {fila['Inversion_Sugerida_$']} USD | {fila['Resultado_Real']} ({fila['Estado']}) | {fila['Rendimiento_Neto_$']:+.2f} USD |\n"

    # Escribir el archivo final
    with open(reporte_md, 'w', encoding='utf-8') as f:
        f.write(contenido)
        
    print("✅ Archivo 'REPORTE_RENDIMIENTO.md' compilado perfectamente.")
if __name__ == "__main__":
    generar_reporte_financiero()
