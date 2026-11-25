"""
Ejecuta todo el proyecto de principio a fin (≈ 15-20 minutos en una laptop):
   1) verifica las relaciones termodinámicas del Cap. 12
   2) estudio de la Segunda Ley (pesos de penalización vs. ruido)
   3) entrena las PINN (casos A y B), evalúa el gemelo y la economía
   4) genera el reporte PDF
Uso:   python run_all.py
Luego: streamlit run app/app.py
"""
import subprocess, sys
from pathlib import Path
R = Path(__file__).parent
py = sys.executable
(R / "resultados").mkdir(exist_ok=True)
pasos = [
    ("Verificación termodinámica", [py, str(R / "simulador" / "termo.py")]),
    ("Estudio de la 2ª Ley", [py, str(R / "evaluacion" / "estudio_segunda_ley.py")]),
    ("Entrenamiento y evaluación del gemelo", [py, str(R / "evaluacion" / "evaluar_gemelo.py")]),
    ("Reporte PDF", [py, str(R / "informe" / "generar_informe.py")]),
]
for nombre, cmd in pasos:
    print(f"\n=== {nombre} ===", flush=True)
    subprocess.run(cmd, check=True, cwd=R)
print("\nListo. Reporte: informe/Proyecto12_Reporte.pdf   |   UI: streamlit run app/app.py")
