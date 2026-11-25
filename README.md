# Proyecto 12 — Gemelo Digital Termodinámico con PINNs (INQ-3108)

Central geotérmica de doble evaporación flash de 25 MWe. Dos casos de fluido:
**A** agua + NCG (CO₂) · **B** salmuera con NaCl + NCG (CO₂/H₂S). Todo con **datos simulados**; hay una pestaña para cargar datos reales.

## Qué contiene
| Carpeta / archivo | Qué es |
|---|---|
| `informe/Proyecto12_Reporte.pdf` | Reporte completo (ecuaciones, resultados, respuesta a la pregunta de decisión) |
| `app/app.py` | Interfaz (Streamlit): gemelo en operación, simulación manual, Cap. 12, estudio 2ª Ley, datos reales |
| `simulador/` | Termodinámica (IAPWS-95, Maxwell, Clapeyron, Joule-Thomson, Newton-Raphson), planta, ensuciamiento por sílice, separador ciclónico |
| `pinn/` | Red PINN (PyTorch) y modelos ya entrenados (`pinn/modelos/`) |
| `evaluacion/` | Estudio de la 2ª Ley y evaluación del gemelo |
| `resultados/` | Tablas y series generadas (CSV/JSON) |

## Cómo usarlo (Windows / Mac / Linux)
1. Instalar Python 3.10 o superior.
2. En una terminal, dentro de esta carpeta:
   ```
   python -m venv venv
   venv\Scripts\activate        (Mac/Linux:  source venv/bin/activate)
   pip install -r requirements.txt
   ```
3. **Abrir la interfaz** (los modelos y resultados ya vienen incluidos):
   ```
   streamlit run app/app.py
   ```
4. (Opcional) **Rehacer todo desde cero** — entrena las redes y regenera resultados y PDF (≈ 15-20 min):
   ```
   python run_all.py
   ```

## Datos reales
En la interfaz, pestaña "Datos reales": descargar la plantilla CSV, llenarla con los registros de planta y subirla.
Con datos reales hay que reentrenar sin etiquetas (solo balances físicos) y recalibrar los parámetros supuestos.

## Advertencias
- Los coeficientes de ensuciamiento y los supuestos de ingeniería están en `simulador/planta.py` (`PAR`) y se explican en el reporte; no provienen de una planta real.
- Referencias bibliográficas y correlaciones de sílice escritas de memoria: verificar en las fuentes antes de entregar.
