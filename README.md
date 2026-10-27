# MTest Live Lab · R ↔ Python

[**Abrir el laboratorio público**](https://mtest-live-lab.streamlit.app/)

Demostración académica para el webinar **«Multicolinealidad: ¿sigue importando en la era del Machine Learning?»**, 27 de octubre de 2026, 19:00 Ecuador. Celebra el hito de 10K+ descargas de MTest en CRAN informado por sus autores; no es un contador en tiempo real.

Ejecuta realmente **MTest 1.0.4 en R** y **mtest-py 0.1.4 en Python**. El motor R aparece primero. El laboratorio no modifica ni reimplementa los algoritmos oficiales.

## Instalación reproducible

Se recomienda Linux x86-64, Python 3.12.13 y R 4.5.3. Un único `environment.yml` fija las dependencias directas de Python y R desde conda-forge y PyPI. Las dependencias transitivas pueden actualizarse al resolver nuevamente el entorno. `locks/` conserva el inventario exacto conda y pip del entorno Linux probado, para reconstrucción independiente y auditoría.

```sh
git clone https://github.com/vmoprojs/mtest-live-lab.git
cd mtest-live-lab
conda env create -f environment.yml
conda activate mtest-live-lab
Rscript --vanilla scripts/install_r.R
python -m streamlit run app.py
```

Abrir http://localhost:8501. También puede usarse micromamba. No hace falta internet después de instalar el entorno: el código fuente oficial MTest está incluido sin cambios en `vendor/`, con procedencia, licencia y SHA-256. En Cloud, la inicialización verifica ese archivo e instala MTest una sola vez en `.r-library`; no descarga paquetes ni instala al pulsar botones. Se requiere R ≥ 4.1. **No usar `packages.txt` con el R 4.0.4 base de Debian 11.**

## Streamlit Community Cloud

Crear app desde este repositorio, rama `main`, entrada `app.py`, Python 3.12 en ajustes avanzados. Elegir `mtest-live-lab` si el subdominio está libre. `environment.yml` debe ser el único archivo de dependencias en la raíz. No se requieren secretos.

La app escucha en `0.0.0.0`. Conserva las protecciones CORS/XSRF. La primera visita verifica R, MTest y jsonlite, y muestra explícitamente los motores disponibles. Un fallo R nunca activa una sustitución por Python. Un endpoint HTTP de salud no demuestra por sí solo que R funcione: ejecutar **Ejecutar ambos** antes de compartir la URL.

## Uso

Cuatro secciones: MTest en R, datos y OLS descriptivo, MTest en Python, y comparación R/Python. El bloque ML compara OLS, Ridge y Lasso con train/test y escalado solo en entrenamiento. Hay datos sintéticos, CSV de ejemplo, carga validada, distribuciones bootstrap, umbrales, resultados CSV y kit reproducible.

Por defecto: Alta, ρ = 0.95, n = nsam = 300, B = 200, semilla de datos 27, semilla bootstrap 2026, c = 0.90 (VIF = 10). Otros escenarios: Baja 0.10 y Casi perfecta 0.999. Los botones ejecutan explícitamente los cálculos; los controles invalidan resultados anteriores. «Reiniciar demostración» recupera los valores iniciales.

## Interpretación y límites científicos

ASL VIF es la proporción bootstrap de R² auxiliares mayores que c. ASL Klein es la proporción de R² auxiliares mayores que el R² global de la misma réplica. **No son p-valores paramétricos tradicionales.** El bootstrap remuestrea filas completas y presupone observaciones independientes.

Ambos motores reciben exactamente los mismos doubles mediante CSV hexadecimal; se comprueba la matriz que R leyó. R² y VIF observados se comparan con rtol = 1e-8, atol = 1e-10. Semillas iguales no producen remuestreos iguales entre lenguajes, por lo que las ASL pueden diferir por Monte Carlo. B = 200 ofrece una demostración rápida, no inferencia de alta precisión. No se exige igualdad de ASL ni se afirma equivalencia formal entre implementaciones.

En remuestreos degenerados, R y Python pueden tratar NaN de manera diferente. La app oculta ASL afectadas y bloquea su comparación; conserva salidas originales identificadas en la descarga. La comprobación detecta valores no finitos y R² auxiliares ≥ 1, pero no audita exhaustivamente el rango de cada réplica.

MTest diagnostica el diseño; no certifica rendimiento predictivo, causalidad ni validez completa del modelo. Ridge y Lasso no eliminan la correlación entre predictores. La simulación conserva coeficientes opuestos: cambiar ρ también cambia la relación señal/ruido de Y. No se comparan errores predictivos bajo señal constante.

## Datos y recursos públicos

**Use exclusivamente datos públicos o sintéticos. No cargue datos personales, confidenciales ni sensibles.** El procesamiento ocurre en el servidor del proveedor. La app no guarda CSV permanentemente, no registra sus contenidos, no usa caché global para datos cargados y no comparte resultados entre sesiones. R usa un directorio temporal diferente por ejecución y lo elimina al finalizar, incluso ante errores y timeout normales. Cerrar una pestaña no garantiza borrado inmediato de memoria del servidor; reiniciar limpia los resultados de esa sesión. La infraestructura del proveedor conserva sus propios registros operativos.

Límites: 2 MB, 5000 filas, 30 columnas de entrada, 2–8 predictores; B ≤ 500, nsam ≤ 2000, B × nsam × p ≤ 2 millones; procesos bootstrap con límite de 90 s. Se admite un análisis por proceso servidor a la vez; otros visitantes reciben un aviso para reintentar. El experimento ML usa 80 remuestreos de entrenamiento. Esto no constituye una prueba de carga ni un servicio con garantía de disponibilidad.

## Docker y alternativa Render

```sh
docker build --platform linux/amd64 -t mtest-live-lab:live02 .
docker run --rm --name mtest-live-lab -p 8501:8501 mtest-live-lab:live02
# Pruebas dentro del entorno Linux
docker run --rm mtest-live-lab:live02 python -m pytest -q
```

La imagen instala MTest durante el build, ejecuta como usuario sin privilegios y usa `/_stcore/health`. `PORT` se toma del proveedor y se valida. Para Render, crear **Web Service → repositorio → Docker → Free**, Dockerfile raíz, health check `/_stcore/health`; no requiere disco persistente ni secretos. No se ha autorizado ningún plan de pago. Verificar memoria disponible y consumo antes del webinar.

Render Free se duerme tras 15 minutos de inactividad y comparte 750 horas mensuales por espacio. Hay cuotas de tráfico y construcción; sin método de pago, al agotarlas puede suspender servicios/construcciones. Con método de pago puede facturar excedentes: revisar límites antes de activar. No se ha creado un servicio Render por este repositorio automáticamente.

## Pruebas

```sh
python -m pytest -q
python scripts/validate_deploy.py
```

Incluyen escenarios revisados, identidad de entrada, llamadas directas oficiales, reproducibilidad, casos inválidos, NaN, ausencia de leakage, aislamiento de sesiones, límites públicos y disponibilidad/versiones. `validate_deploy.py` ensaya los parámetros de presentación y emite evidencia JSON con datos sintéticos.

## Referencias y atribución

MTest: Víctor Morales-Oñate y Bolívar Morales-Oñate. [CRAN](https://cran.r-project.org/package=MTest) · [Código R oficial](https://github.com/vmoprojs/MTest) · [mtest-py en PyPI](https://pypi.org/project/mtest-py/) · [Artículo original](https://doi.org/10.33333/rp.vol51n2.05).

Código del laboratorio: GPL-3.0-or-later, 2026. La distribución oficial MTest conserva su licencia GPL (≥ 3) y sus autores; otras dependencias conservan sus licencias. El código fuente completo de MTest distribuido está en el tarball original, sin modificaciones.

Documentación de infraestructura consultada: [dependencias Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [límites Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/status), [Render Free](https://render.com/docs/free), [Docker en Render](https://render.com/docs/docker).
