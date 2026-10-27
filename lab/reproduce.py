import io
import json
import zipfile

from lab.data import transport_csv


def snippets(response, predictors, config):
    names_json = json.dumps(predictors, ensure_ascii=True)
    response_json = json.dumps(response, ensure_ascii=True)
    r = f'''# design.csv usa texto hexadecimal para conservar exactamente los doubles.
d <- read.csv("design.csv", check.names=FALSE, colClasses="character")
d[] <- lapply(d, as.numeric)
predictores <- jsonlite::fromJSON({json.dumps(names_json)})
respuesta <- jsonlite::fromJSON({json.dumps(response_json)})
datos <- d[, c(respuesta, predictores), drop=FALSE]
alias <- sprintf("x%03d", seq_along(predictores))
names(datos) <- c("response", alias)
modelo <- lm(reformulate(alias, response="response"), data=datos, na.action=na.fail)
resultado <- MTest::MTest(
  object=modelo, nboot={config.n_boot}, nsam={config.nsam},
  seed={config.seed}, valor_vif={config.r2_threshold}, trace=FALSE
)
print(resultado$R.tot)
print(resultado$vif.tot)
print(resultado$pval_vif)
print(resultado$pval_klein)
'''
    py = f'''import pandas as pd
from mtest import mtest

d = pd.read_csv("design.csv", dtype=str).map(float.fromhex)
resultado = mtest(
    d[{predictors!r}], d[{response!r}].to_numpy(),
    n_boot={config.n_boot}, nsam={config.nsam},
    seed={config.seed}, r2_threshold={config.r2_threshold},
    add_intercept=True, return_distributions=True,
)
for key in ("R2_global", "R2_aux", "VIF", "p_vif", "p_klein"):
    print(key, resultado[key])
'''
    return r, py


def reproduction_zip(frame, response, predictors, config):
    r, py = snippets(response, predictors, config)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("design.csv", transport_csv(frame))
        z.writestr("run_R.R", r)
        z.writestr("run_python.py", py)
        z.writestr("LEEME.txt", "Ejecute en esta carpeta: Rscript run_R.R y python run_python.py.\n"
                   "Dependencias: MTest 1.0.4 + jsonlite en R; mtest-py 0.1.4 + pandas en Python.\n"
                   "El CSV usa representación hexadecimal numérica para conservar cada double.\n"
                   "Semillas iguales NO producen los mismos remuestreos entre lenguajes.\n")
    return buffer.getvalue()
