from streamlit.testing.v1 import AppTest
from lab.engines import ROOT


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_dashboard_both_engines_invalidation_reset():
    app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=40).run()
    assert not app.exception
    assert len(app.tabs) == 4
    assert "result_R" not in app.session_state
    button(app, "Ejecutar ambos").click().run()
    assert not app.exception
    assert app.session_state["result_R"].clean
    assert app.session_state["result_Python"].clean
    assert any("Comparables" in s.value for s in app.success)
    app.slider(key="rho").set_value(.96).run()
    assert not app.exception
    assert "result_R" not in app.session_state
    assert "result_Python" not in app.session_state
    button(app, "Ejecutar MTest en R").click().run()
    assert not app.exception
    assert app.session_state["result_R"].vif[0] > 5
    button(app, "Comparar OLS, Ridge y Lasso").click().run()
    assert not app.exception
    assert "ml_result" in app.session_state
    button(app, "Reiniciar demostración").click().run()
    assert not app.exception
    assert app.slider(key="rho").value == .95
    assert app.selectbox(key="scenario").value == "Alta"
    assert not any("widget" in item.value.lower() for item in app.warning)
    assert "result_R" not in app.session_state


def test_python_only_and_statistical_labels():
    app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=40).run()
    button(app, "Ejecutar MTest en Python").click().run()
    assert not app.exception
    assert app.session_state["result_Python"].version == "0.1.4"
    text = " ".join(x.value for x in app.caption) + " ".join(x.value for x in app.markdown)
    assert "no es un p-valor paramétrico" in text
    assert "no eliminan la correlación" in text
    assert "no certifica rendimiento predictivo" in text


def test_source_switch_retains_correct_nsam_and_no_widget_warnings():
    app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=40).run()
    app.radio(key="source").set_value("Cargar CSV").run()
    assert not app.exception
    app.radio(key="source").set_value("Simulador").run()
    assert app.number_input(key="nsam").value == 300
    app.selectbox(key="scenario").set_value("Alta").run()
    assert app.slider(key="rho").value == .95
    assert not any("widget" in item.value.lower() for item in app.warning)
