from pythermalcomfort.models import pmv_ppd_iso

class PlanificadorGreedy:
    def __init__(self, acciones, modelo_de_transicion):
        self.acciones = acciones
        self.modelo_de_transicion = modelo_de_transicion

    def plan(self, agente):
        estado_actual = agente.get_current_state()
        mejor_accion = None
        mejor_score = float('inf')

        for accion in self.acciones:
            try:
                estado_simulado = self.modelo_de_transicion.sample(estado_actual, accion)
                pmv = pmv_ppd_iso(
                    tdb=estado_simulado.tdb,
                    tr=estado_simulado.tr,
                    vr=estado_simulado.vr,
                    rh=estado_simulado.rh,
                    met=estado_simulado.met,
                    clo=estado_simulado.clo
                )["pmv"]
            except Exception:
                continue

            score = abs(pmv)
            if score < mejor_score:
                mejor_score = score
                mejor_accion = accion

        return mejor_accion
