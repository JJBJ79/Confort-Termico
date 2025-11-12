# Confort Termico c/POMDP

import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys
import copy
import math
from itertools import product
import json
import logging
from tqdm import tqdm
from math import floor
import time
from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import traceback
from pythermalcomfort.models import pmv_ppd_iso
from pomdp_py.algorithms.po_uct import POUCT
from pomdp_py.algorithms.po_uct import RandomRollout
from scipy.stats import norm
from copy import deepcopy
from collections import Counter
from typing import Any, Iterator, List
from scipy.stats import truncnorm, gamma
import os
from pomdp_py import POMCP
from pomdp_py.algorithms.po_uct import QNode
from planificador_greedy import PlanificadorGreedy

def setup_logging():
    log_dir = "registro_consola"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    logging.basicConfig(
        filename=os.path.join(log_dir, "registro_consola.log"),
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
        force=True
    )
    
    class RedirigirSalida:
        def write(self, message):
            if message.strip():
                logging.info(message.strip())
        def flush(self):
            pass
    
    sys.stdout = RedirigirSalida()
    sys.stderr = RedirigirSalida()


logging.basicConfig(level=logging.DEBUG)


#funciones unitarias

def sample_tdb(mu, sigma, lower=10, upper=35):

    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

def sample_hr_trunc(mu, sigma, lower=20, upper=80):

    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

def sample_Tt(mu, sigma, lower=16, upper=34):

    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

def sample_vr(shape, scale):

    return gamma.rvs(shape, scale=scale)

def sample_met():

    choices = [0.8, 1, 1.2, 1.6, 2]  
    return random.choice(choices)

def sample_clo(min_val=0.3, max_val=1.5):
    return random.uniform(min_val, max_val)

def calculate_n_eff(particles: List):

    weights = np.array([p.weight for p in particles])
    sum_sq = np.sum(np.square(weights))
    return 1.0 / sum_sq if sum_sq > 0 else 0


def systematic_resample(particles):
    N = len(particles)
    positions = (np.arange(N) + random.random()) / N
    cumulative_sum = np.cumsum([p.weight for p in particles])
    cumulative_sum[-1] = 1.0
    indexes = np.searchsorted(cumulative_sum, positions)
    new_particles = [particles[int(i)].copy() for i in indexes]
    for p in new_particles:
        p.weight = 1.0 / N
    return new_particles

def RS_resample(N, weights, particles):
    cumulative_sum = np.cumsum(weights)
    positions = np.random.uniform(0, 1.0 / N) + np.arange(N) / N
    indexes = np.searchsorted(cumulative_sum, positions)
    indexes = [min(idx, len(particles) - 1) for idx in indexes]
    new_particles = [particles[int(i)].copy() for i in indexes]
    for p in new_particles:
        p.weight = 1.0 / N
    return new_particles

def roughen_particles(particles, sigma_factor=0.1, jitter_min=0.01):
    jitter = max(sigma_factor, jitter_min)
    for i, p in enumerate(particles):
        if not isinstance(p.estado, EstadoConfortTermico):
            logging.warning(f"[WARNING] Partícula {i} con estado inválido: {type(p.estado)}")
            if isinstance(p.estado, dict):
                p.estado = EstadoConfortTermico(
                    tdb=p.estado.get("tdb", 24),
                    tr=p.estado.get("tr", 24),
                    rh=p.estado.get("rh", 50),
                    Tt=p.estado.get("Tt", 25),
                    vr=p.estado.get("vr", 0.3),
                    met=p.estado.get("met", 1.0),
                    clo=p.estado.get("clo", 0.75),
                    pmv=p.estado.get("pmv", None)
                )
            else:
                p.estado = EstadoConfortTermico.generar_estado_aleatorio()

        p.estado.tdb += np.random.normal(0, jitter)
        p.estado.tr  += np.random.normal(0, jitter)
        p.estado.rh  += np.random.normal(0, jitter)
        p.estado.Tt  += np.random.normal(0, jitter)
        p.estado.vr  += np.random.normal(0, jitter)
        p.estado.met += np.random.normal(0, jitter)
        p.estado.clo += np.random.normal(0, jitter)

        p.estado.tdb = np.clip(p.estado.tdb, 22.0, 26.0)
        p.estado.tr  = np.clip(p.estado.tr, 22.0, 26.0)
        p.estado.rh  = np.clip(p.estado.rh, 40.0, 60.0)
        p.estado.Tt  = np.clip(p.estado.Tt, 16.0, 33.0)
        p.estado.vr  = np.clip(p.estado.vr, 0.1, 0.5)
        p.estado.met = np.clip(p.estado.met, 1.0, 1.2)
        p.estado.clo = np.clip(p.estado.clo, 0.5, 1.0)

        p.estado.pmv = p.estado.calcular_pmv()
    return particles

def perturb_weights(particles, min_factor=0.95, max_factor=1.05):
    for p in particles:
        p.weight *= np.random.uniform(min_factor, max_factor)
    total_weight = sum(p.weight for p in particles) + 1e-6 
    for p in particles:
        p.weight /= total_weight
    return particles

def update_particles(particles, n_eff_threshold):
    n_eff = calculate_n_eff(particles)
    logging.debug(f"[DEBUG] n_eff calculado: {n_eff:.2f}")

    if len(particles) < 3:
        logging.warning(f"[FALLBACK] Muy pocas partículas ({len(particles)}). Inyectando respaldo.")
        n_extra = max(5, int(0.2 * len(particles)))
        extra = [Particula(EstadoConfortTermico.generar_estado_aleatorio()) for _ in range(n_extra)]
        total = len(particles) + len(extra)
        all_particles = particles + extra
        for p in all_particles:
            p.weight = 1.0 / total
        return all_particles

    if n_eff < n_eff_threshold:
        logging.info(f"[DEBUG] n_eff {n_eff:.2f} menor al umbral {n_eff_threshold}. Re-muestreando...")
        particles = systematic_resample(particles)

    return particles


class EstadoConfortTermico(pomdp_py.State):

    def __init__(self, tdb, tr, rh, Tt, vr, met, clo, pmv=None):
        self.tdb = tdb
        self.tr = tr
        self.rh = rh
        self.Tt = Tt
        self.vr = vr
        self.met = met
        self.clo = clo
        if pmv is None:
            self.pmv = self.calcular_pmv()
        else:
            self.pmv = pmv

    @classmethod
    def generar_estado_aleatorio(cls):

        max_tries = 1000
        for _ in range(max_tries):
            Tt = random.uniform(16, 33)
            tdb = np.clip(Tt + random.gauss(0, 0.5), 22, 26)
            tr = random.uniform(22, 26)
            rh = random.uniform(40, 60)
            vr = random.uniform(0.1, 0.5)
            met = random.choice([1, 1.2])
            clo = random.uniform(0.5, 1)
            try:
                estado = cls(tdb, tr, rh, Tt, vr, met, clo)
                if (isinstance(estado.pmv, (int, float)) and 
                    np.isfinite(estado.pmv) and 
                    (-3.0 <= estado.pmv <= 3.0)):
                    logging.debug(f"Estado generado con PMV {estado.pmv}")
                    return estado
            except Exception as e:
                logging.error(f"Error generando estado: {e}")
                continue
        raise ValueError("[ERROR] No se pudo generar un estado válido de confort térmico tras múltiples intentos.")
        
    def calcular_pmv(self, max_attempts=1):
        try:
            parametros = {"tdb": self.tdb, "tr": self.tr, "vr": self.vr,"rh": self.rh, "met": self.met, "clo": self.clo}
            logging.debug(f"[DEBUG] Estado antes de calcular PMV: {parametros}")
            
            for k, v in parametros.items():
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    logging.error(f"[ERROR] Parámetro inválido para PMV: {k} = {v}")
                    return None

            resultado = pmv_ppd_iso(
                tdb=self.tdb,
                tr=self.tr,
                vr=self.vr,
                rh=self.rh,
                met=self.met,
                clo=self.clo,
                model="7730-2005"
            )

            pmv_valor = resultado.pmv if hasattr(resultado, "pmv") else resultado.get("pmv", None)
            if pmv_valor is None:
                logging.error("[ERROR] El resultado del modelo no contiene 'pmv'.")
                return None
            if not np.isfinite(pmv_valor) or np.isnan(pmv_valor):
                logging.error("[ERROR] PMV resultó ser NaN o no finito tras el cálculo.")
                return None
            if not (-3.0 <= pmv_valor <= 3.0):
                logging.warning(f"[WARNING] PMV fuera de rango esperado: {pmv_valor}")
                return None

            self.pmv = pmv_valor
            return pmv_valor

        except Exception as e:
            logging.exception(f"[ERROR] Excepción durante cálculo de PMV: {e}")
            return None

    def __getitem__(self, key):
        if key == "tdb":
            return self.tdb
        elif key == "tr":
            return self.tr
        elif key == "rh":
            return self.rh
        elif key == "Tt":
            return self.Tt
        elif key == "vr":
            return self.vr
        elif key == "met":
            return self.met
        elif key == "clo":
            return self.clo
        elif key == "pmv":
            return self.pmv
        else:
            raise KeyError(f"Clave '{key}' no encontrada en EstadoConfortTermico.")

    def to_dict(self):
        return {
            "tdb": self.tdb,
            "tr": self.tr,
            "rh": self.rh,
            "Tt": self.Tt,
            "vr": self.vr,
            "met": self.met,
            "clo": self.clo,
            "pmv": self.pmv
        }

    def __eq__(self, other):
        return (isinstance(other, EstadoConfortTermico) and
                math.isclose(self.tdb, other.tdb, abs_tol=1e-5) and
                math.isclose(self.tr, other.tr, abs_tol=1e-5) and
                math.isclose(self.rh, other.rh, abs_tol=1e-5) and
                math.isclose(self.Tt, other.Tt, abs_tol=1e-5) and
                math.isclose(self.vr, other.vr, abs_tol=1e-5) and
                math.isclose(self.met, other.met, abs_tol=1e-5) and
                math.isclose(self.clo, other.clo, abs_tol=1e-5) and
                math.isclose(self.pmv, other.pmv, abs_tol=1e-5))

    def __hash__(self):
        return hash((
            round(self.tdb, 2),
            round(self.tr, 2),
            round(self.rh, 2),
            round(self.Tt, 2),
            round(self.vr, 2),
            round(self.met, 2),
            round(self.clo, 2)
        ))

    def rhash(self):
        return self.__hash__()


class AccionConfortTermico(pomdp_py.Action):
    def __init__(self, Tt: float = None, vr: float = None):
        self.Tt = round(Tt if Tt is not None else random.uniform(16.0, 34.0), 2)
        self.vr = round(vr if vr is not None else random.uniform(0.0, 2.0), 2)

        if not (16.0 <= self.Tt <= 34.0):
            raise ValueError(f"[ERROR] Tt fuera de rango: {self.Tt}")
        if not (0.0 <= self.vr <= 2.0):
            raise ValueError(f"[ERROR] vr fuera de rango: {self.vr}")

    def __repr__(self):
        return f"AccionConfortTermico(Tt={self.Tt}, vr={self.vr})"

    def __eq__(self, other):
        return (
            isinstance(other, AccionConfortTermico) and
            self.Tt == other.Tt and
            self.vr == other.vr
        )

    def __hash__(self):
        return hash((self.Tt, self.vr))

    def ejecutar_accion(self, estado):

        estado.Tt = self.Tt
        estado.vr = self.vr
        estado.pmv = estado.calcular_pmv()
        logging.debug(f"[DEBUG] Acción aplicada -> Tt={self.Tt}, vr={self.vr}, pmv={estado.pmv}")

    def es_accion_valida(self) -> bool:
        return (16.0 <= self.Tt <= 34.0) and (0.0 <= self.vr <= 2.0)
    
class Particula(pomdp_py.State):
    def __init__(self, estado, weight=1.0):
        if isinstance(estado, Particula):
            logging.debug("[DEBUG] Inicializando partícula a partir de otra partícula.")
            self.estado = copy.deepcopy(estado.estado)
            self.weight = estado.weight
        else:
            if not isinstance(estado, EstadoConfortTermico):
                raise TypeError("[ERROR] 'estado' debe ser una instancia de EstadoConfortTermico.")
            if not isinstance(weight, (int, float)) or weight <= 0:
                raise ValueError("[ERROR] 'weight' debe ser un número positivo.")
            self.estado = copy.deepcopy(estado)
            self.weight = float(weight)

        logging.debug(
            f"[DEBUG] Partícula creada: estado={self.estado.to_dict() if hasattr(self.estado, 'to_dict') else self.estado}, "
            f"weight={self.weight:.4f}"
        )

    def copy(self):
        logging.debug("[DEBUG] Creando copia de Partícula.")
        return Particula(copy.deepcopy(self.estado), self.weight)

    def resample(self, modelo_bayesiano=None):
        if self.weight < 0.05:
            logging.info(f"[INFO] Resample activado para partícula con peso bajo ({self.weight:.4f}).")
            generador = ParticulaGenerativa(self.estado)
            return Particula(generador.sample(), weight=1.0)

        if modelo_bayesiano:
            evidencia = {
                "tdb": self.estado.tdb if hasattr(self.estado, "tdb") else 25,
                "tr":  self.estado.tr if hasattr(self.estado, "tr") else 25,
                "rh":  self.estado.rh if hasattr(self.estado, "rh") else 50,
                "Tt":  self.estado.Tt if hasattr(self.estado, "Tt") else 25,
                "vr":  self.estado.vr if hasattr(self.estado, "vr") else 0.5,
                "met": self.estado.met if hasattr(self.estado, "met") else 1.0,
                "clo": self.estado.clo if hasattr(self.estado, "clo") else 1.0
            }
            pmv_probabilidad = modelo_bayesiano.inferir_variable("PMV", evidencia)
            self.weight *= max(pmv_probabilidad[1], 0.05)
            logging.debug(f"[DEBUG] Nuevo peso ajustado: {self.weight:.4f}")

        return self.copy()

    def __repr__(self):
        return f"Particula(estado={self.estado.to_dict() if hasattr(self.estado, 'to_dict') else self.estado}, weight={self.weight:.4f})"

    def __hash__(self):
        return hash(self.estado)

    def __eq__(self, other):
        return isinstance(other, Particula) and self.estado == other.estado

    def __getattr__(self, attr):
        return getattr(self.estado, attr)

class ParticulaGenerativa(pomdp_py.framework.basics.GenerativeDistribution):
    def __init__(self, estado_base, max_reintentos=5):
        if not isinstance(estado_base, EstadoConfortTermico):
            logging.critical(f"[CRITICAL] estado_base inválido: {estado_base} ({type(estado_base)})")
            raise TypeError("[ERROR] 'estado_base' debe ser una instancia de EstadoConfortTermico.")

        self.estado_base = estado_base
        self.max_reintentos = max_reintentos

    def sample(self):
        required = ["tdb", "tr", "rh", "Tt"]
        if not all(hasattr(self.estado_base, attr) for attr in required):
            logging.critical(f"[CRITICAL] estado_base incompleto: {self.estado_base} ({type(self.estado_base)})")
            if hasattr(self.estado_base, '__dict__'):
                logging.critical(f"[CRITICAL] Atributos de estado_base: {self.estado_base.__dict__}")
            raise ValueError("[ERROR] estado_base no tiene los atributos requeridos.")

        for intento in range(1, self.max_reintentos + 1):
            try:
                mu_tdb = self.estado_base.tdb
                mu_tr = self.estado_base.tr
                mu_hr = self.estado_base.rh
                mu_Tt = self.estado_base.Tt

                sigma_tdb, sigma_tr, sigma_hr, sigma_Tt = 1.5, 1.2, 3.0, 1.0

                tdb_nuevo = sample_tdb(mu_tdb, sigma_tdb)
                tr_nuevo = sample_tdb(mu_tr, sigma_tr)
                hr_nuevo = sample_hr_trunc(mu_hr, sigma_hr)
                Tt_nuevo = sample_Tt(mu_Tt, sigma_Tt)

                vr_nuevo = sample_vr(2.0, 0.5)
                met_nuevo = sample_met()
                clo_nuevo = sample_clo()

                nuevo_estado = EstadoConfortTermico(
                    tdb=tdb_nuevo,
                    tr=tr_nuevo,
                    rh=hr_nuevo,
                    Tt=Tt_nuevo,
                    vr=vr_nuevo,
                    met=met_nuevo,
                    clo=clo_nuevo,
                    pmv=None
                )

                pmv_valor = nuevo_estado.calcular_pmv()
                if pmv_valor is None or not np.isfinite(pmv_valor):
                    logging.warning(f"[WARNING] Intento {intento}: PMV inválido. Estado descartado: {nuevo_estado.to_dict()}")
                    continue

                logging.debug(f"[DEBUG] Estado sampleado válido (PMV={pmv_valor:.2f}): {nuevo_estado.to_dict()}")
                return nuevo_estado

            except Exception as e:
                logging.warning(f"[WARNING] Error al samplear estado en intento {intento}: {e}")

        raise ValueError(f"[ERROR] No se pudo generar un estado válido tras {self.max_reintentos} intentos.")

    def mpe(self):
        return self.estado_base

    def get_histogram(self):
        return {self.sample(): 1.0}


class ModeloBayesianoConfort:
    def __init__(self):
        self.model = BayesianModel([
            ('Tt', 'tdb'),
            ('Tt', 'PMV'),
            ('tdb', 'PMV'),
            ('tr', 'PMV'),
            ('rh', 'PMV'),
            ('vr', 'PMV'),
            ('met', 'PMV'),
            ('clo', 'PMV')
        ])
        self.infer = None
        self.construir_red()

    def construir_red(self):
        try:
            cpd_Tt = TabularCPD(variable='Tt', variable_card=2, values=[[0.5], [0.5]])
            cpd_tdb = TabularCPD(variable='tdb', variable_card=2,
                                 values=[[1.0, 0.0],
                                         [0.0, 1.0]],
                                 evidence=['Tt'], evidence_card=[2])
            cpd_tr  = TabularCPD(variable='tr', variable_card=2, values=[[0.8], [0.2]])
            cpd_rh  = TabularCPD(variable='rh', variable_card=2, values=[[0.7], [0.3]])
            cpd_vr  = TabularCPD(variable='vr', variable_card=2, values=[[0.5], [0.5]])
            cpd_met = TabularCPD(variable='met', variable_card=2, values=[[0.6], [0.4]])
            cpd_clo = TabularCPD(variable='clo', variable_card=2, values=[[0.6], [0.4]])

            values_pmv = np.full((2, 128), 0.5).tolist()  
            cpd_PMV = TabularCPD(variable='PMV', variable_card=2,
                                 values=values_pmv,
                                 evidence=['Tt', 'tdb', 'tr', 'rh', 'vr', 'met', 'clo'],
                                 evidence_card=[2, 2, 2, 2, 2, 2, 2])
            cpds = [cpd_Tt, cpd_tdb, cpd_tr, cpd_rh, cpd_vr, cpd_met, cpd_clo, cpd_PMV]
            self.model.add_cpds(*cpds)
            if not self.model.check_model():
                raise ValueError("[ERROR] El modelo Bayesiano no es válido.")
            self.infer = VariableElimination(self.model)
            logging.info("[INFO] Modelo Bayesiano configurado correctamente.")
        except Exception as e:
            logging.error(f"[ERROR] No se pudo inicializar el modelo bayesiano: {e}")
            self.infer = None

    def inferir_pmv(self, evidencia):
        try:
            if not isinstance(evidencia, dict):
                raise TypeError("[ERROR] La evidencia debe ser un diccionario.")
            if self.infer is None:
                raise RuntimeError("[ERROR] Inferencia no inicializada. Verifica construir_red().")
            logging.debug(f"[DEBUG] Realizando inferencia de PMV con evidencia: {evidencia}")
            query_result = self.infer.query(variables=['PMV'], evidence=evidencia)
            logging.debug(f"[DEBUG] Resultado de inferencia: {query_result.values}")
            return query_result.values
        except Exception as e:
            logging.error(f"[ERROR] Falló en la inferencia de PMV: {e}")
            return [0.5, 0.5]    


class ModeloDeTransicion(pomdp_py.TransitionModel):
    def __init__(self):
        super().__init__()
        logging.info("[DEBUG] Se ha creado una instancia de ModeloDeTransicion")
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            try:
                self.modelo_bayesiano = ModeloBayesianoConfort()
                logging.info("[DEBUG] Modelo bayesiano inicializado correctamente.")
                break
            except Exception as e:
                logging.warning(f"[WARNING] Error en inicialización del modelo bayesiano "
                                f"(intento {attempt}): {e}")
                if attempt == max_attempts:
                    raise RuntimeError(
                        "No se pudo inicializar el modelo bayesiano tras múltiples intentos."
                    )

    def sample(self, estado_actual, accion):
        logging.debug("[DEBUG][Transición] sample() invocado")
        if estado_actual is None:
            raise ValueError("[ERROR] 'estado_actual' es obligatorio.")
        
        base_state = copy.deepcopy(
            estado_actual.estado if hasattr(estado_actual, "estado") else estado_actual
        )
        accion.ejecutar_accion(base_state)
        
        generador = ParticulaGenerativa(base_state)
        nuevo_estado = generador.sample()
        
        nuevo_estado.pmv = nuevo_estado.calcular_pmv()
        return nuevo_estado

    def probability(self, estado_siguiente, estado_actual, accion):
        try:
            keys = ["tdb", "tr", "rh", "met", "clo"]
            thresholds = estado_actual.thresholds
            sigmas     = estado_actual.sigmas

            def soft_ev(val, thresh, sigma):
                diff = float(val) - thresh
                p0 = np.exp(-0.5 * (diff / sigma) ** 2)
                p1 = 1 - p0
                norm = p0 + p1 if (p0 + p1) != 0 else 1.0
                return [p0 / norm, p1 / norm]

            soft_evidence = {
                k: soft_ev(getattr(estado_actual, k), thresholds[k], sigmas[k])
                for k in keys
            }
            total_prob = 0.0

            for bits in product([0, 1], repeat=len(keys)):
                weight = np.prod([soft_evidence[keys[i]][bits[i]]
                                  for i in range(len(keys))])
                evidence = {keys[i]: bits[i] for i in range(len(keys))}
                pmv_probs = self.modelo_bayesiano.inferir_pmv(evidence)
                pmv_obs = 0 if estado_siguiente.pmv < 0 else 1
                total_prob += weight * float(pmv_probs[pmv_obs])

            sigma_act = 0.1
            p_t = norm.pdf(estado_siguiente.Tt, loc=accion.Tt, scale=sigma_act)
            p_v = norm.pdf(estado_siguiente.vr, loc=accion.vr, scale=sigma_act)

            prob = total_prob * p_t * p_v
            logging.info(f"[DEBUG] P(s'|s,a) ≈ {prob:.6f}")
            return prob

        except Exception as e:
            logging.error(f"[ERROR] Falló inferencia bayesiana en transición: {e}")
            raise
            

class ObservacionConfort(pomdp_py.Observation):
    def __init__(self, tdb, rh, Tt, vr):

        self.tdb = float(np.clip(tdb, 22.0, 26.0))
        self.rh  = float(np.clip(rh, 40.0, 60.0))
        self.Tt  = float(np.clip(Tt, 22.0, 26.0))
        self.vr  = float(np.clip(vr, 0.1, 0.5))
        logging.debug(f"[DEBUG] Observación inicializada: tdb={self.tdb}, rh={self.rh}, Tt={self.Tt}, vr={self.vr}")

    def __getitem__(self, key):
        if key == "tdb":
            return self.tdb
        elif key == "rh":
            return self.rh
        elif key == "Tt":
            return self.Tt
        elif key == "vr":
            return self.vr
        else:
            raise KeyError(f"Clave '{key}' no encontrada en ObservacionConfort.")

    def __setitem__(self, key, value):
        if key == "tdb":
            self.tdb = float(value)
        elif key == "rh":
            self.rh = float(value)
        elif key == "Tt":
            self.Tt = float(value)
        elif key == "vr":
            self.vr = float(value)
        else:
            raise KeyError(f"Clave '{key}' no se puede asignar en ObservacionConfort.")

    def simular_variacion(self):
        tdb_variacion = np.random.uniform(-0.5, 0.5)
        rh_variacion  = np.random.uniform(-1.0, 1.0)
        Tt_variacion  = np.random.uniform(-0.3, 0.3)
        vr_variacion  = np.random.uniform(-0.1, 0.1)
        obs_variada = ObservacionConfort(
            tdb = np.clip(self.tdb + tdb_variacion, 10.0, 35.0),
            rh  = np.clip(self.rh + rh_variacion, 20.0, 80.0),
            Tt  = np.clip(self.Tt + Tt_variacion, 16.0, 34.0),
            vr  = np.clip(self.vr + vr_variacion, 0.0, 2.0)
        )
        logging.info(f"[DEBUG] Observación variada generada: {obs_variada}")
        return obs_variada

    def __repr__(self):
        return f"ObservacionConfort(tdb={self.tdb:.2f}, rh={self.rh:.2f}, Tt={self.Tt:.2f}, vr={self.vr:.2f})"

    def __eq__(self, other):
        return (isinstance(other, ObservacionConfort) and
                math.isclose(self.tdb, other.tdb, abs_tol=1e-5) and
                math.isclose(self.rh, other.rh, abs_tol=1e-5) and
                math.isclose(self.Tt, other.Tt, abs_tol=1e-5) and
                math.isclose(self.vr, other.vr, abs_tol=1e-5))

    def __hash__(self):
        hash_val = hash((round(self.tdb, 2), round(self.rh, 2), round(self.Tt, 2), round(self.vr, 2)))
        logging.debug(f"[DEBUG] Hash de observación: {hash_val}")
        return hash_val


logging.basicConfig(level=logging.DEBUG)

class ModeloDeObservacion(pomdp_py.ObservationModel):
    def __init__(self, ruido_observacion=0.15):
        if not isinstance(ruido_observacion, (int, float)) or ruido_observacion < 0:
            raise ValueError("[ERROR] 'ruido_observacion' debe ser un número positivo.")
        self.ruido_observacion = ruido_observacion
        logging.info(f"[DEBUG] ModeloDeObservacion iniciado con ruido_observacion={self.ruido_observacion}")


    def sample(self, estado, action):
        logging.debug("[DEBUG][Observación] sample() fue invocado")

        if estado is None:
            raise ValueError("[ERROR] 'estado' es obligatorio y debe provenir del proceso.")

        required = ["tdb", "rh", "Tt", "vr"]
        if not all(hasattr(estado, attr) for attr in required):
            logging.critical(f"[CRITICAL] Estado malformado recibido por sample(): {estado} (tipo: {type(estado)})")
            if hasattr(estado, '__dict__'):
                logging.critical(f"[CRITICAL] Atributos internos: {estado.__dict__}")
            raise ValueError("[ERROR] El 'estado' debe tener los atributos 'tdb', 'rh', 'Tt' y 'vr'.")

        try:
            tdb_noise = np.random.normal(0.0, self.ruido_observacion)
            rh_noise  = np.random.normal(0.0, self.ruido_observacion)
            Tt_noise  = np.random.normal(0.0, self.ruido_observacion)
            vr_noise  = np.random.normal(0.0, self.ruido_observacion)
            tdb_obs = np.clip(estado.tdb + tdb_noise, 10.0, 35.0)
            rh_obs  = np.clip(estado.rh  + rh_noise, 20.0, 80.0)
            Tt_obs  = np.clip(estado.Tt  + Tt_noise, 16.0, 34.0)
            vr_obs  = np.clip(estado.vr  + vr_noise, 0.0, 2.0)

            obs = ObservacionConfort(tdb=tdb_obs, rh=rh_obs, Tt=Tt_obs, vr=vr_obs)

            obs.met = estado.met
            obs.clo = estado.clo

            logging.debug(f"[DEBUG] Observación generada: {obs}")
            return obs

        except Exception as e:
            logging.exception("[ERROR] Falló la generación de observación:")
            raise

    def probability(self, obs, estado, accion):
        try:
            if not isinstance(obs, ObservacionConfort):
                raise TypeError("[ERROR] 'obs' debe ser de tipo ObservacionConfort.")
            if not (hasattr(estado, "tdb") and hasattr(estado, "rh") and 
                    hasattr(estado, "Tt") and hasattr(estado, "vr")):
                raise TypeError("[ERROR] 'estado' debe tener los atributos 'tdb', 'rh', 'Tt' y 'vr'.")

            sigma = np.clip(self.ruido_observacion * 10.0, 5.0, 30.0)
            state_vals = {
                "tdb": estado.tdb,
                "rh":  estado.rh,
                "Tt":  estado.Tt,
                "vr":  estado.vr
            }
            weights = {"tdb": 0.02, "rh": 0.005, "Tt": 0.01, "vr": 0.01}
            log_prob = 0.0
            for attr in ("tdb", "rh", "Tt", "vr"):
                diff = getattr(obs, attr) - state_vals[attr]
                diff = np.clip(diff, -5.0, 5.0)
                comp = weights[attr] * (-0.5 * (diff / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi)))
                log_prob += comp
                logging.debug(f"[DEBUG] {attr}: diff={diff}, contribución_log={comp}")
            prob = np.exp(log_prob)
            if not np.isfinite(prob):
                logging.error("[ERROR] La probabilidad calculada no es finita, se asigna 0.0.")
                prob = 0.0
            logging.debug(f"[DEBUG] Probabilidad calculada: {prob}")
            return prob

        except Exception as e:
            logging.exception("[ERROR] Falló el cálculo de probabilidad")
            raise

    def get_all_observations(self):

        obs_list = []
        for _ in range(100):
            tdb = np.clip(np.random.normal(25.0, 5.0), 10.0, 35.0)
            rh  = np.clip(np.random.normal(50.0, 15.0), 20.0, 80.0)
            Tt  = np.clip(np.random.normal(25.0, 2.0), 16.0, 34.0)
            vr  = np.clip(np.random.normal(0.5, 0.3), 0.0, 2.0)
            obs_list.append(ObservacionConfort(tdb, rh, Tt, vr))
        logging.debug(f"[DEBUG] Generadas {len(obs_list)} observaciones posibles")
        return obs_list

def crear_belief_desde_sim(sim_state: dict, num_particulas=500, jitter_std=None):

    if jitter_std is None:
        jitter_std = {'tdb':0.3,'tr':0.3,'rh':2.0,'Tt':0.5,'vr':0.05,'met':0.05,'clo':0.05}
    particulas = []
    for _ in range(num_particulas):
        tdb = np.clip(np.random.normal(sim_state['tdb'], jitter_std['tdb']), 10.0, 35.0)
        tr  = np.clip(np.random.normal(sim_state['tr'],  jitter_std['tr']),  10.0, 35.0)
        rh  = np.clip(np.random.normal(sim_state['rh'],  jitter_std['rh']),   20.0, 80.0)
        Tt  = np.clip(np.random.normal(sim_state['Tt'],  jitter_std['Tt']),   16.0, 34.0)
        vr  = np.clip(np.random.normal(sim_state['vr'],  jitter_std['vr']),   0.0, 2.0)
        met = sim_state.get('met', 1.0)
        clo = sim_state.get('clo', 0.75)
        e = EstadoConfortTermico(tdb=tdb, tr=tr, rh=rh, Tt=Tt, vr=vr, met=met, clo=clo)
        particulas.append(Particula(e, weight=1.0/num_particulas))
    return pomdp_py.Particles(particulas)
    

class ModeloDeRecompensa(pomdp_py.RewardModel):
    def __init__(self, pmv_deseado=0.0, factor_pmv=1.0):
        super().__init__()
        if not isinstance(pmv_deseado, (int, float)):
            raise ValueError("[ERROR] 'pmv_deseado' debe ser un número (int o float).")
        if not isinstance(factor_pmv, (int, float)) or factor_pmv <= 0:
            raise ValueError("[ERROR] 'factor_pmv' debe ser un número positivo.")
        self.pmv_deseado = pmv_deseado
        self.factor_pmv = factor_pmv

    def reward(self, estado_siguiente, accion):    
        try:
            pmv_actual = getattr(estado_siguiente, "pmv", None)
            if pmv_actual is None or not np.isfinite(pmv_actual):
                if hasattr(estado_siguiente, "calcular_pmv"):
                    pmv_actual = estado_siguiente.calcular_pmv()
                else:
                    logging.error("[ERROR] El estado siguiente no tiene 'calcular_pmv'.")
                    return -20.0

            desviacion = abs(pmv_actual - self.pmv_deseado)
            if desviacion < 0.5:
                reward_val = 10.0 * self.factor_pmv
            elif desviacion < 1.5:
                reward_val = 5.0 * self.factor_pmv
            else:
                reward_val = -10.0 * self.factor_pmv

            logging.debug(
                f"[DEBUG][Reward] pmv_siguiente={pmv_actual:.3f}, "
                f"pmv_objetivo={self.pmv_deseado:.3f}, recompensa={reward_val:.3f}"
            )
            return reward_val

        except Exception as e:
            logging.error(f"[ERROR] Falló cálculo de recompensa: {e}")
            return -20.0

    def sample(self, estado_actual, accion, estado_siguiente):
        try:
            return self.reward(estado_siguiente, accion)
        except Exception as e:
            logging.error(f"[ERROR] Falló sample() en ModeloDeRecompensa: {e}")
            return -20.0
    

class ModeloDePolitica(pomdp_py.RolloutPolicy):
    def __init__(self, pmv_deseado: float):
        super().__init__()
        if not isinstance(pmv_deseado, (int, float)):
            raise ValueError("'pmv_deseado' debe ser numérico.")
        self.pmv_deseado = pmv_deseado

    def get_all_actions(self, state=None, history=None):
        acciones = [
            AccionConfortTermico(Tt=round(t, 2), vr=round(v, 2))
            for t in np.linspace(16.0, 34.0, 10)
            for v in np.linspace(0.0, 2.0, 5)
        ]
        if not acciones:
            logging.critical("[POLÍTICA] get_all_actions devolvió una lista vacía.")
        return acciones

    def sample(self, observacion, history=None):
        logging.debug(f"[ENTRADA] sample() recibido: {observacion!r}")
        try:
            acciones = self.get_all_actions(observacion, history)
            if not acciones:
                logging.critical("[POLÍTICA] No hay acciones disponibles — sample() cancelado.")
                raise RuntimeError("La política no tiene acciones para elegir.")

            faltantes = [var for var in ['tdb', 'tr', 'vr', 'rh', 'met', 'clo']
                         if not hasattr(observacion, var)]
            if faltantes:
                logging.critical(f"[POLÍTICA] Observación incompleta: faltan {faltantes}")
                raise ValueError(f"Observación sin atributos requeridos: {faltantes}")

            pmv_actual = (
                observacion.pmv if hasattr(observacion, "pmv") and observacion.pmv is not None
                else pmv_ppd_iso(
                    tdb=observacion.tdb,
                    tr=observacion.tr,
                    vr=observacion.vr,
                    rh=observacion.rh,
                    met=observacion.met,
                    clo=observacion.clo,
                    model="7730-2005"
                ).pmv
            )

            Tt_act, vr_act = observacion.Tt, observacion.vr
            Tt_min = getattr(observacion, "Tt_min", 16.0)
            Tt_max = getattr(observacion, "Tt_max", 34.0)
            vr_min = getattr(observacion, "vr_min", 0.0)
            vr_max = getattr(observacion, "vr_max", 2.0)

            if pmv_actual > self.pmv_deseado + 1.0:
                new_Tt = max(Tt_min, Tt_act - 0.5)
                new_vr = vr_act
            elif pmv_actual < self.pmv_deseado - 1.0:
                new_Tt = min(Tt_max, Tt_act + 0.5)
                new_vr = min(vr_max, vr_act + 0.5)
            else:
                new_Tt = Tt_act + random.uniform(-0.2, 0.2)
                new_vr = vr_act + random.uniform(-0.1, 0.1)

            new_Tt = round(min(max(new_Tt, Tt_min), Tt_max), 2)
            new_vr = round(min(max(new_vr, vr_min), vr_max), 2)

            accion = AccionConfortTermico(Tt=new_Tt, vr=new_vr)
            if accion is None:
                logging.critical("[POLÍTICA] Acción generada es None, lo cual no debe ocurrir.")
                raise RuntimeError("La política generó una acción nula.")
            logging.debug(f"[POLÍTICA] Acción seleccionada: {accion!r}")
            return accion

        except Exception as e:
            logging.exception("[ERROR] Falló selección de acción en sample()")
            raise
            

class ModeloDeEnvironment(pomdp_py.Environment):
    def __init__(self, modelo_transicion, modelo_observacion, modelo_recompensa, estado_inicial):
        super().__init__(modelo_transicion, modelo_observacion, modelo_recompensa, estado_inicial)
        
        if not isinstance(estado_inicial, EstadoConfortTermico):
            raise TypeError("[ERROR] 'estado_inicial' debe ser una instancia de EstadoConfortTermico.")
        
        logging.info("[INFO] Entorno inicializado correctamente.")

    def step(self, accion):
        try:
            logging.debug(f"[DEBUG] Ejecutando acción: {accion}")
            estado_siguiente = self.transition_model.sample(self.state, accion)
            observacion = self.observation_model.sample(accion, estado_siguiente)
            recompensa = self.reward_model.reward(estado_siguiente, accion)
            
            self.state = estado_siguiente 
            return observacion, recompensa, estado_siguiente

        except Exception as e:
            logging.error(f"[ERROR] Fallo en step(): {e}")
            return None, -20.0, self.state


class EntornoConfort(pomdp_py.Environment):
    def __init__(self, estados, estado_inicial):
        if not isinstance(estados, (list, set)) or not all(isinstance(e, EstadoConfortTermico) for e in estados):
            raise ValueError("[ERROR] 'estados' debe ser una lista o conjunto de instancias de EstadoConfortTermico.")
        if not isinstance(estado_inicial, EstadoConfortTermico):
            raise ValueError("[ERROR] 'estado_inicial' debe ser una instancia válida de EstadoConfortTermico.")

        self.estados = copy.deepcopy(estados)
        self.estado_inicial = estado_inicial

        super().__init__(estado_inicial)
        logging.debug(f"[DEBUG] EntornoConfort inicializado con {len(self.estados)} estados posibles.")

    def estados_posibles(self):
        return self.estados

    def step(self, accion):
        try:

            estado_siguiente = self.transition_model.sample(self.state, accion)
            observacion = self.observation_model.sample(accion, estado_siguiente)
            recompensa = self.reward_model.reward(estado_siguiente, accion)

            self.state = estado_siguiente

            return observacion, recompensa, estado_siguiente

        except Exception as e:
            logging.error(f"[ERROR] Fallo en step(): {e}")
            return None, -20.0, self.state

    def __repr__(self):
        estados_lista = list(self.estados) if isinstance(self.estados, set) else self.estados
        estados_ejemplo = estados_lista[:3] if len(estados_lista) > 3 else estados_lista
        return f"EntornoConfort(estado_inicial={self.estado_inicial}, estados={len(self.estados)}, ejemplo={estados_ejemplo})"        

class WrapperBelief:
    def __init__(self, belief: Any) -> None:
        self.belief = belief
        self.particles = getattr(belief, "particles", None)
        if self.particles is None:
            msg = "El belief proporcionado no tiene el atributo 'particles'."
            logging.error(msg)
            raise ValueError(msg)

    def sample(self) -> Any:
        if not self.particles:
            msg = "No hay partículas disponibles para muestrear."
            logging.error(msg)
            raise ValueError(msg)
        particle = random.choice(self.particles)
        logging.debug(f"Partícula muestreada: {particle}")
        return particle

    def __len__(self) -> int:
        return len(self.particles)

    def __iter__(self) -> Iterator:
        return iter(self.particles)

    def __getitem__(self, index: int) -> Any:
        return self.particles[index]

    def update(self, action: Any, observation: Any) -> Any:
        if hasattr(self.belief, "update"):
            result = self.belief.update(action, observation)
            logging.debug("Belief actualizado correctamente.")
            return result
        else:
            msg = "El belief original no tiene 'update' implementado."
            logging.error(msg)
            raise NotImplementedError(msg)

    def copy(self) -> "WrapperBelief":
        if hasattr(self.belief, "copy"):
            new_belief = self.belief.copy()
            logging.debug("Belief copiado correctamente.")
            return WrapperBelief(new_belief)
        else:
            msg = "El belief original no tiene 'copy' implementado."
            logging.error(msg)
            raise NotImplementedError(msg)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.belief, attr)
    
class CreenciaConfort(pomdp_py.Particles):
    def mpe(self):
        mejor = max(self.particles, key=lambda p: p.weight)
        logging.debug(f"[DEBUG] MPE candidate: {mejor}, tipo: {type(mejor)}")
        return mejor.estado
    

def crear_creencia_inicial_en_particulas(num_particulas: int = 1000) -> pomdp_py.Particles:
    try:
        logging.info(f"[INFO] Creando creencia inicial con {num_particulas} partículas...")
        particulas = []

        while len(particulas) < num_particulas:
            try:
                estado = EstadoConfortTermico.generar_estado_aleatorio()
                if isinstance(estado.pmv, (int, float)) and np.isfinite(estado.pmv):
                    p = Particula(estado, weight=1.0)
                    particulas.append(p)
                    logging.debug(f"[DEBUG] Partícula {len(particulas)} generada con PMV: {estado.pmv}")
            except Exception as ex:
                logging.error(f"[ERROR] Falló al generar partícula: {ex}")

        total_peso = sum(p.weight for p in particulas) or 1.0
        for p in particulas:
            p.weight /= total_peso

        belief = pomdp_py.Particles(particulas)
        logging.info("[INFO] Creencia inicial generada correctamente con partículas validadas.")
        return belief

    except Exception as e:
        logging.error("[ERROR] Falló en la generación de la creencia inicial:", exc_info=True)
        return pomdp_py.Particles([])
    
def validate_estado(estado: Any) -> Any:

    estado.tdb = np.clip(getattr(estado, "tdb", 24), 22, 26)
    estado.tr  = np.clip(getattr(estado, "tr", 24), 22, 26)
    estado.rh  = np.clip(getattr(estado, "rh", 50), 40, 60)
    estado.Tt  = np.clip(getattr(estado, "Tt", 25), 16, 33)
    estado.vr  = np.clip(getattr(estado, "vr", 0.3), 0.1, 0.5)
    estado.met = np.clip(getattr(estado, "met", 1.0), 1.0, 1.2)
    estado.clo = np.clip(getattr(estado, "clo", 0.75), 0.5, 1.0)
    return estado


def observacion_a_vector(observacion: Any) -> np.ndarray:

    orden = ["tdb", "tr", "rh", "Tt", "vr", "met", "clo"]

    if isinstance(observacion, dict):
        return np.array([observacion.get(clave, 0) for clave in orden], dtype=float)
    elif hasattr(observacion, '__dict__'):
        return np.array([getattr(observacion, clave, 0) for clave in orden], dtype=float)
    else:
        raise TypeError("[ERROR] Tipo de observación no soportado.")


def convert_belief_to_state_list(belief: Any) -> List[Any]:

    return [getattr(p, 'estado', p) for p in belief.particles]


def update_belief_with_resample(agente: Any, accion: Any, observacion_real: Any,
                                N_min: int = 50, reinject_ratio: float = 0.5) -> None:

    try:
        logging.debug(f"[DEBUG] agente.cur_belief = {agente.cur_belief!r}")
        if not (hasattr(agente, "policy") and hasattr(agente, "cur_belief") and 
                agente.cur_belief and agente.cur_belief.particles):
            logging.error("[ERROR] El agente no tiene política o creencia válida.")
            return

        raw = agente.cur_belief.particles or []
        n_raw = len(raw)
        logging.debug(f"[DEBUG] Paso1 raw size = {n_raw}")

        particulas_seguras = []
        for obj in raw:
            if isinstance(obj, Particula):
                p = deepcopy(obj)
            else:
                estado = deepcopy(obj)
                p = Particula(estado)
                p.weight = 1.0 / max(n_raw, 1)
            particulas_seguras.append(p)
        logging.debug(f"[DEBUG] Paso1 particulas_seguras = {len(particulas_seguras)}")

        for p in particulas_seguras:
            try:
                if p.estado is not None:
                    p.estado.pmv = p.estado.calcular_pmv()
            except Exception:
                logging.exception("[ERROR] Falló calcular PMV en Paso2")

        particulas_validas = [p for p in particulas_seguras if p.estado is not None]
        logging.debug(f"[DEBUG] Paso3 particulas_validas = {len(particulas_validas)}")

        if len(particulas_validas) < N_min:
            logging.warning(f"[FALLBACK] Solo {len(particulas_validas)} válidas (<{N_min}); inyectando {N_min} fresh.")
            fresh = []
            for _ in range(N_min):
                try:
                    e = EstadoConfortTermico.generar_estado_aleatorio()
                    p_new = Particula(e)
                    p_new.weight = 1.0 / N_min
                    fresh.append(p_new)
                except Exception:
                    logging.exception("[ERROR] Falló crear partícula fresh")
            agente.set_belief(pomdp_py.Particles(deepcopy(fresh)))
            logging.info("[INFO] Fallback: creencia reemplazada con partículas fresh.")
            return

        missing = n_raw - len(particulas_validas)
        if missing > 0:
            logging.warning(f"[WARNING] Rellenando {missing} faltantes")
            for _ in range(missing):
                try:
                    e = EstadoConfortTermico.generar_estado_aleatorio()
                    p_new = Particula(e)
                    p_new.weight = 1.0 / n_raw
                    particulas_validas.append(p_new)
                except Exception:
                    logging.exception("[ERROR] Falló crear partícula faltante")
        logging.debug(f"[DEBUG] Paso4 valid size tras rellenar = {len(particulas_validas)}")

        try:
            particulas_finales = roughen_particles(particulas_validas) or []
        except Exception:
            logging.exception("[ERROR] Falló roughen_particles")
            return
        logging.debug(f"[DEBUG] Paso5 particulas_finales = {len(particulas_finales)}")

        for i, p in enumerate(particulas_finales):
            if not isinstance(p.estado, EstadoConfortTermico):
                logging.warning(f"[DEBUG] Partícula {i}: estado no es instancia de EstadoConfortTermico.")
                if isinstance(p.estado, dict):
                    try:
                        p.estado = EstadoConfortTermico(
                            tdb=p.estado.get("tdb", 24),
                            tr=p.estado.get("tr", 24),
                            rh=p.estado.get("rh", 50),
                            Tt=p.estado.get("Tt", 25),
                            vr=p.estado.get("vr", 0.3),
                            met=p.estado.get("met", 1.0),
                            clo=p.estado.get("clo", 0.75),
                            pmv=p.estado.get("pmv", None)
                        )
                        logging.debug(f"[DEBUG] Partícula {i}: estado convertido correctamente.")
                    except Exception as ex:
                        logging.error(f"[ERROR] Fallo al convertir el estado de la partícula {i}: {ex}")
                else:
                    try:
                        p.estado = EstadoConfortTermico.generar_estado_aleatorio()
                        logging.debug(f"[DEBUG] Partícula {i}: estado regenerado correctamente.")
                    except Exception as ex:
                        logging.error(f"[ERROR] Fallo al regenerar el estado de la partícula {i}: {ex}")

        modelo_obs = ModeloDeObservacion()
        for p in particulas_finales:
            try:
                p.weight = modelo_obs.probability(observacion_real, p.estado, accion)
            except Exception:
                logging.exception("[ERROR] Falló calcular peso en Paso7")

        try:
            if all(p.weight == 0 for p in particulas_finales):
                logging.warning("[WARNING] Todos los pesos son cero antes de softmax. Asignando pesos uniformes.")
                for p in particulas_finales:
                    p.weight = 1.0

            log_ps = [np.log(p.weight) if p.weight > 0 else -np.inf for p in particulas_finales]
            m = max(log_ps) if log_ps else 0
            ws = [np.exp(lp - m) if lp != -np.inf else 0.0 for lp in log_ps]
            S = sum(ws) or 1.0
            for i, p in enumerate(particulas_finales):
                p.weight = max(ws[i] / S, 1e-6)
            Z = sum(p.weight for p in particulas_finales) or 1.0
            for p in particulas_finales:
                p.weight /= Z
        except Exception:
            logging.exception("[ERROR] Falló en softmax Paso8")

        try:
            n_eff = calculate_n_eff(particulas_finales)
            if n_eff < 5:
                extra = max(10, int(reinject_ratio * len(particulas_finales)))
                for _ in range(extra):
                    e = EstadoConfortTermico.generar_estado_aleatorio()
                    p_extra = Particula(e)
                    p_extra.weight = modelo_obs.probability(observacion_real, e, accion)
                    particulas_finales.append(p_extra)
            if len(particulas_finales) < N_min:
                faltan2 = N_min - len(particulas_finales)
                for _ in range(faltan2):
                    e = EstadoConfortTermico.generar_estado_aleatorio()
                    p_new = Particula(e)
                    p_new.weight = modelo_obs.probability(observacion_real, e, accion)
                    particulas_finales.append(p_new)
        except Exception:
            logging.exception("[ERROR] Falló refuerzo masivo Paso10 o asegurar N_min Paso11")

        try:
            Zf = sum(p.weight for p in particulas_finales) + 1e-12
            for p in particulas_finales:
                p.weight /= Zf
            neff_final = calculate_n_eff(particulas_finales)
            logging.debug(f"[TRACE] n_eff final={neff_final:.2f}, #={len(particulas_finales)}")
        except Exception:
            logging.exception("[ERROR] Falló normalizar Paso12")

        agente.set_belief(pomdp_py.Particles(deepcopy(particulas_finales)))
        logging.info("[INFO] update_belief_with_resample finalizó con éxito.")

    except Exception:
        tb = traceback.format_exc()
        logging.error("¡¡Excepción en update_belief_with_resample!!\n%s", tb)
        raise
        

def agregar_jitter(particulas, sigma_tdb=0.5, sigma_rh=3, sigma_Tt=1.0, sigma_vr=0.1, sigma_met=0.05, sigma_clo=0.05):

    for p in particulas:
        p.estado.tdb += np.random.normal(0, sigma_tdb)
        p.estado.tdb = np.clip(p.estado.tdb, 22, 26)    
        p.estado.rh += np.random.normal(0, sigma_rh)
        p.estado.rh = np.clip(p.estado.rh, 40, 60)       
        p.estado.Tt += np.random.normal(0, sigma_Tt)
        p.estado.Tt = np.clip(p.estado.Tt, 16, 33)        
        p.estado.vr += np.random.normal(0, sigma_vr)
        p.estado.vr = np.clip(p.estado.vr, 0.1, 0.5)        
        p.estado.met += np.random.normal(0, sigma_met)
        p.estado.met = np.clip(p.estado.met, 1.0, 1.2)        
        p.estado.clo += np.random.normal(0, sigma_clo)
        p.estado.clo = np.clip(p.estado.clo, 0.5, 1.0)        
        p.estado.pmv = p.estado.calcular_pmv()


class NodoVF(pomdp_py.algorithms.po_uct.QNode):
    def __init__(self, N=0, V=0.0):
        super().__init__(N, V)
        self.visits = N
        self.value = V
        self.children = dict()

    def update(self, value):
        self.visits += 1
        self.value += value

    def best_action(self):
        from collections import defaultdict

        if not self.children:
            return None

        acumulado = defaultdict(lambda: {"visits": 0, "value": 0.0})
        for (accion, _), nodo in self.children.items():
            acumulado[accion]["visits"] += getattr(nodo, "visits", 0)
            acumulado[accion]["value"] += getattr(nodo, "value", 0.0)

        if not acumulado:
            return None

        mejor_accion = max(acumulado.items(), key=lambda x: x[1]["visits"])[0]
        logging.info(f"[BEST_ACTION] Acción con más visitas: {mejor_accion}")
        return mejor_accion


from pomdp_py import RolloutPolicy

class RolloutPolicySoftmax(RolloutPolicy): 
    def __init__(self, acciones, tau=1.0):
        self.acciones = acciones
        self.tau = tau

    def sample(self, state, history):
        import random, math
        valores = [random.uniform(0.0, 1.0) for _ in self.acciones]
        exp_valores = [math.exp(v / self.tau) for v in valores]
        total = sum(exp_valores)
        probs = [v / total for v in exp_valores]
        seleccion = random.choices(self.acciones, weights=probs, k=1)[0]
        logging.debug(f"[ROLLOUT SOFTMAX] Acción seleccionada: {seleccion}")
        return seleccion


class PlanificadorPOUCT_VF(pomdp_py.POUCT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree = None
        self.num_sims = kwargs.get("num_sims", 0)
        self.exploration_const = kwargs.get("exploration_const", 300)

        logging.debug("[DEBUG] Constructor activo: PlanificadorPOUCT_VF inicializado")
        logging.debug(f"[DEBUG] Rollout policy: {type(self.rollout_policy)}")
        logging.warning(" Clase PlanificadorPOUCT_VF instanciada en tiempo real desde línea X")

        def mi_hook(belief, tree, sim, max_depth):
            logging.warning(f" Sim #{sim} lanzada a profundidad {max_depth}")
        self._debug_hook = mi_hook

    def plan(self, agent):
        logging.debug("[DEBUG] Entrando al método plan() del PlanificadorPOUCT_VF")

        try:
            acciones = agent.policy.get_all_actions()
            logging.debug(f"[DEBUG] Total de acciones posibles: {len(acciones)}")
            logging.debug(f"[DEBUG] Ejemplos de acciones: {[str(a) for a in acciones[:5]]}")
        except Exception as e:
            logging.error(f"[ERROR] Error al obtener acciones: {e}")
            acciones = []

        try:
            mpe = agent.cur_belief.mpe()
            logging.debug(f"[DEBUG] MPE desde creencia actual: {mpe}")
        except Exception as e:
            logging.error(f"[ERROR] Fallo al obtener MPE: {e}")

        logging.debug(f"[DEBUG] num_sims configuradas: {self.num_sims}")
        num_particles = len(getattr(agent.cur_belief, "particles", []))
        logging.debug(f"[DEBUG] Número de partículas en creencia: {num_particles}")
        logging.info(f"[INFO] Preparando rollout: belief={type(agent.cur_belief)}, acciones={len(acciones)}")

        try:
            action = self._plan(agent)
            self.tree = self.get_tree()

            if not action:
                logging.critical("[PLAN VF] No se pudo seleccionar ninguna acción. El árbol puede estar vacío.")
            else:
                logging.debug(f"[DEBUG] Acción elegida: {action}")

            if self.tree and hasattr(self.tree, "children"):
                hijos = list(self.tree.children.keys())
                logging.info(f"[INSPECCIÓN] Claves de hijos en el árbol raíz: {hijos}")
            else:
                logging.warning("[INSPECCIÓN] Árbol raíz sin hijos detectado.")

            return action
        except Exception as e:
            logging.exception(f"[ERROR] Excepción durante planificación personalizada: {e}")
            self.tree = None
            return None

    def _plan(self, agent):
        from collections import defaultdict

        self.agent = agent
        self.tree = NodoVF()

        for sim in range(self.num_sims):
            try:
                state = agent.cur_belief.random()
                logging.debug(f"[PLAN VF] Sim #{sim} usando estado: {state}")
                self.simulate(state, [], self.tree, 0, self.max_depth)
            except Exception as e:
                logging.exception(f"[PLAN VF] Simulación #{sim} falló: {e}")

        if not self.tree.children:
            logging.critical("[PLAN VF] El árbol no tiene hijos al finalizar las simulaciones.")
            return None

        logging.debug(f"[PLAN VF] Total de hijos en raíz: {len(self.tree.children)}")
        logging.info(f"[VERIFICACIÓN] Hijos en raíz: {list(self.tree.children.keys())}")

        acciones_exploradas = [accion for (accion, _) in self.tree.children.keys()]
        total = len(acciones_exploradas)
        conteo = defaultdict(int)
        for acc in acciones_exploradas:
            conteo[acc] += 1

        logging.info(f"[ojo] Total de nodos en raíz: {total}")
        for acc, n in sorted(conteo.items(), key=lambda x: x[1], reverse=True):
            porcentaje = (n / total) * 100
            logging.info(f"↳ Acción={acc} aparece en {n} nodos ({porcentaje:.1f}%)")

        valores_por_accion = defaultdict(list)
        for (accion, obs), nodo in self.tree.children.items():
            visitas = getattr(nodo, "visits", 0)
            valor = getattr(nodo, "value", 0.0)
            valores_por_accion[accion].append(valor)
            logging.debug(f"[PLAN VF] Hijo: acción={accion}, obs={obs}, visitas={visitas}, valor acumulado={valor:.2f}")

        accion = self.tree.best_action()

        if accion is None:
            logging.error("[PLAN VF] best_action() devolvió None — ninguna acción supera en visitas.")
            return None

        logging.info("[RESUMEN VISITAS POR ACCIÓN]")
        contador_acciones = defaultdict(int)
        for (accion_hija, _), nodo in self.tree.children.items():
            contador_acciones[accion_hija] += getattr(nodo, "visits", 0)

        if contador_acciones:
            max_visitas = max(contador_acciones.values(), default=0)
            for accion_item, visitas in sorted(contador_acciones.items(), key=lambda x: x[1], reverse=True):
                barra = "|" * int((visitas / max_visitas) * 20)
                logging.info(f"{str(accion_item):<35} {barra:<20} ({visitas} visitas)")
        else:
            logging.warning("[Advertencia] Ninguna acción fue visitada — árbol sin expansión.")

        logging.info("[ESTADÍSTICAS DE RECOMPENSA POR ACCIÓN]")
        for accion, valores in sorted(valores_por_accion.items(), key=lambda x: len(x[1]), reverse=True):
            if valores:
                media = sum(valores) / len(valores)
                varianza = sum((v - media) ** 2 for v in valores) / len(valores)
                logging.info(f"↳ Acción={str(accion):<35} Media={media:>6.2f}  Varianza={varianza:>6.2f}  N={len(valores)}")
            else:
                logging.info(f"↳ Acción={accion:<35} Sin valores registrados.")

        visitas_total = sum(getattr(nodo, "visits", 0) for nodo in self.tree.children.values())
        logging.info(f"[RESUMEN FINAL] Acción elegida: {accion}")
        logging.info(f"[RESUMEN FINAL] Total de hijos: {len(self.tree.children)}, visitas acumuladas: {visitas_total}")

        try:
            estado_mpe = agent.cur_belief.mpe()
            pmv_mpe = getattr(estado_mpe, "pmv", None)
            if pmv_mpe is not None:
                logging.info(f"[RESUMEN FINAL] Acción elegida: {accion}, PMV (MPE): {pmv_mpe:.2f}")
            else:
                logging.info(f"[RESUMEN FINAL] Acción elegida: {accion}, PMV: No disponible")
        except Exception as e:
            logging.warning(f"[RESUMEN FINAL] Acción elegida: {accion}, PMV: Error al obtener MPE ({e})")

        return accion

    def select_action(self, node, state, exploration_const):
        try:
            acciones = self.agent.policy.get_all_actions()
        except Exception as e:
            logging.error(f"[select_action] No se pudo acceder a las acciones del agente: {e}")
            return None

        acciones_nuevas = [a for a in acciones if (a, None) not in node.children]
        if acciones_nuevas:
            seleccion = random.choice(acciones_nuevas)
            logging.debug(f"[SELECT] Acción no explorada seleccionada aleatoriamente: {seleccion}")
            return seleccion

        best_score = float("-inf")
        best_action = None

        for accion in acciones:
            qnode = node.children.get((accion, None))
            Q = getattr(qnode, "value", 0.0) / max(getattr(qnode, "visits", 1), 1)
            N = getattr(node, "visits", 1)
            n = getattr(qnode, "visits", 0)
            score = Q + exploration_const * ((N ** 0.5) / (1 + n))

            if score > best_score:
                best_score = score
                best_action = accion

        logging.debug(f"[SELECT] Acción seleccionada con UCT: {best_action}")
        return best_action

    def simulate(self, state, history, tree_node, depth, max_depth):
        try:
            if tree_node is None:
                logging.warning("[SIMULATE] Nodo del árbol nulo. Deteniendo simulación.")
                return 0.0

            if depth >= max_depth:
                logging.debug("[SIMULATE] Se alcanzó la profundidad máxima.")
                return 0.0

            action = self.select_action(tree_node, state, self.exploration_const)
            if action is None:
                logging.critical("[SIMULATE] Acción seleccionada es None. Interrumpiendo simulación.")
                return 0.0

            next_state = self.agent.transition_model.sample(state, action)
            if next_state is None:
                logging.warning(f"[SIMULATE] Transición fallida desde {state} con {action}")
                return 0.0

            observation = self.agent.observation_model.sample(next_state, action)
            reward = self.agent.reward_model.reward(state, action)
        
            clave = (action, observation)

            if clave not in tree_node.children:
                logging.debug(f"[SIMULATE] Expansión en nodo hoja con acción={action}, obs={observation}")
                tree_node.children[clave] = NodoVF()
                future_value = self.rollout(next_state, history + [clave], depth + 1, max_depth)
            else:
                future_value = self.simulate(
                    next_state, history + [clave],
                    tree_node.children[clave], depth + 1, max_depth
                    )

            tree_node.children[clave].update(future_value)

            return reward + self.discount_factor * future_value

        except Exception as e:
            logging.exception(f"[SIMULATE] Excepción en simulate: {e}")
            return -20.0

    def rollout(self, state, history, depth, max_depth):
        try:
            logging.debug(f"[ROLLOUT] Profundidad actual: {depth}/{max_depth}")
            if self.rollout_policy is None:
                logging.critical("[ROLLOUT] No hay política de rollout definida.")
                return 0.0

            action = self.rollout_policy.sample(state, history)
            if action is None:
                logging.critical(f"[ROLLOUT] Acción es None en profundidad {depth}. Historial: {history}")
                return 0.0

            next_state = self.agent.transition_model.sample(state, action)
            if next_state is None:
                logging.warning(f"[ROLLOUT] Transición inválida con acción {action} desde estado {state}")
                return 0.0

            observation = self.agent.observation_model.sample(next_state, action)
            reward = self.agent.reward_model.reward(state, action)

            if depth >= max_depth:
                logging.debug("[ROLLOUT] Límite de profundidad alcanzado.")
                return reward

            return reward + self.discount_factor * self.rollout(
                next_state, history + [(action, observation)], depth + 1, max_depth
            )

        except Exception as e:
            logging.exception(f"[ROLLOUT] Excepción durante rollout: {e}")
            return -20.0

    def get_tree(self):
        return self.tree or getattr(self, "_tree", None) or getattr(self, "_search_tree", None)
    

class AgentePersonalizado(pomdp_py.Agent):
    def __init__(self, belief, policy, transition_model, observation_model, reward_model, pmv, nombre=None, planificador=None):
        self.nombre = nombre or "AgenteDesconocido"
        try:
            if not hasattr(belief, "particles"):
                raise TypeError("'belief' debe tener el atributo 'particles'.")

            belief_corr = []
            for b in belief.particles:
                if isinstance(b, Particula):
                    belief_corr.append(b)
                elif isinstance(b, dict):
                    estado = EstadoConfortTermico(**b)
                    belief_corr.append(Particula(estado))
                else:
                    belief_corr.append(Particula(b))
            self.set_belief(CreenciaConfort(belief_corr))

            if not isinstance(policy, ModeloDePolitica):
                raise TypeError("'policy' debe ser ModeloDePolitica.")

            for modelo, nombre_m in [(transition_model, "transición"),
                                     (observation_model, "observación"),
                                     (reward_model, "recompensa")]:
                if modelo is None:
                    raise ValueError(f"[ERROR] Modelo de {nombre_m} no puede ser None.")

            self.pmv_objetivo = pmv
            self.policy = policy

            super().__init__(self.cur_belief, policy, transition_model, observation_model, reward_model)

            self.rollout_policy = policy
            self.planificador = planificador or PlanificadorPOUCT_VF(
                max_depth=10,
                num_sims=500,
                discount_factor=0.95,
                rollout_policy=self.rollout_policy
            )

            logging.info(f"[INFO] {self.nombre} inicializado (pmv_objetivo={pmv}).")
        except Exception:
            logging.exception("[ERROR] Falló la inicialización de AgentePersonalizado")
            raise

    def get_current_state(self):
        max_attempts = 5
        attempts = 0
        required_attrs = ["tdb", "rh", "Tt", "vr"]

        while attempts < max_attempts:
            try:
                candidate = self.cur_belief.mpe()
                logging.debug(f"[DEBUG] MPE candidate: {candidate}, tipo: {type(candidate)}")

                if hasattr(candidate, '__dict__'):
                    logging.debug(f"[DEBUG] candidate.__dict__: {candidate.__dict__}")

                if isinstance(candidate, Particula):
                    state = candidate.estado
                elif isinstance(candidate, dict):
                    logging.warning("[WARNING] Candidate es dict. Reconstruyendo como EstadoConfortTermico.")
                    state = EstadoConfortTermico(**candidate)
                elif all(hasattr(candidate, attr) for attr in required_attrs):
                    state = candidate
                else:
                    logging.critical(f"[CRITICAL] Candidate inválido sin forma reconocida: {candidate}, tipo: {type(candidate)}")
                    raise ValueError("[ERROR] Candidate no tiene la forma esperada.")

                if not all(hasattr(state, attr) for attr in required_attrs):
                    logging.critical(f"[CRITICAL] Estado malformado (falta atributos): {state}, tipo: {type(state)}")
                    if hasattr(state, '__dict__'):
                        logging.critical(f"[CRITICAL] Estado.__dict__: {state.__dict__}")
                    raise ValueError("[ERROR] El 'estado' debe tener los atributos 'tdb', 'rh', 'Tt' y 'vr'.")

                if not hasattr(state, "pmv") or not isinstance(state.pmv, (float, int)) or not np.isfinite(state.pmv):
                    logging.critical(f"[CRITICAL] 'pmv' inválido: {state.pmv}")
                    raise ValueError("[ERROR] 'pmv' inválido en estado.")

                logging.debug(f"[DEBUG] Estado válido obtenido: {state}")
                return state

            except Exception as e:
                logging.error(f"[ERROR] get_current_state intento {attempts+1}: {e}")
            attempts += 1

        logging.warning("[WARNING] No se pudo obtener un estado válido tras múltiples intentos. Se usará fallback.")
        for p in self.cur_belief.particles:
            if hasattr(p, "estado") and all(hasattr(p.estado, attr) for attr in required_attrs):
                logging.info("[INFO] Fallback: usando primera partícula válida.")
                return p.estado

        raise ValueError("[ERROR] No se pudo obtener un estado válido ni siquiera con fallback.")

    def plan(self):
        max_attempts = 5
        attempts = 0

        while attempts < max_attempts:
            try:
                logging.debug(f"[DEBUG] Planificador: {type(self.planificador)}")
                logging.debug(f"[DEBUG] # Partículas: {len(self.cur_belief.particles)}")

                mpe = self.cur_belief.mpe()
                logging.debug(f"[DEBUG] Resultado de mpe(): {mpe}, tipo: {type(mpe)}")

                if hasattr(mpe, 'estado'):
                    logging.debug(f"[DEBUG] mpe().estado: {mpe.estado}, tipo: {type(mpe.estado)}")

                estado_actual = self.get_current_state()
                logging.debug(f"[DEBUG] Estado actual obtenido en plan(): {estado_actual}")

                print(f"[DEBUG] Clase del planificador: {type(self.planificador)}")
                print(f"[DEBUG] Tipo de creencia actual: {type(self.cur_belief)}")

                try:
                    accion = self.planificador.plan(self)
                except Exception as inner_e:
                    logging.error(f"[ERROR] Excepción interna en self.planificador.plan(): {type(inner_e).__name__} - {inner_e}")
                    import traceback
                    traceback.print_exc()
                    raise

                logging.info(f"[INFO] ¿Planificador tiene get_tree? {hasattr(self.planificador, 'get_tree')}")
                if hasattr(self.planificador, "get_tree"):
                    tree = self.planificador.get_tree()
                    logging.debug(f"[DEBUG] Tipo de nodo raíz: {type(tree)}")
                    logging.debug(f"[DEBUG] Atributos disponibles en nodo raíz: {dir(tree)}")

                    if not tree or not getattr(tree, "children", None):
                        logging.warning("[WARNING] El árbol del planificador está vacío o sin acciones exploradas.")
                    else:
                        logging.info("[INFO] El planificador actual no implementa 'get_tree()'. No se inspeccionará el árbol.")

                if accion is None:
                    raise ValueError("planner.plan() devolvió None")

                logging.debug(f"[DEBUG] Acción seleccionada: {accion}")
                return accion

            except Exception as e:
                logging.error(f"[ERROR] Falló plan() en intento {attempts+1}: {e}")
                import traceback
                traceback.print_exc()

            attempts += 1

        raise ValueError("No se pudo generar una acción válida tras múltiples intentos en plan().")

    def actualizar_belief(self, accion, observacion):
        try:
            update_belief_with_resample(self, accion, observacion)
            logging.debug("[DEBUG] Creencia actualizada.")
        except Exception:
            logging.exception("[ERROR] Falló actualizar_belief()")
            raise

    def set_state(self, nuevo_estado):
        try:
            if not isinstance(nuevo_estado, EstadoConfortTermico):
                raise TypeError("El nuevo estado debe ser una instancia de EstadoConfortTermico.")
            nueva_particula = Particula(nuevo_estado)
            nueva_creencia = CreenciaConfort([nueva_particula])
            self.set_belief(nueva_creencia)
            logging.info("[INFO] Estado del agente reiniciado correctamente con nueva creencia.")
        except Exception:
            logging.exception("[ERROR] No se pudo establecer el nuevo estado.")
            raise           

class ProblemaConfortTermico(pomdp_py.POMDP):
    def __init__(self, agente, entorno):
        if not isinstance(agente, AgentePersonalizado):
            raise ValueError("[ERROR] 'agente' debe ser una instancia de AgentePersonalizado.")
        if not isinstance(entorno, EntornoConfort):
            raise ValueError("[ERROR] 'entorno' debe ser una instancia de EntornoConfort.")
        super().__init__(agente, entorno)
        self.agente = agente
        self.entorno = entorno
        self.belief_inicial = agente.cur_belief
        logging.info("[INFO] ProblemaConfortTermico inicializado correctamente.")

    @staticmethod
    def crear(ruido_observacion,
              pmv_deseado,
              num_particulas=1000,
              num_estados=50,
              use_greedy: bool = False):
        try:
            logging.info("[INFO] Configurando el problema de confort térmico...")

            lista_estados = [
                EstadoConfortTermico.generar_estado_aleatorio()
                for _ in range(num_particulas)
            ]
            creencia_inicial = pomdp_py.Particles([Particula(e) for e in lista_estados])

            policy_model      = ModeloDePolitica(pmv_deseado=pmv_deseado)
            transition_model  = ModeloDeTransicion()
            observation_model = ModeloDeObservacion(ruido_observacion=ruido_observacion)
            reward_model      = ModeloDeRecompensa(pmv_deseado=pmv_deseado)

            plan_pouct = PlanificadorPOUCT_VF(
                max_depth=15,
                discount_factor=0.95,
                num_sims=1000,
                exploration_const=1.2,
                rollout_policy=RolloutPolicySoftmax(
                    policy_model.get_all_actions(),
                    tau=0.7
                )
            )
            plan_greedy = PlanificadorGreedy(
                acciones=policy_model.get_all_actions(),
                modelo_de_transicion=transition_model
            )

            planificador_seleccionado = plan_greedy if use_greedy else plan_pouct

            agente = AgentePersonalizado(
                nombre="AgenteConfortTermico",
                pmv=pmv_deseado,
                belief=creencia_inicial,
                policy=policy_model,
                transition_model=transition_model,
                observation_model=observation_model,
                reward_model=reward_model,
                planificador=planificador_seleccionado   # ← aquí
            )
            logging.info("[INFO] Agente configurado con éxito.")

            estados_posibles = [
                EstadoConfortTermico.generar_estado_aleatorio()
                for _ in range(num_estados)
            ]
            estado_inicial = np.random.choice(estados_posibles)
            entorno = EntornoConfort(
                estados=estados_posibles,
                estado_inicial=estado_inicial
            )
            logging.info("[INFO] Entorno configurado con éxito.")

            problema = ProblemaConfortTermico(agente, entorno)
            logging.info("[INFO] ProblemaConfortTermico creado con éxito.")
            return problema

        except Exception as e:
            logging.error("[ERROR] Ocurrió un error durante la creación del problema:", exc_info=True)
            raise

_agent_singleton = None

def construir_agente_si_no_existe(belief=None, ruido_observacion=0.5, pmv_deseado=0.0,
                                  num_particulas=1000, num_estados=50, use_greedy: bool = False):

    global _agent_singleton
    if _agent_singleton is not None:
        return _agent_singleton

    try:
        policy_model      = ModeloDePolitica(pmv_deseado=pmv_deseado)
        transition_model  = ModeloDeTransicion()
        observation_model = ModeloDeObservacion(ruido_observacion=ruido_observacion)
        reward_model      = ModeloDeRecompensa(pmv_deseado=pmv_deseado)

        plan_pouct = PlanificadorPOUCT_VF(
            max_depth=15,
            discount_factor=0.95,
            num_sims=1000,
            exploration_const=1.2,
            rollout_policy=RolloutPolicySoftmax(
                policy_model.get_all_actions(),
                tau=0.7
            )
        )
        plan_greedy = PlanificadorGreedy(
            acciones=policy_model.get_all_actions(),
            modelo_de_transicion=transition_model
        )
        planificador_seleccionado = plan_greedy if use_greedy else plan_pouct

        if belief is None:
            lista_estados = [
                EstadoConfortTermico.generar_estado_aleatorio()
                for _ in range(num_particulas)
            ]
            creencia_inicial = pomdp_py.Particles([Particula(e) for e in lista_estados])
        else:
            creencia_inicial = belief

        agente = AgentePersonalizado(
            nombre="AgenteConfortTermico",
            pmv=pmv_deseado,
            belief=creencia_inicial,
            policy=policy_model,
            transition_model=transition_model,
            observation_model=observation_model,
            reward_model=reward_model,
            planificador=planificador_seleccionado
        )

        _agent_singleton = agente
        return _agent_singleton

    except Exception as e:
        class _FallbackAgent:
            def __init__(self, default_new_Tt=24.0, default_vr=0.5):
                self.default_new_Tt = float(default_new_Tt)
                self.default_vr = float(default_vr)
                self.belief = None
            def set_belief(self, belief):
                self.belief = belief
            def plan(self):
                class A: pass
                a = A()
                a.new_Tt = self.default_new_Tt
                a.vr = self.default_vr
                return a
        _agent_singleton = _FallbackAgent()
        return _agent_singleton

            
def safe_plan_call(planificador, agente):
    logging.info("[DEBUG] Verificando agente y planificador antes de ejecutar plan()...")

    if not hasattr(agente, "cur_belief") or not hasattr(agente, "policy"):
        logging.error("[ERROR] El objeto agente no contiene las propiedades necesarias (cur_belief, policy).")
        return AccionConfortTermico(cambio_de_Tt=0.0, cambio_de_flujo_de_aire=0.0)

    logging.debug(f"[DEBUG] Tipo de creencia del agente: {type(agente.cur_belief)}")

    if not agente.cur_belief.particles:
        logging.warning("[WARNING] `belief.particles` está vacío, generando partículas de respaldo...")
        respaldo = Particula(EstadoConfortTermico.generar_estado_aleatorio(), weight=1.0)
        agente.set_belief(pomdp_py.Particles([respaldo]))

    particulas_seguras = [p for p in agente.cur_belief.particles if isinstance(p, Particula)]
    if not particulas_seguras:
        logging.warning("[WARNING] No hay partículas válidas; creando una de respaldo.")
        particulas_seguras = [Particula(EstadoConfortTermico.generar_estado_aleatorio(), weight=1.0)]

    total = sum(p.weight for p in particulas_seguras) or 1.0
    for p in particulas_seguras:
        p.weight = np.clip(p.weight / total, 0.01, 1.0)

    new_belief = pomdp_py.Particles(copy.deepcopy(particulas_seguras))
    agente.set_belief(new_belief)
    agente.cur_belief_states = convert_belief_to_state_list(new_belief)

    for idx, part in enumerate(agente.cur_belief.particles[:5]):
        logging.info(f"[DEBUG] Partícula {idx}: {part}")

    logging.debug(f"[DEBUG] ¿Planner tiene método 'plan'? {'plan' in dir(planificador)}")

    try:
        accion = planificador.plan(agente)
        if accion is None:
            raise ValueError("El planificador devolvió None.")
        logging.info(f"[DEBUG] Acción planificada con éxito: {accion}")
        return accion

    except Exception as e:
        logging.warning(f"[WARNING] Exception en `plan()`: {e}")
        try:
            belief_state = agente.cur_belief.mpe() if hasattr(agente.cur_belief, "mpe") else None
            default_action = agente.policy.rollout(belief_state)
            logging.info(f"[DEBUG] Fallback: política rollout → {default_action}")
            return default_action
        except Exception:
            logging.info("[DEBUG] Fallback final: acción neutra (sin cambio)")
            return AccionConfortTermico(cambio_de_Tt=0.0, cambio_de_flujo_de_aire=0.0)


def probar_planificador_con_arbol(problema_confort, planificador, pasos=5):
    logging.info(f"[DEBUG] Probando planificador {type(planificador).__name__} con árbol de búsqueda...")
    
    tiene_arbol = hasattr(planificador, 'tree') and planificador.tree is not None
    logging.info(f"[DEBUG] ¿Planificador tiene árbol?: {tiene_arbol}")

    if tiene_arbol:
        try:
            logging.debug("[DEBUG] ** Inspeccionando el árbol con TreeDebugger **")
            debugger = TreeDebugger(planificador.tree)
            debugger.display(width=80)
        except Exception as e:
            logging.error(f"[ERROR] Falló al inspeccionar el árbol con TreeDebugger: {e}")

    accion = safe_plan_call(planificador, problema_confort.agente)

    for i in tqdm(range(pasos), desc="Ejecutando simulación"):
        logging.info(f"\n[DEBUG] === Paso {i+1} ===")
        try:
            observacion_real = problema_confort.agente.observation_model.sample(problema_confort.entorno.estado_inicial, accion)
            update_belief_with_resample(problema_confort.agente, accion, observacion_real)
            planificador.update(problema_confort.agente, accion, observacion_real)
        except Exception as e:
            logging.error(f"[ERROR] Falló en la actualización durante la simulación: {e}")
        accion = safe_plan_call(planificador, problema_confort.agente)

    try:
        tree_str = planificador.get_tree()
        logging.info(f"[INFO] Árbol de búsqueda final:\n{tree_str}")
    except Exception as e:
        logging.error(f"[ERROR] Falló al obtener el árbol final: {e}")
    
    logging.info("[INFO] Simulación de planificación con árbol completada.")
    
def prueba_basica():

    logging.info("[INFO] Ejecutando prueba básica del sistema POMDP.")

    try:

        estado_inicial = EstadoConfortTermico(
            tdb=24.0,
            tr=25.3,
            rh=38.0,
            Tt=20.0,
            vr=0.75,
            met=1.05,
            clo=1.86,
            pmv=1.0
        )
        logging.debug("[DEBUG] Estado inicial creado correctamente: {}".format(estado_inicial))
    except Exception as e:
        logging.error("[ERROR] Falló la inicialización de EstadoConfortTermico: {}".format(e))
        return

    try:
        modelo_transicion = ModeloDeTransicion()
        logging.debug("[DEBUG] Modelo de Transición creado: {}".format(modelo_transicion))
    except Exception as e:
        logging.error("[ERROR] Falló la creación del ModeloDeTransicion: {}".format(e))
        return

    try:
        logging.info("[TEST] Probando `crear_creencia_inicial_en_particulas()`...")
        creencia = crear_creencia_inicial_en_particulas(num_particulas=1000)
        logging.info("[TEST] Creencia inicial creada con éxito: {}".format(creencia))

        pesos = [p.weight for p in creencia.particles]
        logging.debug("[DEBUG] Rango de pesos en la creencia inicial: min={}, max={}, promedio={}".format(
            min(pesos), max(pesos), sum(pesos) / len(pesos)
        ))

        if min(pesos) < 0.5:
            logging.warning("[WARNING] Algunos pesos son demasiado bajos. Ajustando...")
            for p in creencia.particles:
                p.weight = max(0.5, p.weight)

    except Exception as e:
        logging.error("[TEST ERROR] La prueba de creencia inicial falló: {}: {}".format(type(e).__name__, e))
        return

    if not isinstance(ModeloDePolitica(), pomdp_py.Policy):
        logging.error("[ERROR] 'ModeloDePolitica' no es una política válida.")
        return

    try:
        pouct = pomdp_py.POUCT(
            max_depth=5,
            discount_factor=0.95,
            num_sims=100,
            exploration_const=1.0,
            rollout_policy=ModeloDePolitica()
        )
        logging.debug("[DEBUG] Planificador POUCT configurado: {}".format(pouct))
    except Exception as e:
        logging.error("[ERROR] Falló la configuración de POUCT: {}".format(e))
        return

    try:
        estados_posibles = [estado_inicial]
        entorno = EntornoConfort(estados=estados_posibles, estado_inicial=estado_inicial)
        logging.debug("[DEBUG] Entorno creado correctamente: {}".format(entorno))
    except Exception as e:
        logging.error("[ERROR] Falló en la creación del entorno: {}".format(e))
        return

    if not isinstance(entorno, pomdp_py.Environment):
        logging.error("[ERROR] 'entorno' no es una instancia válida de pomdp_py.Environment.")
        return

    try:
        accion = pouct.plan(entorno)
        logging.debug("[DEBUG] Acción seleccionada por el planificador: {}".format(accion))
    except Exception as e:
        logging.error("[ERROR] Falló la planificación con POUCT: {}".format(e))
        return


def inspeccionar_arbol(planificador):
    arbol = planificador.get_tree()
    if arbol is None:
        print(" Árbol no generado (None).")
        return
    if not hasattr(arbol, "children") or not arbol.children:
        print(" Nodo raíz sin hijos: ningún rollout exitoso.")
        return

    print(f" Árbol generado: {len(arbol.children)} hijos en el nodo raíz.")
    for accion, nodo in arbol.children.items():
        visitas = getattr(nodo, "visits", "¿?")
        valor = getattr(nodo, "value", "¿?")
        print(f"├─ Acción: {accion} | Visitas: {visitas} | Valor estimado: {valor:.3f}")


def comparar_planificadores(agent, planners, num_episodios=5):
    resultados = {}

    for nombre, planificador in planners.items():
        logging.info(f"\n[COMPARACIÓN] Ejecutando planificador: {nombre}")
        agent.planificador = planificador
        resultados[nombre] = {'pmv': [], 'acciones': [], 'estados_iniciales': []}

        for paso in range(num_episodios):
            logging.info(f"[{nombre}] Paso {paso + 1}/{num_episodios}")

            estado_nuevo = EstadoConfortTermico.generar_estado_aleatorio()
            agent.set_state(estado_nuevo)
            resultados[nombre]['estados_iniciales'].append(estado_nuevo)

            particulas = [Particula(EstadoConfortTermico.generar_estado_aleatorio()) for _ in range(100)]
            agent.set_belief(CreenciaConfort(particulas))

            try:
                accion = agent.plan()
            except Exception:
                logging.exception(f"[ERROR] agent.plan() falló ({nombre})")
                break

            if accion is None or not getattr(accion, "es_accion_valida", lambda: False)():
                logging.warning("[WARNING] Acción inválida. Se usará fallback aleatorio.")
                accion = AccionConfortTermico()

            try:
                estado = agent.get_current_state()
                obs = agent.observation_model.sample(estado, accion)
                update_belief_with_resample(agent, accion, obs)

                pmv = getattr(estado, "pmv", None)
                resultados[nombre]['pmv'].append(pmv)
                resultados[nombre]['acciones'].append(accion)

                logging.info(f"[{nombre}] Acción: {accion}, PMV: {pmv}")
            except Exception:
                logging.exception("[ERROR] Falló actualización de creencia")
                break

    return resultados


import matplotlib.pyplot as plt

def graficar_resultados(resultados_acumulados):

    plt.figure(figsize=(10, 5))
    datos_graficados = False

    for nombre, pmvs_raw in resultados_acumulados.items():
        pmvs = [pmv for pmv in pmvs_raw if pmv is not None and np.isfinite(pmv)]
        if pmvs:
            plt.plot(pmvs, label=nombre)
            datos_graficados = True

    if not datos_graficados:
        logging.warning("[GRAFICO] No hay datos válidos para graficar.")
        return

    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=-0.5, color='gray', linestyle='--', linewidth=0.5)
    plt.title("Evolución del PMV por simulación")
    plt.xlabel("Paso")
    plt.ylabel("PMV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pmv_evolucion.png")
    plt.show()
    plt.close()
    logging.info("[GRAFICO] pmv_evolucion.png generado con éxito.")

def compute_action_from_obs(obs):
    belief = crear_belief_desde_sim(obs, num_particulas=300)
    agent = construir_agente_si_no_existe()
    agent.set_belief(belief)
    accion = agent.plan()
    return {'new_Tt': float(getattr(accion,'new_Tt', accion)), 'vr': float(getattr(accion,'vr', 0.0))}

_agent_singleton = None

def construir_agente_si_no_existe(num_particulas=300, planner_params=None):
    
    global _agent_singleton
    if _agent_singleton is not None:
        return _agent_singleton

    try:
        transition_model = ModeloDeTransicion()
        observation_model = ModeloDeObservacion()
        reward_model = ModeloDeRecompensa()

        init_belief = None

        try:
            from pomdp_py import POUCT, Agent
            planner = POUCT(model=None, planning_time=0.1)  # adapta parámetros a tu uso
            _agent_singleton = Agent(init_belief, transition_model, observation_model, reward_model, planner)
        except Exception:
            _agent_singleton = AgentePersonalizado(init_belief, transition_model, observation_model, reward_model)  # ajusta
        return _agent_singleton

    except Exception as e:
        class _FallbackAgent:
            def __init__(self, default_new_Tt=24.0, default_vr=0.5):
                self.default_new_Tt = float(default_new_Tt)
                self.default_vr = float(default_vr)
                self.belief = None
            def set_belief(self, belief):
                self.belief = belief
            def plan(self):
                class A:
                    pass
                a = A()
                a.new_Tt = self.default_new_Tt
                a.vr = self.default_vr
                return a

        _agent_singleton = _FallbackAgent()
        return _agent_singleton

import math
from collections import deque

def _safe_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def imprimir_arbol_pouct(root, max_depth=4, min_visits=0, indent="  "):
    def _node_repr(node):
        action = _safe_attr(node, ["action", "accion"])
        visits = _safe_attr(node, ["visits", "N"], 0)
        value = _safe_attr(node, ["value", "V"], None)
        av = f"visits={visits}"
        vv = f"value={value:.3f}" if isinstance(value, (int, float)) and not math.isnan(value) else "value=NA"
        act = repr(action) if action else "Root"
        return f"{act} | {av} | {vv}"

    def _walk(node, depth, prefix=""):
        if node is None:
            return
        if depth > max_depth:
            return
        visits = _safe_attr(node, ["visits", "N"], 0)
        if visits < min_visits:
            return
        print(prefix + _node_repr(node))
        children = _safe_attr(node, ["children", "hijos"], []) or []
        for i, c in enumerate(children):
            branch = "└─ " if i == len(children) - 1 else "├─ "
            _walk(c, depth + 1, prefix + indent + branch)

    _walk(root, 0, "")

def exportar_arbol_dot(root, path="arbol_pouct.dot"):

    try:
        from collections import deque
        import numpy as _np
    except Exception:
        from collections import deque
        _np = None

    def to_int_safe(x):
        try:
            return int(x)
        except Exception:
            if _np is not None:
                try:
                    return int(_np.asarray(x).item())
                except Exception:
                    return None
            return None

    try:
        lines = []
        lines.append("digraph POUCT {")
        lines.append("  node [shape=box, style=rounded];")

        q = deque([root])
        seen = set()

        # crear nodos y aristas
        while q:
            n = q.popleft()
            nid = id(n)
            if nid in seen:
                continue
            seen.add(nid)

            # obtener visits/value con seguridad
            raw_vis = getattr(n, "visits", None)
            raw_vis = getattr(n, "N", raw_vis)
            vis = to_int_safe(raw_vis)
            if vis is None:
                vis_label = "visits=?"
            else:
                vis_label = f"visits={vis}"

            val = getattr(n, "value", None)
            if val is None:
                val_label = ""
            else:
                try:
                    valf = float(val)
                    val_label = f"\\nvalue={valf:.3f}"
                except Exception:
                    val_label = f"\\nvalue={val}"

            # etiqueta: si root intenta diferenciar, mantenemos "Node" y id breve
            label = f"Node_{nid}\\n{vis_label}{val_label}"
            lines.append(f'  n{nid} [label="{label}"];')

            ch = getattr(n, "children", None) or getattr(n, "hijos", None)
            if isinstance(ch, dict):
                for key, child in ch.items():
                    # agregar arista y encolar child
                    lines.append(f"  n{nid} -> n{id(child)};")
                    q.append(child)
            else:
                for child in (ch or []):
                    lines.append(f"  n{nid} -> n{id(child)};")
                    q.append(child)

        lines.append("}")
        # escribir archivo
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logging.info("[INFO] DOT exportado a %s (nodos=%d)", path, len(seen))
    except Exception:
        logging.exception("[ERROR] exportar_arbol_dot falló")


def main():
    logging.info("[INFO] Iniciando simulaciones múltiples.")

    use_greedy = False

    num_simulaciones = 1

    resultados_acumulados = {"POUCT": [], "Greedy": []}

    for sim in range(num_simulaciones):
        logging.info(f"\n[INFO] === Simulación #{sim + 1} ===")
        pmv_objetivo = 0.0

        try:
            confort = ProblemaConfortTermico.crear(
                ruido_observacion=0.15,
                pmv_deseado=pmv_objetivo,
                use_greedy=use_greedy
            )
        except Exception:
            logging.exception("[ERROR] No se pudo crear ProblemaConfortTermico")
            continue

        agent = confort.agente

        policy = ModeloDePolitica(pmv_deseado=pmv_objetivo)
        policy.get_all_actions = lambda state=None, history=None: [
            AccionConfortTermico(Tt=round(t, 2), vr=round(v, 2))
            for t in np.linspace(16.0, 34.0, 10)
            for v in np.linspace(0.0, 2.0, 5)
        ]

        agent.policy = policy
        agent.rollout_policy = RolloutPolicySoftmax(
            acciones=policy.get_all_actions(),
            tau=0.7
        )

        puct = PlanificadorPOUCT_VF(
            max_depth=15,
            discount_factor=0.95,
            num_sims=1000,
            exploration_const=1.2,
            rollout_policy=agent.rollout_policy
        )

        planners = {"POUCT": puct}
        if use_greedy:
            greedy = PlanificadorGreedy(
                acciones=policy.get_all_actions(),
                modelo_de_transicion=agent.transition_model
            )
            planners["Greedy"] = greedy

        if not agent.cur_belief.particles:
            logging.warning("[WARNING] La creencia está vacía. Se reinicia con la inicial.")
            agent.set_belief(confort.belief_inicial)

        resultados = comparar_planificadores(agent, planners, num_episodios=200)

        # Diagnóstico corto: listar hijos inmediatos de la raíz (si existe)
        if 'puct' in locals() and puct is not None:
            root = getattr(puct, "root", getattr(puct, "tree", None))
            if root is None:
                logging.info("DIAG: root is None")
            else:
                children = getattr(root, "children", {}) or {}
                for i, (k, n) in enumerate(list(children.items())[:20]):
                    action_key = (k[0] if (isinstance(k, (list, tuple)) and len(k) > 0) else k)
                    logging.info(
                        "DIAG CHILD %02d: key_action=%r node_id=%s visits=%s N=%s value=%s",
                        i, action_key, id(n), getattr(n, "visits", None), getattr(n, "N", None), getattr(n, "value", None)
                    )
            logging.info("DIAG IDs: puct id=%s, puct.root id=%s, puct.tree id=%s",
                         id(puct), id(getattr(puct, "root", None)), id(getattr(puct, "tree", None)))
        else:
            logging.info("DIAG: puct no está definido en este contexto")

        # --- Unificar y usar la raíz real del POUCT (colocar justo después de comparar_planificadores) ---
        real_root = None
        # Preferir puct.tree si existe
        if getattr(puct, "tree", None) is not None:
            real_root = puct.tree
        # Si no, probar get_tree()
        elif hasattr(puct, "get_tree") and callable(getattr(puct, "get_tree")):
            try:
                cand = puct.get_tree()
                if cand is not None:
                    real_root = cand
            except Exception:
                real_root = None
        # Si sigue None, usar puct.root
        if real_root is None:
            real_root = getattr(puct, "root", None)

        # Forzar puct.root a la raíz encontrada (intento seguro)
        if real_root is not None:
            try:
                puct.root = real_root
            except Exception:
                pass
        else:
            logging.info("[DIAG] No se encontró root en puct (tree/get_tree/root)")

        # Usar real_root para inspección, normalización y exportación
        try:
            if not use_greedy and real_root is not None:
                logging.info("[INFO] Imprimiendo árbol POUCT (ASCII)")
                imprimir_arbol_pouct(real_root, max_depth=3, min_visits=5)

                # Normalizar visits en todo el árbol (antes de exportar)
                from collections import deque
                q = deque([real_root])
                seen = set()
                while q:
                    n = q.popleft()
                    nid = id(n)
                    if nid in seen:
                        continue
                    seen.add(nid)
                    try:
                        v = int(getattr(n, "visits", getattr(n, "N", getattr(n, "num_visits", 0))))
                        setattr(n, "visits", v)
                        setattr(n, "N", v)
                        try:
                            setattr(n, "num_visits", v)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    for c in (getattr(n, "children", {}) or {}).values():
                        q.append(c)

                # Guardar valor original por si hace falta restaurarlo luego
                original_root_vis = getattr(real_root, "visits", None)

                try:
                    children_vals = (getattr(real_root, "children", {}) or {}).values()
                    child_visits = []
                    for c in children_vals:
                        try:
                            child_visits.append(int(getattr(c, "visits", getattr(c, "N", getattr(c, "num_visits", 0)))))
                        except Exception:
                            pass
                    if child_visits:
                        # Estrategia escogida: suma de visits de hijos (refleja total de simulaciones)
                        root_vis = int(sum(child_visits))
                        setattr(real_root, "visits", root_vis)
                        try:
                            setattr(real_root, "N", root_vis)
                        except Exception:
                            pass
                        try:
                            setattr(real_root, "num_visits", root_vis)
                        except Exception:
                            pass
                        logging.info("[DIAG] root visits normalizado a %s (sum children)", root_vis)
                    else:
                        logging.info("[DIAG] child_visits vacío, no se cambió root.visits")
                except Exception:
                    logging.exception("[ERROR] al normalizar visits en root")

                logging.info("[INFO] Exportando árbol POUCT a arbol_pouct_sim%d.dot", sim+1)
                exportar_arbol_dot(real_root, path=f"arbol_pouct_sim{sim+1}.dot")

                # Si hace falta restaurar el valor original más tarde, descomenta:
                # if original_root_vis is not None:
                #     try:
                #         setattr(real_root, "visits", original_root_vis)
                #     except Exception:
                #         pass

            else:
                logging.info("[INFO] Árbol POUCT no disponible (use_greedy=True o root es None).")
        except Exception as e:
            logging.exception(f"[ERROR] Al inspeccionar/normalizar/exportar árbol POUCT: {e}")
        # --- fin unificación y exportación ---

        for nombre, datos in resultados.items():
            resultados_acumulados[nombre].append(datos)
            pmvs = [pmv for pmv in datos["pmv"]
                    if pmv is not None and np.isfinite(pmv)]
            media = np.mean(pmvs) if pmvs else float("nan")
            desviacion = np.std(pmvs) if pmvs else float("nan")
            en_confort = sum(abs(pmv) < 0.5 for pmv in pmvs)
            logging.info(
                f"[RESUMEN] {nombre}: PMV promedio={media:.2f}, "
                f"desviación={desviacion:.2f}, "
                f"en zona de confort={en_confort}/{len(pmvs)}"
            )
            if pmvs:
                plt.plot(range(len(pmvs)), pmvs, alpha=0.6,
                         label=f"{nombre} sim#{sim+1}")

    for nombre, simulaciones in resultados_acumulados.items():
        todos_pmvs = [
            pmv for datos in simulaciones
            for pmv in datos["pmv"]
            if pmv is not None and np.isfinite(pmv)
        ]
        if not todos_pmvs:
            logging.warning(f"[RESUMEN GLOBAL] {nombre}: no hay PMV válidos.")
            continue
        media = np.mean(todos_pmvs)
        desviacion = np.std(todos_pmvs)
        en_confort = sum(abs(pmv) < 0.5 for pmv in todos_pmvs)
        logging.info(
            f"[RESUMEN GLOBAL] {nombre}: PMV promedio={media:.2f}, "
            f"desviación={desviacion:.2f}, "
            f"en zona de confort={en_confort}/{len(todos_pmvs)}"
        )

    plt.title("Curvas de PMV por simulación")
    plt.xlabel("Paso de tiempo")
    plt.ylabel("PMV")
    plt.legend()
    plt.grid(True)
    plt.show()

    logging.info("[INFO] Simulaciones finalizadas con éxito.")

if __name__ == "__main__":
    logging.basicConfig(
        filename="registro_consola.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
        force=True
    )
    main()
