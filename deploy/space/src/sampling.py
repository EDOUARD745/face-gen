"""Améliorations du contrôle d'attributs À L'INFÉRENCE (sans ré-entraînement).

Deux mécanismes complémentaires, tous deux appliqués après coup sur un modèle
déjà entraîné :

1. CALIBRATION DE LA CONDITION (`AgeCalibration`)
   La réponse en âge du modèle est monotone mais compressée vers le centre du
   support d'entraînement (cf. `src.age_response`). On inverse cette fonction de
   réponse : pour obtenir un visage de 30 ans on injecte la valeur de condition
   qui, empiriquement, produit 30 ans. Coût nul à l'échantillonnage.

2. ÉCHANTILLONNAGE PAR REJET (`sample_best_of`)
   On génère k candidats pour la même condition et on conserve celui que le
   classifieur d'attributs juge le plus conforme (rejection sampling guidé par
   discriminateur, cf. Azadi et al., 2019). Coût : k fois plus de passages
   d'échantillonnage ; la diversité intra-condition baisse légèrement puisqu'on
   sélectionne dans la population générée.
"""
import json
import pathlib

import numpy as np
import torch

from .data import denormalize_age, normalize_age


class AgeCalibration:
    """Table monotone « âge voulu -> valeur de condition à injecter »."""

    def __init__(self, cible_annees, condition_annees, min_age=18, max_age=70):
        self.cible = np.asarray(cible_annees, dtype=np.float64)
        self.condition = np.asarray(condition_annees, dtype=np.float64)
        self.min_age, self.max_age = min_age, max_age

    @classmethod
    def load(cls, path):
        d = json.loads(pathlib.Path(path).read_text())
        if not d.get("applicable", True):
            raise ValueError(
                f"calibration non applicable (amplitude de réponse "
                f"{d.get('amplitude_reponse_annees', float('nan')):.1f} ans) : "
                f"le modèle ne suit pas la condition d'âge")
        return cls(d["cible_annees"], d["condition_annees"])

    def years(self, age_years):
        """Âge voulu (années, array-like) -> condition (années)."""
        return np.interp(np.asarray(age_years, dtype=np.float64),
                         self.cible, self.condition)

    def apply(self, attrs):
        """Renvoie une copie de `attrs` dont l'âge normalisé est calibré."""
        a = attrs["age"]
        years = denormalize_age(a.detach().cpu().numpy())
        cond = self.years(years)
        norm = (np.clip(cond, self.min_age, self.max_age) - self.min_age) \
            / (self.max_age - self.min_age)
        out = dict(attrs)
        out["age"] = torch.tensor(norm, dtype=a.dtype, device=a.device)
        return out


def load_calibration(path):
    """Charge une calibration si le fichier existe, sinon None."""
    p = pathlib.Path(path) if path else None
    if p is not None and p.exists():
        try:
            return AgeCalibration.load(p)
        except ValueError as e:
            print(f"Calibration ignorée : {e}")
    return None


@torch.no_grad()
def conformity_score(x, attrs, clf, w_age=1.0, w_gender=1.0, w_skin=1.0):
    """Score de NON-conformité (plus bas = plus conforme à la condition).

    Âge : erreur absolue ramenée à l'amplitude 18-70 ; genre et peau :
    probabilité manquante sur la classe demandée.
    """
    age_p, gender_p, skin_p = clf(x)
    age_err = (denormalize_age(age_p) - denormalize_age(attrs["age"])).abs() / 52.0
    pg = gender_p.softmax(1).gather(1, attrs["gender"][:, None]).squeeze(1)
    ps = skin_p.softmax(1).gather(1, attrs["skin"][:, None]).squeeze(1)
    return w_age * age_err + w_gender * (1 - pg) + w_skin * (1 - ps)


@torch.no_grad()
def sample_best_of(gen_fn, attrs, n, clf, k=4, **score_kw):
    """Génère k lots de candidats et garde, par position, le plus conforme."""
    best_x, best_s = None, None
    for _ in range(max(1, k)):
        x = gen_fn(attrs, n)
        if clf is None or k <= 1:
            return x
        s = conformity_score(x, attrs, clf, **score_kw)
        if best_x is None:
            best_x, best_s = x, s
        else:
            m = s < best_s
            best_x[m], best_s[m] = x[m], s[m]
    return best_x


# Bornes colorimétriques mesurées sur FairFace 48 px (1024 images de
# validation, 99e centile) : au-delà, un tirage sort de la plage des images
# réelles. Environ 2-3 % des tirages du modèle v2 franchissent ce seuil et
# apparaissent comme des visages verts ou sursaturés.
# Statistiques colorimétriques des visages réels (FairFace 48 px, 4096 images
# de validation) : moyenne et écart-type de la moyenne de chaque canal.
# Un tirage est écarté si l'un de ses canaux s'éloigne de plus de 3,21
# écarts-types, seuil correspondant au 99,5e centile des images réelles.
#
# Ce critère vise les tirages franchement cassés (dominantes bleues, cyan,
# magenta) et NON le biais systématique du modèle, qui éclaircit la peau de
# +0,053 en luminance et comprime de 32 % l'écart entre groupes (voir rapport,
# section 8.1). Un filtre ne corrige pas un décalage de distribution : il ne
# fait qu'écarter les valeurs extrêmes.
REAL_MEAN = (0.4806, 0.3556, 0.3025)
REAL_STD = (0.1398, 0.1219, 0.1241)
REAL_Z_P995 = 3.21


@torch.no_grad()
def colour_outliers(x, z_max=REAL_Z_P995):
    """Masque (B,) des tirages hors de la plage colorimétrique du réel."""
    ch = ((x.clamp(-1, 1) + 1) / 2).mean((2, 3))
    mu = torch.tensor(REAL_MEAN, device=x.device, dtype=ch.dtype)
    sd = torch.tensor(REAL_STD, device=x.device, dtype=ch.dtype)
    return ((ch - mu).abs() / sd).max(1).values > z_max


@torch.no_grad()
def resample_artifacts(gen_fn, attrs, n, x=None, marge=0.4, max_rounds=2):
    """Écarte les tirages aberrants en SUR-GÉNÉRANT une fois, puis en
    sélectionnant les plus proches de la distribution réelle.

    La régénération en boucle (tirer, filtrer, retirer les fautifs, répéter)
    donne un coût imprévisible : avec 28 % de rejet et cinq passes, la latence
    peut tripler. Ici le surcoût est fixe et connu d'avance : un lot unique de
    n(1 + marge) échantillons, dont on garde les n meilleurs. Une seconde passe
    n'a lieu que s'il reste des aberrants parmi les retenus.
    """
    supp = max(1, int(round(n * marge)))

    def score(y):
        ch = ((y.clamp(-1, 1) + 1) / 2).mean((2, 3))
        mu = torch.tensor(REAL_MEAN, device=y.device, dtype=ch.dtype)
        sd = torch.tensor(REAL_STD, device=y.device, dtype=ch.dtype)
        return ((ch - mu).abs() / sd).max(1).values

    if x is None:
        x = gen_fn(attrs, n)
    for _ in range(max_rounds):
        s_x = score(x)
        if (s_x <= REAL_Z_P995).all():
            break
        attrs_supp = {k: v[:supp] for k, v in attrs.items()}
        y = gen_fn(attrs_supp, supp)
        # on remplace les pires tirages par les meilleurs candidats
        pires = s_x.argsort(descending=True)[:supp]
        s_y = score(y)
        for rang, i in enumerate(pires.tolist()):
            if s_y[rang] < s_x[i]:
                x[i] = y[rang]
    return x


def make_controlled_sampler(gen_fn, clf=None, calibration=None, best_of=1,
                            **score_kw):
    """Enveloppe un générateur brut avec calibration + rejet.

    `gen_fn(attrs, n)` -> images [-1,1]. Renvoie une fonction de même signature.
    """
    def sampler(attrs, n):
        a = calibration.apply(attrs) if calibration is not None else attrs
        if best_of > 1 and clf is not None:
            return sample_best_of(gen_fn, a, n, clf, k=best_of, **score_kw)
        return gen_fn(a, n)
    return sampler
