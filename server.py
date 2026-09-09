"""VISAGE Studio - serveur du frontend premium.

Lancement :
    python server.py --ckpt runs/ddpm/ckpt_last.pt [--cgan-ckpt runs/cgan/ckpt_last.pt]
    -> http://localhost:8000

Sans checkpoint (design/démo) :
    python server.py --demo

Architecture : FastAPI sert studio/index.html + une API de jobs asynchrones.
La génération tourne dans un thread ; le frontend interroge /api/job/<id>
pour afficher la progression du débruitage en temps réel.
"""
import argparse
import base64
import io
import os
import pathlib
import threading
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app import Engine, SKIN_CHOICES

HERE = pathlib.Path(__file__).parent
app = FastAPI(title="VISAGE Studio")
ENGINE: Engine = None
JOBS = {}          # id -> {status, done, total, images, captions, error}
JOBS_LOCK = threading.Lock()
MAX_JOBS = 40      # éviction des vieux jobs


def _png_b64(arr):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _new_job():
    with JOBS_LOCK:
        if len(JOBS) > MAX_JOBS:
            for k in list(JOBS)[: len(JOBS) - MAX_JOBS]:
                JOBS.pop(k, None)
        jid = uuid.uuid4().hex[:12]
        JOBS[jid] = {"status": "running", "done": 0, "total": 1,
                     "images": None, "captions": None, "error": None}
    return jid


def _run(jid, fn):
    def target():
        try:
            fn()
            JOBS[jid]["status"] = "done"
        except Exception as e:  # remonte l'erreur au frontend
            JOBS[jid]["status"] = "error"
            JOBS[jid]["error"] = str(e)
    threading.Thread(target=target, daemon=True).start()


def _cb(jid):
    def cb(done, total):
        JOBS[jid]["done"], JOBS[jid]["total"] = done, total
    return cb


class GenReq(BaseModel):
    age: float = 30
    gender: int = 0
    skin: int = 0
    n: int = 6
    guidance: float = 3.0
    steps: int = 40
    seed: int = -1


class MorphReq(BaseModel):
    age_a: float = 20
    gender_a: int = 0
    skin_a: int = 0
    age_b: float = 68
    gender_b: int = 0
    skin_b: int = 0
    frames: int = 8
    guidance: float = 3.0
    seed: int = 0
    gif: bool = False


class AtlasReq(BaseModel):
    ages: list = [25, 45, 65]
    guidance: float = 3.0
    steps: int = 30
    seed: int = 7


class CompareReq(BaseModel):
    age: float = 30
    gender: int = 0
    skin: int = 0
    n: int = 4
    guidance: float = 3.0
    seed: int = 42


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "studio" / "index.html").read_text(encoding="utf-8")


@app.get("/api/meta")
def meta():
    return {"demo": ENGINE.demo, "device": str(ENGINE.device),
            "image_size": ENGINE.image_size, "skins": SKIN_CHOICES,
            "cgan": ENGINE.cgan is not None}


@app.post("/api/generate")
def generate(req: GenReq):
    jid = _new_job()

    def work():
        imgs = ENGINE.generate(req.age, req.gender, req.skin, req.n,
                               req.guidance, req.steps, req.seed,
                               progress_cb=_cb(jid))
        JOBS[jid]["images"] = [_png_b64(a) for a in imgs]
    _run(jid, work)
    return {"job": jid}


@app.post("/api/morph")
def morph(req: MorphReq):
    jid = _new_job()

    def work():
        imgs = ENGINE.interpolate(req.age_a, req.gender_a, req.skin_a,
                                  req.age_b, req.gender_b, req.skin_b,
                                  req.frames, req.guidance, req.seed)
        JOBS[jid]["images"] = [_png_b64(a) for a in imgs]
        if req.gif:
            path, _ = ENGINE.make_gif(req.age_a, req.gender_a, req.skin_a,
                                      req.age_b, req.gender_b, req.skin_b,
                                      req.frames, req.guidance, req.seed)
            JOBS[jid]["gif"] = ("data:image/gif;base64," + base64.b64encode(
                pathlib.Path(path).read_bytes()).decode())
    _run(jid, work)
    return {"job": jid}


@app.post("/api/atlas")
def atlas(req: AtlasReq):
    jid = _new_job()

    def work():
        pairs = ENGINE.atlas(req.ages, req.guidance, req.steps, req.seed,
                             progress_cb=_cb(jid))
        JOBS[jid]["images"] = [_png_b64(a) for a, _ in pairs]
        JOBS[jid]["captions"] = [c for _, c in pairs]
    _run(jid, work)
    return {"job": jid}


@app.post("/api/compare")
def compare(req: CompareReq):
    jid = _new_job()

    def work():
        ddpm, cgan = ENGINE.compare(req.age, req.gender, req.skin, req.n,
                                    req.guidance, req.seed,
                                    progress_cb=_cb(jid))
        JOBS[jid]["images"] = [_png_b64(a) for a in ddpm]
        JOBS[jid]["images_b"] = [_png_b64(a) for a in cgan]
    _run(jid, work)
    return {"job": jid}


@app.get("/api/job/{jid}")
def job(jid: str):
    j = JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "job inconnu")
    return j


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("FACEGEN_CKPT"))
    p.add_argument("--cgan-ckpt", default=os.environ.get("FACEGEN_CGAN_CKPT"))
    p.add_argument("--demo", action="store_true")
    p.add_argument("--no-calibration", action="store_true",
                   help="désactive la calibration post-hoc de l'âge")
    p.add_argument("--best-of", type=int, default=1)
    p.add_argument("--clf", default="runs/classifier/attr_clf.pt")
    p.add_argument("--port", type=int, default=8000)
    # Requis pour charger le cGAN (son checkpoint n'embarque pas sa taille) ;
    # le DDPM se reconstruit depuis sa config. Préset mac -> 48.
    p.add_argument("--image-size", type=int, default=64)
    args = p.parse_args()
    if args.ckpt is None and not args.demo:
        for cand in ("ckpt_last.pt", "runs/ddpm/ckpt_last.pt"):
            if os.path.exists(cand):
                args.ckpt = cand
                break
    ENGINE = Engine(ckpt=args.ckpt, cgan_ckpt=args.cgan_ckpt,
                    image_size=args.image_size, demo=args.demo,
                    calibration=None if args.no_calibration
                    else "runs/ddpm/age_calibration.json",
                    best_of=args.best_of, clf=args.clf)
    print(f"VISAGE Studio -> http://localhost:{args.port} "
          f"({'DÉMO' if ENGINE.demo else 'modèle chargé'}, {ENGINE.device})")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
