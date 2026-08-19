#!/usr/bin/env python
"""data_v6 示例平衡图：探针接受样本中挑代表解，画 psi 等高线 + 分离面 +
雪点/接触点/线圈/壁标记，供用户确认物理正确性后全量生成。

与 probe_v6 同种子同判据（seed 123, i_seed=123*100000+k），重建平衡后按
接受判据过滤，画前 --n 个接受样本。用法（仓库根目录，需要 freegs_snow fork）:
    export PYTHONPATH="D:/D_F/Fusion/AI/PINN/freegs_snow"
    python dn_fno_2608/scripts/plot_v6_examples.py
    python dn_fno_2608/scripts/plot_v6_examples.py --config snow_double --n 2

输出：dn_fno_2608/data_v6/_probe/figs/<config>_ex<k>.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from gs_pino_dn_fno_2608 import generate_dn_dataset as g
from freegs import boundary, control, critical, jtor

CONFIGS = ["dn", "sn", "snow_single", "snow_double", "limiter"]
FLAGS = dict(
    xpt_jitter=0.06, xpt_jitter_z=0.10,
    isoflux_sampling=True, anchor_midplane=True,
    max_isoflux_residual=0.35, max_xpt_deviation=0.10,
    min_anchor_xpt_dist=0.15, coil_margin=0.05,
    min_core_depth=0.005, save_constraint_diag=True,
)


def sample_eq(config, k, check_limited=False, xpt_r0=None):
    """Replay probe sample k (seed 123); returns (eq, ok, info) — eq only if
    the same acceptance criteria as _solve_one pass (mirrors its logic).

    check_limited=True: enable freegs' in-solver wall check (limit_it=0) and
    reject any sample that becomes is_limited (core touches the wall) — the
    solver-native variant of the wall-contact criterion.
    xpt_r0: override the X-point centre R (physical-tuning scans)."""
    cfg = g.build_cfg(machine="mastu_simple", config=config, **FLAGS)
    if xpt_r0 is not None:
        cfg["xpt_r0"] = xpt_r0
    rng = np.random.default_rng(123)
    params = g.sample_params(rng, alpha=True, ranges=cfg["param_ranges"])
    for _ in range(k):
        params = g.sample_params(rng, alpha=True, ranges=cfg["param_ranges"])
    paxis, Ip, fvac = params["paxis"], params["Ip"], params["fvac"]
    am, an = params.get("alpha_m", 1.0), params.get("alpha_n", 2.0)
    rng2 = np.random.default_rng(123 * 100_000 + k)
    if cfg["kind"] == "snowflake":
        sf = g.sample_snowflake_points(rng2, r0=cfg["snow_r0"], z0=cfg["snow_z0"],
                                       n=cfg["xpt_n"], jitter_r=cfg["snow_jitter_r"],
                                       jitter_z=cfg["snow_jitter_z"])
        lo, up = sf[0], sf[-1]
    elif cfg["kind"] == "limiter":
        lo = up = None
    else:
        lo, up = g.sample_xpoints(rng2, jitter=cfg["xpt_jitter"],
                                  jitter_z=cfg["xpt_jitter_z"],
                                  r0=cfg["xpt_r0"], z0=cfg["xpt_z0"])
    anchor = g.sample_anchor(rng2, isoflux=cfg["isoflux_sampling"],
                             midplane=cfg["anchor_midplane"],
                             r_range=cfg["anchor_r_range"])

    tokamak = cfg["machine_factory"]()
    if cfg["kind"] == "snowflake":
        for name, val in g.MASTU_INIT_CURRENTS.items():
            tokamak[name].current = val
    eq = g.freegs.Equilibrium(tokamak=tokamak, Rmin=g.RMIN, Rmax=g.RMAX,
                              Zmin=g.ZMIN, Zmax=g.ZMAX,
                              nx=cfg["nx"], ny=cfg["ny"],
                              boundary=boundary.freeBoundaryHagenow)
    profiles = jtor.ConstrainPaxisIp(eq, paxis=paxis, Ip=Ip, fvac=fvac,
                                     alpha_m=am, alpha_n=an)

    info = {"cfg": cfg, "params": params, "anchor": np.asarray(anchor, np.float32),
            "x_coords": (np.asarray([*lo, *up] if cfg["xpt_n"] == 2
                                    else [*lo, 0.0, 0.0], np.float32)
                         if cfg["kind"] != "limiter" else None)}
    try:
        if cfg["kind"] == "snowflake":
            sf_pts = [lo] if cfg["xpt_n"] == 1 else [lo, up]
            c = control.constrain(snowflake=sf_pts,
                                  isoflux=[(*anchor, g.SNOW_REF_R, 0.0)] * len(sf_pts),
                                  gamma=g.SNOWFLAKE_GAMMA,
                                  snowflake_weight=cfg["weights"][0],
                                  snowflake_eps=g.SNOWFLAKE_EPS)
            ok = False
            for idx, w in enumerate(cfg["weights"]):
                try:
                    conv = g.freegs.solve(eq, profiles, c, rtol=g.SNOWFLAKE_RTOL,
                                          maxits=g.SNOWFLAKE_MAXITS, show=False,
                                          convergenceInfo=True,
                                          check_limited=check_limited, limit_it=0)
                    ok = True
                    break
                except RuntimeError:
                    if idx + 1 < len(cfg["weights"]):
                        c.snowflake_weight = cfg["weights"][idx + 1]
            if not ok:
                return None
        elif cfg["kind"] == "limiter":
            axis_tgt = (float(rng2.uniform(*cfg["axis_r_range"])), 0.0)
            info["axis_tgt"] = np.asarray(axis_tgt, np.float32)
            c = control.constrain(xpoints=[axis_tgt], gamma=g.LIMITER_GAMMA)
            g.freegs.solve(eq, profiles, c, check_limited=True, limit_it=0,
                           maxits=g.LIMITER_MAXITS, rtol=g.LIMITER_RTOL,
                           show=False, convergenceInfo=False)
            g.freegs.solve(eq, profiles, constrain=None, check_limited=True,
                           limit_it=0, maxits=g.LIMITER_MAXITS, rtol=g.LIMITER_RTOL,
                           show=False, convergenceInfo=False)
            if not getattr(eq, "is_limited", False):
                return None
            info["is_limited"] = True
            info["wall_contact"] = 1.0  # limiter: wall contact by construction
            info["Rlim"] = float(eq.Rlim)   # contact point set by check_limited
            info["Zlim"] = float(eq.Zlim)   # (limiter_R/Z = limiter geometry, None here)
        else:
            if cfg["xpt_n"] == 1:
                c = control.constrain(xpoints=[lo], isoflux=[(*lo, *anchor)], gamma=g.GAMMA)
            else:
                c = control.constrain(xpoints=[lo, up],
                                      isoflux=[(*lo, *anchor), (*up, *anchor)],
                                      gamma=g.GAMMA)
            solve_maxits = g.SNOWFLAKE_MAXITS if cfg["machine"] == "mastu_simple" else g.MAXITS
            conv = g.freegs.solve(eq, profiles, c, rtol=g.RTOL, maxits=solve_maxits,
                                  show=False, convergenceInfo=True,
                                  check_limited=check_limited, limit_it=0)
            info["n_iter"] = len(conv[1])
    except Exception:
        return None

    # --- acceptance: the generator's shared checks, verbatim ---
    # (g._acceptance_checks: separatrix count, axis selection, snowflake
    # residuals, data_v4 geometry checks, data_v6 wall-contact gate — probe
    # and figure replay can never drift from generation)
    if check_limited and getattr(eq, "is_limited", False):
        return None  # solver-native wall-contact rejection
    opt, xpt = critical.find_critical(eq.R, eq.Z, eq.psi())
    sep = [p for p in xpt if float(p[2]) >= eq.psi_bndry - 1e-6]
    if cfg["kind"] != "limiter":
        acc = g._acceptance_checks(cfg, eq, profiles, tokamak, lo, up, anchor,
                                   opt, xpt, sep, params["Ip"])
        if acc is None:
            return None
        _, psi_bndry, wall_contact, wall_contact_excess, inwall_frac, _, _, _ = acc
        info["psi_bndry"] = psi_bndry
        info["wall_contact"] = wall_contact
        info["wall_contact_excess"] = wall_contact_excess
        info["inwall_sep_frac"] = inwall_frac
        if cfg["kind"] == "snowflake":
            info["xpts_actual"] = np.asarray(sep, np.float64)
    else:
        info["psi_bndry"] = eq.psi_bndry
        info["wall_contact"] = 1.0  # limiter: wall contact by construction
        info["wall_contact_excess"] = 0.0
    info["sep"] = sep
    info["opt"] = opt
    return eq, info


def _plot(ax, eq, info, config, title):
    psi = eq.psi()
    # full-domain psi contours — the psi field exists over the whole solve
    # domain (vacuum, legs, background field); cover it from the domain min
    # to the magnetic axis
    psi_lvls = np.linspace(float(psi.min()), float(eq.psi_axis), 15)
    ax.contour(eq.R, eq.Z, psi, levels=psi_lvls, colors="k",
               linewidths=0.4, alpha=0.55)
    # separatrix: one smooth psi=psi_bndry contour over the whole domain.
    # Wall-internal segments are NOT clipped: the in-wall structure gate
    # (INWALL_SEP_FRAC_TOL) has already rejected samples with significant
    # solenoid-overlap structures at generation, so everything drawn is the
    # real solution (snowflake points keep small inner-column crossings by
    # design — accepted physics)
    ax.contour(eq.R, eq.Z, psi, levels=[eq.psi_bndry], colors="r",
               linewidths=1.8)
    for o in info.get("opt", []):
        ax.plot(o[0], o[1], "o", ms=4, color="tab:blue", zorder=6)
    for p in info.get("sep", []):
        ax.plot(p[0], p[1], "x", ms=6, mew=1.5, color="tab:orange", zorder=6)
    for label, coil in eq.tokamak.coils:
        if not coil.control:
            continue
        if hasattr(coil, "coils"):
            for _, sub, _ in coil.coils:
                ax.plot(sub.R, sub.Z, "s", ms=2.5, color="gray", alpha=0.85)
        elif hasattr(coil, "Rs"):  # Solenoid: stack endpoints
            ax.plot([coil.Rs, coil.Rs], [coil.Zsmin, coil.Zsmax], "|",
                    ms=3, color="gray", alpha=0.85)
        else:
            ax.plot(coil.R, coil.Z, "s", ms=2.5, color="gray", alpha=0.85)
    if eq.tokamak.wall is not None:
        # wall point list is open ((1.6,1) -> (1.6,-1)); close it explicitly
        Rw = np.append(eq.tokamak.wall.R, eq.tokamak.wall.R[0])
        Zw = np.append(eq.tokamak.wall.Z, eq.tokamak.wall.Z[0])
        ax.plot(Rw, Zw, "k-", lw=1.2, alpha=0.85)
    if config == "limiter":
        ax.plot(info["Rlim"], info["Zlim"], "g*", ms=16, zorder=8,
                label=f"limit ({info['Rlim']:.3f},{info['Zlim']:.3f})")
    else:
        xc = info["x_coords"]
        for r, z in [(xc[0], xc[1]), (xc[2], xc[3])]:
            if r == 0.0 and z == 0.0:
                continue
            ax.plot(r, z, "g*", ms=13, zorder=8,
                    label="snow pt" if config.startswith("snow") else "X-pt")
        ar, az = info["anchor"]
        if ar != 0.0 or az != 0.0:
            ax.plot(ar, az, "mv", ms=8, zorder=8, label="anchor")
    if "xpts_actual" in info:
        a = np.asarray(info["xpts_actual"])
        ax.plot(a[:, 0], a[:, 1], "c+", ms=10, mew=1.6, zorder=7, label="actual X-pt")
    ax.set_xlim(g.RMIN, g.RMAX)
    ax.set_ylim(g.ZMIN, g.ZMAX)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="all", choices=CONFIGS + ["all"])
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    out_dir = Path("dn_fno_2608/data_v6/_probe/figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = CONFIGS if args.config == "all" else [args.config]
    for config in configs:
        drawn = 0
        for k in range(200):
            got = sample_eq(config, k)
            if got is None:
                continue
            eq, info = got
            p = info["params"]
            title = (f"MASTU_simple {config} | Ip={p['Ip']:.2e} paxis={p['paxis']:.0f} "
                     f"fvac={p['fvac']:.2f} am={p['alpha_m']:.2f} an={p['alpha_n']:.2f}")
            if config == "limiter":
                title += f" | Rlim={info['Rlim']:.3f} Zlim={info['Zlim']:.3f}"
            elif config.startswith("snow"):
                res = info.get("snowflake_res")
                if res is None:  # recompute quickly for the title
                    res = []
                    lo_ = info["x_coords"][:2]
                    res = [g._at(eq, lo_, "Br"), g._at(eq, lo_, "Bz"),
                           float(np.asarray(eq.hessianZZ(*lo_, eps=g.SNOWFLAKE_EPS)).reshape(-1)[0]),
                           float(np.asarray(eq.hessianRZ(*lo_, eps=g.SNOWFLAKE_EPS)).reshape(-1)[0])]
                title += f" | Br={res[0]:.1e} Bz={res[1]:.1e} ZZ={res[2]:.1e} RZ={res[3]:.1e}"
            if config != "limiter":
                title += (f" | wall-contact={info['wall_contact_excess']:.1%}"
                          if info["wall_contact"] else " | no wall-contact")
            fig, ax = plt.subplots(figsize=(7, 8))
            _plot(ax, eq, info, config, title)
            ax.legend(loc="upper right", fontsize=7)
            fname = out_dir / f"{config}_ex{drawn + 1}.png"
            fig.savefig(fname, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"-> {fname}")
            drawn += 1
            if drawn >= args.n:
                break
        if drawn == 0:
            print(f"{config}: no accepted sample found in k<200")


if __name__ == "__main__":
    main()
