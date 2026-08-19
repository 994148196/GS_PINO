import json

for b in ("all", "dn", "sn"):
    m = json.load(open(
        f"dn_fno_2608/kan/experiments/exp001_kan_v5/model_b19ch_kan_mix/eval_{b}/metrics.json"))
    g, r = m["geometry"], m["gs_residual"]
    xo = g["o_point_cm"]["mean"]
    xl = g["x_lo_cm"]["mean"]
    xu = g["x_up_cm"]["mean"]
    print(f"{b:3s} relL2 {m['rel_l2_pct']['mean']:6.2f}% | "
          f"RMSE {m['rmse_phys_Wb']['mean']:.2e} | R2 {m['r_squared']['mean']:.4f} | "
          f"Jself {m['j_self_check_pct']['mean']:5.1f}% | GSratio {r['ratio_pred_true']:.3f} | "
          f"Xpt {xl:.1f}/{xu:.1f}cm | Opt {xo:.1f}cm | sep {g['sep_mean_cm']['mean']:.1f}cm | "
          f"fail {m['n_find_critical_fail']}")
