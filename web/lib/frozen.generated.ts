/**
 * GENERATED FILE - DO NOT EDIT.
 *
 *   python scripts/phase6_build_frontend_data.py
 *
 * Every figure below is read from a committed artefact in outputs/ or from the
 * predictions mirror in live_log/, and carries the artefact's filename in its
 * `source` field so a number on a web page can be traced the way a number in
 * REPORT.md can. Nothing here is computed except DECOMPOSITION, which is
 * arithmetic over three named artefacts and is flagged `derived: true`.
 *
 * 2026-27 results and closing odds are deliberately ABSENT: L4.8 keeps them
 * out of the repository until May 2027, so the frontend reads those from
 * Supabase at request time instead.
 *
 * Built 2026-09-23T10:10:00Z from 14 artefacts.
 */

export const META = {
  "devMatches": 1520,
  "sourceMatches": 1900,
  "seasons": "2021-22 to 2025-26",
  "holdoutSeason": "2026-2027",
  "cutoffDate": "2026-09-04",
  "cutoffFixture": "Ipswich Town v Liverpool",
  "expectedHoldoutMatches": 360,
  "bootstrapDraws": 10000,
  "bootstrapSeed": 20260901,
  "primaryMetric": "RPS with log loss, under the sign-agreement rule",
  "book": "B365C",
  "devig": "proportional",
  "cadenceCostLogLoss": 0.0009134,
  "cadenceCostRps": 0.0002268,
  "logRefitsPerSeason": 38,
  "frozenRefitsRange": "109 to 120",
  "freezeSha": "b36befd3e13d5f7f5d9b0af83b7db819e86d7b13e6fdcab4c55db0599fd248f6",
  "pinSha": "be9b1b084db57a3bef2e71f83a04c9d89299cbe1599b936f0615a69b1c266e15",
  "protocolSha": "c419e51bc0e9165f7e79277a75345c33fef21b6decfc086976ac30e23d75e36c",
  "freezeShaMatchesMirror": true,
  "noHistoryShareOfHoldout": 0.195,
  "generatedAt": "2026-09-23T10:10:00Z"
} as const;

export const LADDER = [
  {
    "key": "D0",
    "label": "D0 - base rate",
    "kind": "baseline",
    "n": 1520,
    "logLoss": 1.0688875039038455,
    "rps": 0.2318511825754078,
    "accuracy": 0.44473684210526315,
    "balancedAccuracy": 0.3333333333333333,
    "macroF1": 0.20522161505768063,
    "brier": 0.6466651036857494,
    "note": "Season base rates only. Best calibration in the project and the worst model in it.",
    "source": "phase4_ladder_pooled.csv"
  },
  {
    "key": "D1",
    "label": "D1 - results-derived features",
    "kind": "ladder",
    "n": 1520,
    "logLoss": 1.0039286603133988,
    "rps": 0.20768661472157898,
    "accuracy": 0.5230263157894737,
    "balancedAccuracy": 0.4393986284073184,
    "macroF1": 0.38051776773720847,
    "brier": 0.5994097897517485,
    "note": "Current-season results. 83% of the whole distance to Dixon-Coles.",
    "source": "phase4_ladder_pooled.csv"
  },
  {
    "key": "D2",
    "label": "D2 - + dynamic state",
    "kind": "ladder",
    "n": 1520,
    "logLoss": 1.0008629027959084,
    "rps": 0.2066712060020117,
    "accuracy": 0.525,
    "balancedAccuracy": 0.4414906786165235,
    "macroF1": 0.38230435619626874,
    "brier": 0.5973582999644979,
    "note": null,
    "source": "phase4_ladder_pooled.csv"
  },
  {
    "key": "D2_rescaled",
    "label": "D2 rescaled - Amendment 4",
    "kind": "ladder",
    "n": 1520,
    "logLoss": 1.0002829219829275,
    "rps": 0.2064575379858117,
    "accuracy": 0.525,
    "balancedAccuracy": 0.4414906786165235,
    "macroF1": 0.38239286406825673,
    "brier": 0.5969538278400535,
    "note": null,
    "source": "phase4_d34_pooled.csv"
  },
  {
    "key": "D3",
    "label": "D3 - + context, Block C",
    "kind": "ladder",
    "n": 1520,
    "logLoss": 1.0012496062596037,
    "rps": 0.20661549225436102,
    "accuracy": 0.5210526315789473,
    "balancedAccuracy": 0.43791933846649006,
    "macroF1": 0.3791786521053744,
    "brier": 0.597560200576238,
    "note": "Not significant against D2 rescaled. The gate stands failed.",
    "source": "phase4_d34_pooled.csv"
  },
  {
    "key": "D4",
    "label": "D4 - + prior-season FBref",
    "kind": "ladder",
    "n": 1520,
    "logLoss": 0.9997494393965212,
    "rps": 0.20618831291596168,
    "accuracy": 0.5269736842105263,
    "balancedAccuracy": 0.44256146171176747,
    "macroF1": 0.38332530599637993,
    "brier": 0.596270940776593,
    "note": "139 columns, and Elo v1's single rating still has a lower log loss.",
    "source": "phase4_d34_pooled.csv"
  },
  {
    "key": "elo_v1",
    "label": "Elo v1 - one K=20 rating",
    "kind": "rating",
    "n": 1520,
    "logLoss": 0.9994313008246677,
    "rps": 0.20667039138488577,
    "accuracy": 0.5355263157894737,
    "balancedAccuracy": 0.44713343731689403,
    "macroF1": 0.3874139626352016,
    "brier": 0.595798470488079,
    "note": "Flat 1500 start, 60-point home advantage. Beats every engineered rung.",
    "source": "phase4_ladder_pooled.csv"
  },
  {
    "key": "poisson_walkforward",
    "label": "Poisson walk-forward",
    "kind": "rating",
    "n": 1520,
    "logLoss": 0.9904153939794983,
    "rps": 0.20354687300695612,
    "accuracy": 0.5269736842105263,
    "balancedAccuracy": 0.4513443588918323,
    "macroF1": 0.3907064937497074,
    "brier": 0.5899830466303654,
    "note": null,
    "source": "phase4_ladder_pooled.csv"
  },
  {
    "key": "dc_walkforward",
    "label": "Dixon-Coles walk-forward",
    "kind": "frozen",
    "n": 1520,
    "logLoss": 0.9903635065353805,
    "rps": 0.20349883260772664,
    "accuracy": 0.5263157894736842,
    "balancedAccuracy": 0.45106465898754067,
    "macroF1": 0.3917350023174469,
    "brier": 0.5896948042349883,
    "note": "The frozen model. Refits once per distinct scored date.",
    "source": "phase4_ladder_pooled.csv + phase2_poisson_dc_fold_summary.csv"
  },
  {
    "key": "E1a_sot",
    "label": "E1a - shots-on-target ratings",
    "kind": "arm",
    "n": 1520,
    "logLoss": 0.9812432263507834,
    "rps": 0.20116868565377008,
    "accuracy": 0.5276315789473685,
    "balancedAccuracy": 0.4514289486931908,
    "macroF1": 0.39105155832151456,
    "brier": 0.5833629886920167,
    "note": "Lower point estimate than Dixon-Coles, and the difference is not significant.",
    "source": "phase5_e1a_pooled.csv"
  },
  {
    "key": "E1b",
    "label": "E1b - shot residual",
    "kind": "arm",
    "n": 1520,
    "logLoss": 0.9975796392814814,
    "rps": 0.2056791560173178,
    "accuracy": 0.5276315789473685,
    "balancedAccuracy": 0.4438715720498791,
    "macroF1": 0.3846857798034769,
    "brier": 0.5950200697288268,
    "note": null,
    "source": "phase5_e1b_pooled.csv"
  },
  {
    "key": "E1c",
    "label": "E1c - finishing residual",
    "kind": "arm",
    "n": 1520,
    "logLoss": 0.9995609688490407,
    "rps": 0.2063655021033932,
    "accuracy": 0.5256578947368421,
    "balancedAccuracy": 0.44259653553184286,
    "macroF1": 0.3833485161778721,
    "brier": 0.5964406305031676,
    "note": "Fired the sign-agreement rule against Elo v1: reported INCONCLUSIVE.",
    "source": "phase5_e1c_pooled.csv"
  },
  {
    "key": "market_B365C_proportional",
    "label": "Market - Bet365 closing",
    "kind": "market",
    "n": 1520,
    "logLoss": 0.9605733901135316,
    "rps": 0.19469180196268818,
    "accuracy": 0.5513157894736842,
    "balancedAccuracy": 0.4724484827478069,
    "macroF1": 0.4094796472397452,
    "brier": 0.5701943368645321,
    "note": "Chosen on completeness before any score existed. Never fitted to anything.",
    "source": "phase5_market_pooled.csv"
  }
] as const;

export const DELTAS = [
  {
    "comparison": "D1 - D0",
    "label": "Current-season results pay",
    "left": "D1",
    "right": "D0",
    "n": 1520,
    "logLossDelta": -0.06495884359044668,
    "logLossCi": [
      -0.08121046936568393,
      -0.04938187889002471
    ],
    "rpsDelta": -0.024164567853828832,
    "rpsCi": [
      -0.02972398006817006,
      -0.018833351406700515
    ],
    "signsAgree": true,
    "excludesZero": true,
    "verdict": "SIGNIFICANT",
    "source": "phase4_ladder_deltas.csv"
  },
  {
    "comparison": "D2 - D1",
    "label": "Dynamic state pays, barely",
    "left": "D2",
    "right": "D1",
    "n": 1520,
    "logLossDelta": -0.003065757517490198,
    "logLossCi": [
      -0.004072129989201748,
      -0.002076148092555847
    ],
    "rpsDelta": -0.001015408719567277,
    "rpsCi": [
      -0.001354086828582584,
      -0.0006848185919822827
    ],
    "signsAgree": true,
    "excludesZero": true,
    "verdict": "SIGNIFICANT",
    "source": "phase4_ladder_deltas.csv"
  },
  {
    "comparison": "D2rescaled - D1",
    "label": "Amendment 4's rescaled rung",
    "left": "D2_rescaled",
    "right": "D1",
    "n": 1520,
    "logLossDelta": -0.003645738330471352,
    "logLossCi": [
      -0.004892061889897334,
      -0.0024072931640715983
    ],
    "rpsDelta": -0.0012290767357673001,
    "rpsCi": [
      -0.0016445835050625914,
      -0.0008193160002510405
    ],
    "signsAgree": true,
    "excludesZero": true,
    "verdict": "SIGNIFICANT",
    "source": "phase4_a4_deltas.csv"
  },
  {
    "comparison": "D3 - D2rescaled",
    "label": "Context block: no",
    "left": "D3",
    "right": "D2_rescaled",
    "n": 1520,
    "logLossDelta": 0.0009666842766762095,
    "logLossCi": [
      -0.0008220612626872546,
      0.0027660429011830624
    ],
    "rpsDelta": 0.00015795426854931874,
    "rpsCi": [
      -0.0002718887054890174,
      0.0005839124106662872
    ],
    "signsAgree": true,
    "excludesZero": false,
    "verdict": "NOT SIGNIFICANT",
    "source": "phase4_d34_deltas.csv"
  },
  {
    "comparison": "D4 - D3",
    "label": "Prior-season FBref: no",
    "left": "D4",
    "right": "D3",
    "n": 1520,
    "logLossDelta": -0.0015001668630824872,
    "logLossCi": [
      -0.004483347810620705,
      0.0014817745901269358
    ],
    "rpsDelta": -0.00042717933839930963,
    "rpsCi": [
      -0.001229047113823633,
      0.0003711906185119568
    ],
    "signsAgree": true,
    "excludesZero": false,
    "verdict": "NOT SIGNIFICANT",
    "source": "phase4_d34_deltas.csv"
  },
  {
    "comparison": "D4 - D2rescaled",
    "label": "The two blocks together: no",
    "left": "D4",
    "right": "D2_rescaled",
    "n": 1520,
    "logLossDelta": -0.0005334825864062774,
    "logLossCi": [
      -0.0036297898238767234,
      0.0025881031456705837
    ],
    "rpsDelta": -0.00026922506984999075,
    "rpsCi": [
      -0.001097768179753985,
      0.0005699910681665871
    ],
    "signsAgree": true,
    "excludesZero": false,
    "verdict": "NOT SIGNIFICANT",
    "source": "phase4_d34_deltas.csv"
  },
  {
    "comparison": "D2rescaled - DixonColes",
    "label": "The unmeasured 13% residual",
    "left": "D2_rescaled",
    "right": "dc_walkforward",
    "n": 1520,
    "logLossDelta": 0.00991941544754698,
    "logLossCi": [
      -0.003299278668456529,
      0.023137330079320874
    ],
    "rpsDelta": 0.0029587053780850813,
    "rpsCi": [
      -0.0006760638379784491,
      0.006622557719018391
    ],
    "signsAgree": true,
    "excludesZero": false,
    "verdict": "NOT SIGNIFICANT",
    "source": "phase4_a4_deltas.csv"
  },
  {
    "comparison": "E1a - DixonColes",
    "label": "Why the freeze took Dixon-Coles, not E1a",
    "left": "E1a_sot",
    "right": "dc_walkforward",
    "n": 1520,
    "logLossDelta": -0.00912028018459704,
    "logLossCi": [
      -0.021774217028028037,
      0.003912245149213716
    ],
    "rpsDelta": -0.0023301469539565326,
    "rpsCi": [
      -0.00613410644934303,
      0.001632253526969914
    ],
    "signsAgree": true,
    "excludesZero": false,
    "verdict": "NOT SIGNIFICANT",
    "source": "phase5_e1a_deltas.csv"
  },
  {
    "comparison": "E1b - D2rescaled",
    "label": "Shot residual: no",
    "left": "E1b",
    "right": "D2_rescaled",
    "n": 1520,
    "logLossDelta": -0.0027032827014460917,
    "logLossCi": [
      -0.005362049954566067,
      1.150517875485694e-05
    ],
    "rpsDelta": -0.0007783819684939095,
    "rpsCi": [
      -0.0015344373660217123,
      -1.7233929631151697e-05
    ],
    "signsAgree": true,
    "excludesZero": false,
    "verdict": "NOT SIGNIFICANT",
    "source": "phase5_e1b_deltas.csv"
  },
  {
    "comparison": "E1c - D2rescaled",
    "label": "Finishing residual: no",
    "left": "E1c",
    "right": "D2_rescaled",
    "n": 1520,
    "logLossDelta": -0.0007219531338868048,
    "logLossCi": [
      -0.0022861039663708825,
      0.0008366751072873048
    ],
    "rpsDelta": -9.203588241853017e-05,
    "rpsCi": [
      -0.0005164400644407002,
      0.0003319669855255899
    ],
    "signsAgree": true,
    "excludesZero": false,
    "verdict": "NOT SIGNIFICANT",
    "source": "phase5_e1c_deltas.csv"
  },
  {
    "comparison": "market - DixonColes",
    "label": "THE GAP",
    "left": "market_B365C_proportional",
    "right": "dc_walkforward",
    "n": 1520,
    "logLossDelta": -0.02979011642184882,
    "logLossCi": [
      -0.042292625009805115,
      -0.017594342262515387
    ],
    "rpsDelta": -0.008807030645038436,
    "rpsCi": [
      -0.012501635307712715,
      -0.005170570919524872
    ],
    "signsAgree": true,
    "excludesZero": true,
    "verdict": "SIGNIFICANT",
    "source": "phase5_market_deltas.csv"
  },
  {
    "comparison": "market - D4",
    "label": "The gap at D4",
    "left": "market_B365C_proportional",
    "right": "D4",
    "n": 1520,
    "logLossDelta": -0.03917604928298953,
    "logLossCi": [
      -0.05040559595192479,
      -0.02787184761320598
    ],
    "rpsDelta": -0.011496510953273526,
    "rpsCi": [
      -0.015050862934122089,
      -0.007917770053681385
    ],
    "signsAgree": true,
    "excludesZero": true,
    "verdict": "SIGNIFICANT",
    "source": "phase5_market_deltas.csv"
  },
  {
    "comparison": "market - D0",
    "label": "The whole measurable range",
    "left": "market_B365C_proportional",
    "right": "D0",
    "n": 1520,
    "logLossDelta": -0.10831411379031382,
    "logLossCi": [
      -0.1280580020497278,
      -0.08891392934172147
    ],
    "rpsDelta": -0.037159380612719645,
    "rpsCi": [
      -0.043881541909206616,
      -0.030606297620126153
    ],
    "signsAgree": true,
    "excludesZero": true,
    "verdict": "SIGNIFICANT",
    "source": "phase5_market_deltas.csv"
  }
] as const;

export const DECOMPOSITION = {
  "total": 0.07852399736846494,
  "totalSource": "phase4_ladder_pooled.csv",
  "derived": true,
  "parts": [
    {
      "label": "Current-season results",
      "span": "D0 -> D1",
      "logLoss": 0.06495884359044668,
      "share": 0.8272483032879068,
      "significant": true,
      "source": "phase4_ladder_pooled.csv"
    },
    {
      "label": "Continuously updated rating state",
      "span": "D1 -> D2 rescaled",
      "logLoss": 0.003645738330471352,
      "share": 0.046428333409519885,
      "significant": true,
      "source": "phase4_a4_deltas.csv"
    },
    {
      "label": "Everything still unaccounted for",
      "span": "D2 rescaled -> Dixon-Coles",
      "logLoss": 0.00991941544754698,
      "share": 0.12632336330257424,
      "significant": false,
      "source": "phase4_a4_deltas.csv"
    },
    {
      "label": "Static historical description",
      "span": "Blocks C and X",
      "logLoss": 0.0,
      "share": 0.0,
      "significant": false,
      "source": "phase4_d34_deltas.csv"
    }
  ]
} as const;

export const GAP = {
  "splits": {
    "favourite probability": [
      {
        "level": "0.00-0.40",
        "n": 202,
        "gap": 0.026165657487334387,
        "ci": [
          -0.007891660674456562,
          0.05905951133736268
        ],
        "gapRps": 0.008209154662945011
      },
      {
        "level": "0.40-0.50",
        "n": 492,
        "gap": 0.03892848291931323,
        "ci": [
          0.015030562279710794,
          0.06467673554773946
        ],
        "gapRps": 0.011278678288383512
      },
      {
        "level": "0.50-0.60",
        "n": 378,
        "gap": 0.02057154804268471,
        "ci": [
          -0.002791247873567007,
          0.04392482015423122
        ],
        "gapRps": 0.00717265278078348
      },
      {
        "level": "0.60-0.70",
        "n": 258,
        "gap": 0.012670937052674178,
        "ci": [
          -0.014013088563493443,
          0.03923641102893085
        ],
        "gapRps": 0.005059766870675584
      },
      {
        "level": "0.70-1.00",
        "n": 190,
        "gap": 0.05156607175127264,
        "ci": [
          0.017197433451055216,
          0.08940868159198635
        ],
        "gapRps": 0.011382342194149391
      }
    ],
    "matchweek bucket": [
      {
        "level": "MW 20-31",
        "n": 480,
        "gap": 0.024121265952852185,
        "ci": [
          0.005768964831732414,
          0.04234471509416559
        ],
        "gapRps": 0.008083996437545128
      },
      {
        "level": "MW 7-19",
        "n": 520,
        "gap": 0.0297562214050626,
        "ci": [
          0.008798741340435824,
          0.05139984963126469
        ],
        "gapRps": 0.008222010564913498
      },
      {
        "level": "MW 1-6",
        "n": 240,
        "gap": 0.04254959850344775,
        "ci": [
          -0.000562408918708242,
          0.09139452340753054
        ],
        "gapRps": 0.010260024910193651
      },
      {
        "level": "MW 32-38",
        "n": 280,
        "gap": 0.028634394758504118,
        "ci": [
          0.0048410152456854275,
          0.05212724045109194
        ],
        "gapRps": 0.009887560065125945
      }
    ],
    "season": [
      {
        "level": "2024-2025",
        "n": 380,
        "gap": 0.008894402096556748,
        "ci": [
          -0.011560467386959622,
          0.02988041389255897
        ],
        "gapRps": 0.004223844269216673
      },
      {
        "level": "2023-2024",
        "n": 380,
        "gap": 0.04176024336790218,
        "ci": [
          0.017419394901747817,
          0.06619514972622423
        ],
        "gapRps": 0.012449445098077688
      },
      {
        "level": "2022-2023",
        "n": 380,
        "gap": 0.03627047837843084,
        "ci": [
          0.010806932780507112,
          0.06198164648150756
        ],
        "gapRps": 0.01037359012062611
      },
      {
        "level": "2025-2026",
        "n": 380,
        "gap": 0.03223534184450553,
        "ci": [
          0.0038958489035627215,
          0.06294290041461201
        ],
        "gapRps": 0.008181243092233267
      }
    ],
    "promoted involved": [
      {
        "level": "neither",
        "n": 1088,
        "gap": 0.030220515780588285,
        "ci": [
          0.0175188184537677,
          0.04262762555377259
        ],
        "gapRps": 0.009157026746765494
      },
      {
        "level": "promoted side",
        "n": 432,
        "gap": 0.028706147666504988,
        "ci": [
          -0.0011807661431633342,
          0.060763966952106126
        ],
        "gapRps": 0.007925558981429548
      }
    ],
    "promoted detail": [
      {
        "level": "neither promoted",
        "n": 1088,
        "gap": 0.030220515780588285,
        "ci": [
          0.017920379477414684,
          0.042769172327245954
        ],
        "gapRps": 0.009157026746765494
      },
      {
        "level": "home promoted",
        "n": 204,
        "gap": 0.006958193033162768,
        "ci": [
          -0.029022248632224513,
          0.04480759630464709
        ],
        "gapRps": 0.003190018705640259
      },
      {
        "level": "away promoted",
        "n": 204,
        "gap": 0.037223089487371734,
        "ci": [
          -0.0028839932390258556,
          0.07932472073899531
        ],
        "gapRps": 0.010248545990116095
      },
      {
        "level": "both promoted",
        "n": 24,
        "gap": 0.14116975657254652,
        "ci": [
          -0.10140443158927404,
          0.47168740458534675
        ],
        "gapRps": 0.028432261751802856
      }
    ],
    "actual outcome": [
      {
        "level": "D",
        "n": 366,
        "gap": 0.048737563530335204,
        "ci": [
          0.028595223867170193,
          0.06981578772523914
        ],
        "gapRps": 0.00955131817215253
      },
      {
        "level": "H",
        "n": 676,
        "gap": 0.03869047802553524,
        "ci": [
          0.019476928194510894,
          0.05924277614905531
        ],
        "gapRps": 0.014124307670976334
      },
      {
        "level": "A",
        "n": 478,
        "gap": 0.002695116242355031,
        "ci": [
          -0.02087101471474065,
          0.025665470474282748
        ],
        "gapRps": 0.0007173057403150492
      }
    ],
    "market pick": [
      {
        "level": "H",
        "n": 958,
        "gap": 0.03622168615788676,
        "ci": [
          0.020520025638239715,
          0.05223129160601164
        ],
        "gapRps": 0.010081493249781566
      },
      {
        "level": "A",
        "n": 562,
        "gap": 0.01882669327749946,
        "ci": [
          -0.0012264389031738452,
          0.0393122654685926
        ],
        "gapRps": 0.006634548126632882
      }
    ],
    "did the favourite deliver": [
      {
        "level": "favourite did not",
        "n": 682,
        "gap": 0.018888626814524426,
        "ci": [
          -0.001547945848290771,
          0.03920213447782919
        ],
        "gapRps": 0.0007710385422350282
      },
      {
        "level": "favourite delivered",
        "n": 838,
        "gap": 0.038662211782463665,
        "ci": [
          0.02356512396331203,
          0.0546033796138434
        ],
        "gapRps": 0.015347062404121878
      }
    ],
    "strong favourite (p>=0.60)": [
      {
        "level": "favourite delivered",
        "n": 320,
        "gap": 0.03392947158567923,
        "ci": [
          0.0166928922632599,
          0.052654351363533115
        ],
        "gapRps": 0.012384067293448372
      },
      {
        "level": "favourite did not",
        "n": 128,
        "gap": 0.01725956628839364,
        "ci": [
          -0.04261054981288738,
          0.07947882692791175
        ],
        "gapRps": -0.0038659114404749504
      }
    ]
  },
  "splitsSource": "phase5_gap_splits.csv",
  "correlations": {
    "columnsExamined": 128,
    "largestAbsR": 0.07850866162331879,
    "largestColumn": "rel_prior_finishing_diff",
    "threshold": 0.0503,
    "clearingThreshold": 20,
    "expectedByChance": 6.4,
    "source": "phase5_gap_correlations.csv"
  },
  "sharpness": {
    "rows": [
      {
        "label": "Market - Bet365 closing",
        "n": 1520,
        "meanMaxP": 0.5390870781056132,
        "meanPDraw": 0.2366247768898007,
        "ece": 0.010593513717450919,
        "biasD": -0.004164696794409839
      },
      {
        "label": "Dixon-Coles",
        "n": 1520,
        "meanMaxP": 0.5489318079985489,
        "meanPDraw": 0.2295561928994064,
        "ece": 0.01677130876897267,
        "biasD": -0.01123328078480415
      },
      {
        "label": "E1a - shots on target",
        "n": 1520,
        "meanMaxP": 0.5174960753012443,
        "meanPDraw": 0.2358341432241826,
        "ece": null,
        "biasD": null
      },
      {
        "label": "D0 - base rate",
        "n": 1520,
        "meanMaxP": null,
        "meanPDraw": null,
        "ece": 0.0076023391812865635,
        "biasD": -0.011403508771929888
      }
    ],
    "derived": true,
    "source": "phase5_e1a_predictions.csv + phase5_market_probabilities.csv",
    "calibrationSource": "phase5_calibration.csv",
    "note": "Mean max p is recomputed from the per-match artefacts. REPORT.md \u00a74.3 cites phase5_calibration.csv, which does not carry those columns; the values reproduce exactly."
  }
} as const;

export const TIER2 = {
  "quantities": {
    "delta_total": {
      "point": 0.03590483591704032,
      "ci": [
        0.01861537637305387,
        0.053544882357943406
      ],
      "excludesZero": true
    },
    "delta_recency": {
      "point": 0.035567916198511594,
      "ci": [
        0.01767355749102408,
        0.05333489832045166
      ],
      "excludesZero": true
    },
    "delta_sample": {
      "point": 0.00033691971852873437,
      "ci": [
        -0.0007136857226280152,
        0.0014063967929684898
      ],
      "excludesZero": false
    },
    "delta_half_minus_static": {
      "point": 0.008231168037209073,
      "ci": [
        0.0014153972334188666,
        0.014892954407663046
      ],
      "excludesZero": true
    },
    "share_recency": {
      "point": 0.9906163136545953,
      "ci": [
        null,
        null
      ],
      "excludesZero": false
    }
  },
  "source": "phase4_tier2_decomposition.csv"
} as const;

export const FOLDS = [
  {
    "fold": 1,
    "trainSeasons": "2021-2022",
    "testSeason": "2022-2023",
    "trainMatches": 380,
    "testMatches": 380,
    "maxTrainDate": "2022-05-22",
    "minTestDate": "2022-08-05",
    "temporalOrderValid": true,
    "overlapValid": true
  },
  {
    "fold": 2,
    "trainSeasons": "2021-2022 + 2022-2023",
    "testSeason": "2023-2024",
    "trainMatches": 760,
    "testMatches": 380,
    "maxTrainDate": "2023-05-28",
    "minTestDate": "2023-08-11",
    "temporalOrderValid": true,
    "overlapValid": true
  },
  {
    "fold": 3,
    "trainSeasons": "2021-2022 + 2022-2023 + 2023-2024",
    "testSeason": "2024-2025",
    "trainMatches": 1140,
    "testMatches": 380,
    "maxTrainDate": "2024-05-19",
    "minTestDate": "2024-08-16",
    "temporalOrderValid": true,
    "overlapValid": true
  },
  {
    "fold": 4,
    "trainSeasons": "2021-2022 + 2022-2023 + 2023-2024 + 2024-2025",
    "testSeason": "2025-2026",
    "trainMatches": 1520,
    "testMatches": 380,
    "maxTrainDate": "2025-05-25",
    "minTestDate": "2025-08-15",
    "temporalOrderValid": true,
    "overlapValid": true
  }
] as const;

export const LEAKAGE = {
  "tests": 9,
  "passed": 9,
  "source": "phase0_leakage_audit.csv"
} as const;

export const PREDICTIONS = [
  {
    "matchId": "2026-2027_Aston-Villa_Nottingham",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-12",
    "scheduledKickoff": "15:00",
    "homeTeam": "Aston Villa",
    "awayTeam": "Nottingham",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.37301039662552904,
    "pDraw": 0.30414183014779933,
    "pAway": 0.32284777322667163,
    "lambdaHome": 1.298014552575443,
    "lambdaAway": 1.192885876807715,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "9f3ac588c1fbb60e7a8fd7344bed1b94a412eda7fac2d68c198bc57f2303559d",
    "prevHash": "0000000000000000000000000000000000000000000000000000000000000000"
  },
  {
    "matchId": "2026-2027_Bournemouth_Brentford",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-12",
    "scheduledKickoff": "15:00",
    "homeTeam": "Bournemouth",
    "awayTeam": "Brentford",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.42522812195964205,
    "pDraw": 0.28700755934046634,
    "pAway": 0.2877643186998917,
    "lambdaHome": 1.498593303210512,
    "lambdaAway": 1.1993171412735888,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "e59df9d5bad697a7de7def253d33847155f988173c91e3de6b79a0efa7d56c71",
    "prevHash": "9f3ac588c1fbb60e7a8fd7344bed1b94a412eda7fac2d68c198bc57f2303559d"
  },
  {
    "matchId": "2026-2027_Chelsea_Hull",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-12",
    "scheduledKickoff": "15:00",
    "homeTeam": "Chelsea",
    "awayTeam": "Hull",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.5484001154140404,
    "pDraw": 0.2611111616804913,
    "pAway": 0.1904887229054685,
    "lambdaHome": 1.7834218273637603,
    "lambdaAway": 0.9780187206006373,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "2de530dfd296489fab5d8acc0f5eae01c269b6de389c4f0509d10544bc1f8396",
    "prevHash": "e59df9d5bad697a7de7def253d33847155f988173c91e3de6b79a0efa7d56c71"
  },
  {
    "matchId": "2026-2027_Crystal-Palace_Ipswich-Town",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-12",
    "scheduledKickoff": "15:00",
    "homeTeam": "Crystal Palace",
    "awayTeam": "Ipswich Town",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.5763335309569086,
    "pDraw": 0.2195548965621015,
    "pAway": 0.20411157248098988,
    "lambdaHome": 2.2647824191798964,
    "lambdaAway": 1.3188339930966062,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "923ed096e99dcdf855250834c529dc742129238fecdd445a38f72e97a4893561",
    "prevHash": "2de530dfd296489fab5d8acc0f5eae01c269b6de389c4f0509d10544bc1f8396"
  },
  {
    "matchId": "2026-2027_Liverpool_Fulham",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-12",
    "scheduledKickoff": "15:00",
    "homeTeam": "Liverpool",
    "awayTeam": "Fulham",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.5901928928461345,
    "pDraw": 0.23385804429430535,
    "pAway": 0.17594906285956016,
    "lambdaHome": 2.0561014594534828,
    "lambdaAway": 1.0613906802805435,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "0ca8815882864407f8b9830a9708f9bc4dc27e1b22cc71ce912209e94f645c2b",
    "prevHash": "923ed096e99dcdf855250834c529dc742129238fecdd445a38f72e97a4893561"
  },
  {
    "matchId": "2026-2027_Tottenham_Everton",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-12",
    "scheduledKickoff": "17:30",
    "homeTeam": "Tottenham",
    "awayTeam": "Everton",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.2498166676005715,
    "pDraw": 0.31031125140549926,
    "pAway": 0.4398720809939293,
    "lambdaHome": 0.9605919446140725,
    "lambdaAway": 1.348610870527354,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "49dffac7f9fbcbc6d75fcc4f83807069f6477bf257616a059f7ad4cdfdf341d0",
    "prevHash": "0ca8815882864407f8b9830a9708f9bc4dc27e1b22cc71ce912209e94f645c2b"
  },
  {
    "matchId": "2026-2027_Sunderland_Arsenal",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-12",
    "scheduledKickoff": "20:00",
    "homeTeam": "Sunderland",
    "awayTeam": "Arsenal",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.1359568110257159,
    "pDraw": 0.2977312056376611,
    "pAway": 0.566311983336623,
    "lambdaHome": 0.5874149556789713,
    "lambdaAway": 1.4444066549397891,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "891e794a12c6cbe52092b89f1d4be3d8a5a21c7c4b0f903a6e3e1c85d634a152",
    "prevHash": "49dffac7f9fbcbc6d75fcc4f83807069f6477bf257616a059f7ad4cdfdf341d0"
  },
  {
    "matchId": "2026-2027_Coventry_Brighton",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-13",
    "scheduledKickoff": "14:00",
    "homeTeam": "Coventry",
    "awayTeam": "Brighton",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.30467517227054686,
    "pDraw": 0.28806646868140157,
    "pAway": 0.4072583590480517,
    "lambdaHome": 1.2420766314417202,
    "lambdaAway": 1.4653908373628604,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "dd6c2841a5ee3626f0a70510e3968f922d3755b077fe13ec93ce51fdef464c0a",
    "prevHash": "891e794a12c6cbe52092b89f1d4be3d8a5a21c7c4b0f903a6e3e1c85d634a152"
  },
  {
    "matchId": "2026-2027_Manchester-Utd_Manchester-City",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-13",
    "scheduledKickoff": "16:30",
    "homeTeam": "Manchester Utd",
    "awayTeam": "Manchester City",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.2928029845556577,
    "pDraw": 0.2472515183008351,
    "pAway": 0.4599454971435073,
    "lambdaHome": 1.496216560123327,
    "lambdaAway": 1.9005170572724575,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "ea0cd8be8345dbf8ffbea96a23c45c0f4e622ca79dd9b720aaef0778d8027b60",
    "prevHash": "dd6c2841a5ee3626f0a70510e3968f922d3755b077fe13ec93ce51fdef464c0a"
  },
  {
    "matchId": "2026-2027_Leeds-United_Newcastle",
    "season": "2026-2027",
    "roundId": 1,
    "matchweekLabel": 4,
    "scheduledDate": "2026-09-14",
    "scheduledKickoff": "20:00",
    "homeTeam": "Leeds United",
    "awayTeam": "Newcastle",
    "stateCutoffDate": "2026-09-12",
    "generatedAtUtc": "2026-09-11T19:21:39Z",
    "pHome": 0.4165181421630724,
    "pDraw": 0.29335946409736585,
    "pAway": 0.29012239373956167,
    "lambdaHome": 1.4398194685158703,
    "lambdaAway": 1.1688339051425047,
    "rho": -0.13244300576120613,
    "fitMatches": 1930,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "84e2b736ea35d9f00bfb06f92cebc84f7b5763d796593f48b2e3b4e1ef45db34",
    "prevHash": "ea0cd8be8345dbf8ffbea96a23c45c0f4e622ca79dd9b720aaef0778d8027b60"
  },
  {
    "matchId": "2026-2027_Brentford_Chelsea",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-18",
    "scheduledKickoff": "20:00",
    "homeTeam": "Brentford",
    "awayTeam": "Chelsea",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.47121393417500984,
    "pDraw": 0.23797816464123936,
    "pAway": 0.290807901183751,
    "lambdaHome": 2.061187780393676,
    "lambdaAway": 1.6086009111526627,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "038fb389f07596239d4cb8f3fd5974d714d773fd3908e5a29320dd99b80593e9",
    "prevHash": "84e2b736ea35d9f00bfb06f92cebc84f7b5763d796593f48b2e3b4e1ef45db34"
  },
  {
    "matchId": "2026-2027_Tottenham_Aston-Villa",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-19",
    "scheduledKickoff": "12:30",
    "homeTeam": "Tottenham",
    "awayTeam": "Aston Villa",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.2650546343846298,
    "pDraw": 0.32365587562859843,
    "pAway": 0.41128948998677184,
    "lambdaHome": 0.9786633581369748,
    "lambdaAway": 1.2732055822903179,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "3d13a46195fdfb815c0818459b948711f7d686256a954517bd1a077789551e1e",
    "prevHash": "038fb389f07596239d4cb8f3fd5974d714d773fd3908e5a29320dd99b80593e9"
  },
  {
    "matchId": "2026-2027_Brighton_Arsenal",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-19",
    "scheduledKickoff": "15:00",
    "homeTeam": "Brighton",
    "awayTeam": "Arsenal",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.21634624947028705,
    "pDraw": 0.30175265685756825,
    "pAway": 0.4819010936721445,
    "lambdaHome": 0.9214153626708825,
    "lambdaAway": 1.4761426923595415,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "ac563ea3781720f605828f24a97de6a968e2667b6c9b42ea45b993ea42fa398e",
    "prevHash": "3d13a46195fdfb815c0818459b948711f7d686256a954517bd1a077789551e1e"
  },
  {
    "matchId": "2026-2027_Everton_Ipswich-Town",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-19",
    "scheduledKickoff": "15:00",
    "homeTeam": "Everton",
    "awayTeam": "Ipswich Town",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.600388521776796,
    "pDraw": 0.21532962782819506,
    "pAway": 0.1842818503950089,
    "lambdaHome": 2.3468787036847387,
    "lambdaAway": 1.2752814686149356,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "aba77a080ee3774bb4b6b1bce66af80072cf83661a234dd2693734ea549c9ac8",
    "prevHash": "ac563ea3781720f605828f24a97de6a968e2667b6c9b42ea45b993ea42fa398e"
  },
  {
    "matchId": "2026-2027_Newcastle_Hull",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-19",
    "scheduledKickoff": "15:00",
    "homeTeam": "Newcastle",
    "awayTeam": "Hull",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.2158431595104573,
    "pDraw": 0.3517461495863058,
    "pAway": 0.43241069090323697,
    "lambdaHome": 0.7308305304489711,
    "lambdaAway": 1.1362560176326593,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "4a2999f52b77948974f439d9ccd9c7ab0aff02daa09f4b9ca361f7c82f6b43bd",
    "prevHash": "aba77a080ee3774bb4b6b1bce66af80072cf83661a234dd2693734ea549c9ac8"
  },
  {
    "matchId": "2026-2027_Nottingham_Coventry",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-19",
    "scheduledKickoff": "17:30",
    "homeTeam": "Nottingham",
    "awayTeam": "Coventry",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.6691964224680876,
    "pDraw": 0.20682533522013327,
    "pAway": 0.12397824231177915,
    "lambdaHome": 2.3123679726464577,
    "lambdaAway": 0.935483879062256,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "bfc102d8352b89a64d55d75e5b7fa1fc02370fabc314a42f8fb9870e95fa923b",
    "prevHash": "4a2999f52b77948974f439d9ccd9c7ab0aff02daa09f4b9ca361f7c82f6b43bd"
  },
  {
    "matchId": "2026-2027_Bournemouth_Liverpool",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-20",
    "scheduledKickoff": "14:00",
    "homeTeam": "Bournemouth",
    "awayTeam": "Liverpool",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.40411977814369304,
    "pDraw": 0.27957941852263846,
    "pAway": 0.3163008033336686,
    "lambdaHome": 1.5689271940720044,
    "lambdaAway": 1.3707569027393969,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "b3d1fe2ae0dbee1c4f16cab3f789ee78dd41500a3887260e52e600e0f30a37b7",
    "prevHash": "bfc102d8352b89a64d55d75e5b7fa1fc02370fabc314a42f8fb9870e95fa923b"
  },
  {
    "matchId": "2026-2027_Leeds-United_Crystal-Palace",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-20",
    "scheduledKickoff": "14:00",
    "homeTeam": "Leeds United",
    "awayTeam": "Crystal Palace",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.629780662836829,
    "pDraw": 0.23100313245941492,
    "pAway": 0.13921620470375623,
    "lambdaHome": 2.072310930843501,
    "lambdaAway": 0.9005962076193678,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "c7b101f209eb3c9bfd6cf6223d4816e91aa31189fca9e87e032744a9b822100d",
    "prevHash": "b3d1fe2ae0dbee1c4f16cab3f789ee78dd41500a3887260e52e600e0f30a37b7"
  },
  {
    "matchId": "2026-2027_Manchester-City_Sunderland",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-20",
    "scheduledKickoff": "14:00",
    "homeTeam": "Manchester City",
    "awayTeam": "Sunderland",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.6672721079194406,
    "pDraw": 0.23944163062244092,
    "pAway": 0.09328626145811879,
    "lambdaHome": 1.8528496156712577,
    "lambdaAway": 0.5754441687803378,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "3152904ff53a573785d8ebd7f60e54d33e0795eaaa8c7cea5e6f7f756b3d4b78",
    "prevHash": "c7b101f209eb3c9bfd6cf6223d4816e91aa31189fca9e87e032744a9b822100d"
  },
  {
    "matchId": "2026-2027_Fulham_Manchester-Utd",
    "season": "2026-2027",
    "roundId": 2,
    "matchweekLabel": 5,
    "scheduledDate": "2026-09-20",
    "scheduledKickoff": "16:30",
    "homeTeam": "Fulham",
    "awayTeam": "Manchester Utd",
    "stateCutoffDate": "2026-09-18",
    "generatedAtUtc": "2026-09-18T05:57:30Z",
    "pHome": 0.24314163520851803,
    "pDraw": 0.2774750334112266,
    "pAway": 0.4793833313802554,
    "lambdaHome": 1.1439456846871714,
    "lambdaAway": 1.672075017373046,
    "rho": -0.15415721664317825,
    "fitMatches": 1940,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "35c6c582d973c9e1f259a7c7ff63e11c3371ca14c8ed22a1aa1f867629afe8ff",
    "prevHash": "3152904ff53a573785d8ebd7f60e54d33e0795eaaa8c7cea5e6f7f756b3d4b78"
  },
  {
    "matchId": "2026-2027_Ipswich-Town_Liverpool",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-04",
    "scheduledKickoff": "20:00",
    "homeTeam": "Ipswich Town",
    "awayTeam": "Liverpool",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.26482371227572304,
    "pDraw": 0.17729443024636415,
    "pAway": 0.5578818574779127,
    "lambdaHome": 2.0848129037608527,
    "lambdaAway": 2.9477348633498606,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "837ca8a05e48c79345a5101a5575aee7b091e8b49d4bce56ef43f8e99ca9df54",
    "prevHash": "35c6c582d973c9e1f259a7c7ff63e11c3371ca14c8ed22a1aa1f867629afe8ff"
  },
  {
    "matchId": "2026-2027_Newcastle_Bournemouth",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-05",
    "scheduledKickoff": "12:30",
    "homeTeam": "Newcastle",
    "awayTeam": "Bournemouth",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.40093108781821224,
    "pDraw": 0.2712063052877123,
    "pAway": 0.3278626068940754,
    "lambdaHome": 1.4712560401121337,
    "lambdaAway": 1.310446770950357,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "ebf99b2e16b6406fb7e4094690be88b12d4789618ddac36b51169861e598d21f",
    "prevHash": "837ca8a05e48c79345a5101a5575aee7b091e8b49d4bce56ef43f8e99ca9df54"
  },
  {
    "matchId": "2026-2027_Brentford_Sunderland",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-05",
    "scheduledKickoff": "15:00",
    "homeTeam": "Brentford",
    "awayTeam": "Sunderland",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.5294629641702393,
    "pDraw": 0.27662782613116743,
    "pAway": 0.19390920969859332,
    "lambdaHome": 1.5302406275009637,
    "lambdaAway": 0.8283224533678277,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "ec18f1f05d3e66427b2e396cfe645552a2e7be5ef13ff304431471b0ae321128",
    "prevHash": "ebf99b2e16b6406fb7e4094690be88b12d4789618ddac36b51169861e598d21f"
  },
  {
    "matchId": "2026-2027_Brighton_Leeds-United",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-05",
    "scheduledKickoff": "15:00",
    "homeTeam": "Brighton",
    "awayTeam": "Leeds United",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.4783535509945238,
    "pDraw": 0.27036662748568807,
    "pAway": 0.25127982151978834,
    "lambdaHome": 1.5701027668419545,
    "lambdaAway": 1.0766672905910288,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "ec4958af2800982fc11ce74ae519ec9ae9847ad2151857d7b591e10566e5826b",
    "prevHash": "ec18f1f05d3e66427b2e396cfe645552a2e7be5ef13ff304431471b0ae321128"
  },
  {
    "matchId": "2026-2027_Fulham_Crystal-Palace",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-05",
    "scheduledKickoff": "15:00",
    "homeTeam": "Fulham",
    "awayTeam": "Crystal Palace",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.49166424162236805,
    "pDraw": 0.28185475027218426,
    "pAway": 0.22648100810544766,
    "lambdaHome": 1.482453249108004,
    "lambdaAway": 0.9273024994098659,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "09c13f51fb6a2f755a482aaca9217cfbb4c8bfc6b7c29b1e2f4a76cb44322446",
    "prevHash": "ec4958af2800982fc11ce74ae519ec9ae9847ad2151857d7b591e10566e5826b"
  },
  {
    "matchId": "2026-2027_Manchester-City_Coventry",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-05",
    "scheduledKickoff": "15:00",
    "homeTeam": "Manchester City",
    "awayTeam": "Coventry",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.8169843227442222,
    "pDraw": 0.1232779103672678,
    "pAway": 0.05973776688851012,
    "lambdaHome": 3.002964282710584,
    "lambdaAway": 0.7577095433631029,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "945e3f20ab377301adb0b0c85a4ba4f39743e951d1f64f0bbd5f78f04f2e8466",
    "prevHash": "09c13f51fb6a2f755a482aaca9217cfbb4c8bfc6b7c29b1e2f4a76cb44322446"
  },
  {
    "matchId": "2026-2027_Nottingham_Tottenham",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-05",
    "scheduledKickoff": "15:00",
    "homeTeam": "Nottingham",
    "awayTeam": "Tottenham",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.6256564079647642,
    "pDraw": 0.23114044927791957,
    "pAway": 0.14320314275731616,
    "lambdaHome": 1.8995089926980724,
    "lambdaAway": 0.7992351101222768,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "297aee4cda7457f1f4a39b05022d2ec81ea9325ed92c3cdaa0ce273eda755f39",
    "prevHash": "945e3f20ab377301adb0b0c85a4ba4f39743e951d1f64f0bbd5f78f04f2e8466"
  },
  {
    "matchId": "2026-2027_Hull_Aston-Villa",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-05",
    "scheduledKickoff": "17:30",
    "homeTeam": "Hull",
    "awayTeam": "Aston Villa",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.4776106767087163,
    "pDraw": 0.26399173828220085,
    "pAway": 0.2583975850090826,
    "lambdaHome": 1.6263805078460003,
    "lambdaAway": 1.140889659395198,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "6444f7a4410affb1f65104116323c0ac6f33f7be9dc3d2914cf2ae26db2ade47",
    "prevHash": "297aee4cda7457f1f4a39b05022d2ec81ea9325ed92c3cdaa0ce273eda755f39"
  },
  {
    "matchId": "2026-2027_Everton_Manchester-Utd",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-06",
    "scheduledKickoff": "14:00",
    "homeTeam": "Everton",
    "awayTeam": "Manchester Utd",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.33528531944983575,
    "pDraw": 0.2615480472815424,
    "pAway": 0.4031666332686219,
    "lambdaHome": 1.4028105785778118,
    "lambdaAway": 1.5563463031175166,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "a781e018e4e64ea548046ae3097fb0020be97c1d14ec92a52221229f034846ce",
    "prevHash": "6444f7a4410affb1f65104116323c0ac6f33f7be9dc3d2914cf2ae26db2ade47"
  },
  {
    "matchId": "2026-2027_Arsenal_Chelsea",
    "season": "2026-2027",
    "roundId": 3,
    "matchweekLabel": 3,
    "scheduledDate": "2026-09-06",
    "scheduledKickoff": "16:30",
    "homeTeam": "Arsenal",
    "awayTeam": "Chelsea",
    "stateCutoffDate": "2026-09-04",
    "generatedAtUtc": "2026-09-18T06:25:49Z",
    "pHome": 0.7129778088794931,
    "pDraw": 0.18639883144341374,
    "pAway": 0.10062335967709317,
    "lambdaHome": 2.281758061978557,
    "lambdaAway": 0.7519912140170065,
    "rho": -0.07694763157630441,
    "fitMatches": 1920,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "e932834a83c7dd1ab6745c1f364f794cc1eab9e4ee83612d9b2275c3c1fcf5cc",
    "prevHash": "a781e018e4e64ea548046ae3097fb0020be97c1d14ec92a52221229f034846ce"
  },
  {
    "matchId": "2026-2027_Arsenal_Leeds-United",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-10",
    "scheduledKickoff": "12:30",
    "homeTeam": "Arsenal",
    "awayTeam": "Leeds United",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.5079641459371553,
    "pDraw": 0.31559515109724184,
    "pAway": 0.17644070296560266,
    "lambdaHome": 1.314647667262951,
    "lambdaAway": 0.6706206642783912,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "e6611f20bff6d0235dd6f21c1de0b47ce9bcab325d5d27bad76421741fa65101",
    "prevHash": "e932834a83c7dd1ab6745c1f364f794cc1eab9e4ee83612d9b2275c3c1fcf5cc"
  },
  {
    "matchId": "2026-2027_Aston-Villa_Brentford",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-10",
    "scheduledKickoff": "15:00",
    "homeTeam": "Aston Villa",
    "awayTeam": "Brentford",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.30785608658001484,
    "pDraw": 0.2698628383723121,
    "pAway": 0.4222810750476729,
    "lambdaHome": 1.3378012816707066,
    "lambdaAway": 1.5960621433071804,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "d4a97107865781500b59736f6ba823dd8c2e9e58d48858b9aaf2ce28c8109232",
    "prevHash": "e6611f20bff6d0235dd6f21c1de0b47ce9bcab325d5d27bad76421741fa65101"
  },
  {
    "matchId": "2026-2027_Chelsea_Bournemouth",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-10",
    "scheduledKickoff": "15:00",
    "homeTeam": "Chelsea",
    "awayTeam": "Bournemouth",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.38981728211207156,
    "pDraw": 0.24847153982265224,
    "pAway": 0.3617111780652763,
    "lambdaHome": 1.732704175941235,
    "lambdaAway": 1.6650951958709783,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "973966226c2038d023808c3fec237c9b7581b1a0b6eaa953d314feebccc23ad9",
    "prevHash": "d4a97107865781500b59736f6ba823dd8c2e9e58d48858b9aaf2ce28c8109232"
  },
  {
    "matchId": "2026-2027_Ipswich-Town_Fulham",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-10",
    "scheduledKickoff": "15:00",
    "homeTeam": "Ipswich Town",
    "awayTeam": "Fulham",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.3346113485883793,
    "pDraw": 0.2726772532570218,
    "pAway": 0.39271139815459905,
    "lambdaHome": 1.3928410202440271,
    "lambdaAway": 1.5233662551988674,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "0b0786f383fba70bbce1f82e162f0d73a2754ab4337335a73b44fcdf1f444d3e",
    "prevHash": "973966226c2038d023808c3fec237c9b7581b1a0b6eaa953d314feebccc23ad9"
  },
  {
    "matchId": "2026-2027_Sunderland_Brighton",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-10",
    "scheduledKickoff": "15:00",
    "homeTeam": "Sunderland",
    "awayTeam": "Brighton",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.20429962108453956,
    "pDraw": 0.23694731186652723,
    "pAway": 0.5587530670489331,
    "lambdaHome": 1.1580587775611675,
    "lambdaAway": 2.0060104463837996,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "fd8d020f2d6681ed7282a52953e832183a38f44cc48e738612db3a8a6e9d3ed2",
    "prevHash": "0b0786f383fba70bbce1f82e162f0d73a2754ab4337335a73b44fcdf1f444d3e"
  },
  {
    "matchId": "2026-2027_Manchester-Utd_Tottenham",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-10",
    "scheduledKickoff": "17:30",
    "homeTeam": "Manchester Utd",
    "awayTeam": "Tottenham",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.6703815244384412,
    "pDraw": 0.20929483016126507,
    "pAway": 0.12032364540029365,
    "lambdaHome": 2.1715457243364744,
    "lambdaAway": 0.8308233136053247,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "cf8ac6afd1a81734f1eddacc0c08d8b17c2ef0bfbbe8b9dc34f6d8ef7f413b7d",
    "prevHash": "fd8d020f2d6681ed7282a52953e832183a38f44cc48e738612db3a8a6e9d3ed2"
  },
  {
    "matchId": "2026-2027_Crystal-Palace_Nottingham",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-11",
    "scheduledKickoff": "14:00",
    "homeTeam": "Crystal Palace",
    "awayTeam": "Nottingham",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.2918450700241027,
    "pDraw": 0.29762046948198423,
    "pAway": 0.41053446049391323,
    "lambdaHome": 1.1176697644189946,
    "lambdaAway": 1.3666845747928613,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "e2690a1c7e5d41a93d008a1b2b6a2a4411abfce3921660d916736000e75bab47",
    "prevHash": "cf8ac6afd1a81734f1eddacc0c08d8b17c2ef0bfbbe8b9dc34f6d8ef7f413b7d"
  },
  {
    "matchId": "2026-2027_Hull_Everton",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-11",
    "scheduledKickoff": "14:00",
    "homeTeam": "Hull",
    "awayTeam": "Everton",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.40452402308285884,
    "pDraw": 0.37344307581960173,
    "pAway": 0.2220329010975394,
    "lambdaHome": 0.9657718089440543,
    "lambdaAway": 0.6443525403824911,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "bb9018f00e7d6fe3782adbc95eb5ac842ed94b3b0ee2a8adb56b100c6bd903a0",
    "prevHash": "e2690a1c7e5d41a93d008a1b2b6a2a4411abfce3921660d916736000e75bab47"
  },
  {
    "matchId": "2026-2027_Liverpool_Manchester-City",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-11",
    "scheduledKickoff": "16:30",
    "homeTeam": "Liverpool",
    "awayTeam": "Manchester City",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.2777362841706234,
    "pDraw": 0.2683193736434909,
    "pAway": 0.4539443421858856,
    "lambdaHome": 1.2518998484028618,
    "lambdaAway": 1.6490556113344275,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "6cc6f27c13cbe12cf2b3feb32bc1f45c9d7bdf1a0431a53ffd09179ea5f15e7b",
    "prevHash": "bb9018f00e7d6fe3782adbc95eb5ac842ed94b3b0ee2a8adb56b100c6bd903a0"
  },
  {
    "matchId": "2026-2027_Coventry_Newcastle",
    "season": "2026-2027",
    "roundId": 4,
    "matchweekLabel": 6,
    "scheduledDate": "2026-10-12",
    "scheduledKickoff": "20:00",
    "homeTeam": "Coventry",
    "awayTeam": "Newcastle",
    "stateCutoffDate": "2026-10-10",
    "generatedAtUtc": "2026-09-23T10:07:18Z",
    "pHome": 0.06180499573578181,
    "pDraw": 0.2173940893290873,
    "pAway": 0.7208009149351307,
    "lambdaHome": 0.39006472421563904,
    "lambdaAway": 1.8397407277637725,
    "rho": -0.114435159585828,
    "fitMatches": 1950,
    "homeHasHistory": true,
    "awayHasHistory": true,
    "rowHash": "9e42dac5534a160a23e7a64188e0c6918275e81c903f01e7ab4e64dd832341ae",
    "prevHash": "6cc6f27c13cbe12cf2b3feb32bc1f45c9d7bdf1a0431a53ffd09179ea5f15e7b"
  }
] as const;

export const CONTAMINATED = [
  {
    "matchId": "2026-2027_Ipswich-Town_Liverpool",
    "playedDate": "2026-09-04",
    "homeTeam": "Ipswich Town",
    "awayTeam": "Liverpool",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  },
  {
    "matchId": "2026-2027_Hull_Aston-Villa",
    "playedDate": "2026-09-05",
    "homeTeam": "Hull",
    "awayTeam": "Aston Villa",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  },
  {
    "matchId": "2026-2027_Nottingham_Tottenham",
    "playedDate": "2026-09-05",
    "homeTeam": "Nottingham",
    "awayTeam": "Tottenham",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  },
  {
    "matchId": "2026-2027_Manchester-City_Coventry",
    "playedDate": "2026-09-05",
    "homeTeam": "Manchester City",
    "awayTeam": "Coventry",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  },
  {
    "matchId": "2026-2027_Newcastle_Bournemouth",
    "playedDate": "2026-09-05",
    "homeTeam": "Newcastle",
    "awayTeam": "Bournemouth",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  },
  {
    "matchId": "2026-2027_Brighton_Leeds-United",
    "playedDate": "2026-09-05",
    "homeTeam": "Brighton",
    "awayTeam": "Leeds United",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  },
  {
    "matchId": "2026-2027_Brentford_Sunderland",
    "playedDate": "2026-09-05",
    "homeTeam": "Brentford",
    "awayTeam": "Sunderland",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  },
  {
    "matchId": "2026-2027_Fulham_Crystal-Palace",
    "playedDate": "2026-09-05",
    "homeTeam": "Fulham",
    "awayTeam": "Crystal Palace",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  },
  {
    "matchId": "2026-2027_Everton_Manchester-Utd",
    "playedDate": "2026-09-06",
    "homeTeam": "Everton",
    "awayTeam": "Manchester Utd",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  },
  {
    "matchId": "2026-2027_Arsenal_Chelsea",
    "playedDate": "2026-09-06",
    "homeTeam": "Arsenal",
    "awayTeam": "Chelsea",
    "reason": "no pre-kickoff prediction existed; the live log began after this match was played (L2.1)",
    "recordedAtUtc": "2026-09-11T19:22:08Z",
    "excludedFromLog": true,
    "excludedFromHoldout": false
  }
] as const;

