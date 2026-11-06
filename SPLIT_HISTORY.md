# Split history for ap2026-q2

## Provenance and split disclosure

The work was done in one repository (parent tag `submission`, commit `4521da3`) and split afterwards into one repository per question following the instructor's guidance. Commit hashes changed and the old-to-new commit map is included below. Dates in rewritten commits are the original ones; the split commits are dated when they were made. Shared-file commits appear in more than one repository. `SESSION.md` is the full transcript for all questions and refers to old file names and hashes. Agent work on the split ran with `ap2026-split` as its root.

## Path mapping

| Old path or name | New path or name |
|---|---|
| `code/approach3_q2a_mc.py` | `code/approach1_q2a_mc.py` |
| `code/approach4_q2b_deephedge.py` | `code/approach2_q2b_deephedge.py` |
| `results/approach3/` | `results/approach1/` |
| `results/approach4/` | `results/approach2/` |

Some code comments, labels and JSON keys keep the old names, such as approach3 or approach5_members. They were left unchanged so that results stay directly comparable with the parent.

## Commit map

The map is the unedited `git filter-repo` output. `pruned` means the commit became empty for this path set. `same` means filter-repo produced the identical hash; `rewritten` means the hash changed. Shared commits are listed with every repository that retained them.

| Old commit | New commit | Subject | Status | Retained in |
|---|---|---|---|---|
| `7ee9a860ad8a` | `7ee9a860ad8a` | chore: add agent contract before first agent session | same | Q1, Q2, Q3 |
| `bc857ecc516a` | `bc857ecc516a` | chore: scaffold repository | same | Q1, Q2, Q3 |
| `867a243cd90d` | `165ee8cde0ca` | data: add Q1 fetch pipeline and snapshot | rewritten | Q1, Q2, Q3 |
| `278562b16b99` | `c339f374e301` | data: roll holiday option expiries to next trading day | rewritten | Q1, Q2, Q3 |
| `0b680be37ac1` | `41cfe2c03aa2` | feat: Q1A SVI pipeline | rewritten | Q1, Q2, Q3 |
| `c8c840c4b3e4` | pruned | exp: Q1A results on IO | pruned | Q1 |
| `317fe9c55a8a` | pruned | fix: Q1A repricing basis and SVI bounds | pruned | Q1 |
| `18b5cc2f237d` | pruned | exp: Q1A results, settle primary | pruned | Q1 |
| `0e0a6b91984a` | pruned | chore: remove stale Q1A run | pruned | Q1 |
| `7551ad29752f` | pruned | exp: like-for-like forward check | pruned | Q1 |
| `0af1dd09691f` | pruned | feat: Q1B differentiable SVI and neural surface | pruned | Q1 |
| `ab3313adfc8f` | pruned | fix: Q1B robustness coverage reporting | pruned | Q1 |
| `664a71673bbd` | pruned | exp: Q1B results on IO | pruned | Q1 |
| `84a746a51ab8` | pruned | docs: record Q1B open issues | pruned | Q1 |
| `5f007e65d2ca` | `f82b220d7b2e` | data: add Q2 futures snapshot | rewritten | Q1, Q2, Q3 |
| `7950d6fe1852` | `9365eb214e68` | feat: Q2A Monte Carlo engine and estimators | rewritten | Q1, Q2, Q3 |
| `4fd0ca365927` | `dc34caf8d299` | exp: Q2A results | rewritten | Q2 |
| `32bfbcdf9c77` | `3b93bee7aea8` | fix: Q2A study reference consistency | rewritten | Q2 |
| `89a27c6db5ed` | `be95066437b1` | exp: refresh Q2A study outputs | rewritten | Q2 |
| `d67df3903f33` | `db601c8262e6` | feat: Q2B deep hedging | rewritten | Q2 |
| `b2a71278ed7d` | `b114ee6ef19d` | fix: Q2B default output path | rewritten | Q2 |
| `fcdaf1096e58` | `f19cd2db52f9` | exp: Q2B results | rewritten | Q2 |
| `4fc1d1053860` | `c2ef60208581` | data: add Q3 macro panel | rewritten | Q1, Q2, Q3 |
| `6255defa60cc` | pruned | feat: Q3A factors, nowcast, probit, clustering | pruned | Q3 |
| `1922099796cc` | pruned | exp: Q3A results | pruned | Q3 |
| `d2cf1893d830` | pruned | exp: Q3A post-hoc probit regularization check | pruned | Q3 |
| `a87b9f3dfb18` | pruned | feat: Q3B LSTM and leakage ablation | pruned | Q3 |
| `5131fe20204d` | pruned | exp: Q3B results | pruned | Q3 |
| `9790ce9e52a3` | pruned | fix: Q3B training loop diagnostics | pruned | Q3 |
| `17bbfd75a41e` | pruned | exp: Q3B post-hoc retraining | pruned | Q3 |
| `fb4781adbadd` | `fe4b3bcfd792` | feat: compare layer and reproduction script | rewritten | Q1, Q2, Q3 |
| `b5c09a83baf1` | `a1e499e51954` | docs: update agent contract and README | rewritten | Q1, Q2, Q3 |
| `7290853b7bbe` | `3277f23395c9` | fix: Q1B objective scaling and shared initialization | rewritten | Q1, Q2, Q3 |
| `d4a7f215effb` | `8ee36a0af23a` | exp: Q1B results after objective scaling | rewritten | Q1, Q2, Q3 |
| `80009968446d` | pruned | docs: correct Q1B tenor-bucket aggregation | pruned | Q1 |
| `076e9d43a48c` | `1d94709e13c4` | fix: use repo-relative paths in outputs | rewritten | Q1, Q2, Q3 |
| `e6ae1a473870` | `f29c836ec797` | docs: note historical approach2/io run | rewritten | Q1, Q2, Q3 |
| `ab12e7080c73` | `3c0888fcad5f` | data: commit CFFEX raw snapshot for offline Q1 cross-check | rewritten | Q1, Q2, Q3 |
| `4a6164be766d` | `330bb0bcb395` | fix: commit legacy SVI snapshots used by Q1 comparisons | rewritten | Q1, Q2, Q3 |
| `509ac9f68ea4` | `69cb9ba43a8d` | chore: add reproduction check | rewritten | Q1, Q2, Q3 |
| `311bade4fe68` | `5b6460834068` | docs: document reproduction caveats | rewritten | Q1, Q2, Q3 |
| `8105e5dfd0ec` | `215438e95ed4` | exp: refresh Q1A close results from current code | rewritten | Q1, Q2, Q3 |
| `b226622810fd` | pruned | fix: Q3B post-hoc origin_date label | pruned | Q3 |
| `e27222bdeaf5` | `a753ab9e334c` | feat: Q1B cold-start SVI solver comparison | rewritten | Q1, Q2, Q3 |
| `99210e96e740` | pruned | exp: Q1B cold-start solver results | pruned | Q1 |
| `3015c7822b0c` | `0a8d683b2796` | report: add project report | rewritten | Q1, Q2, Q3 |
| `dad4ca07163c` | `22bc4069bf9d` | report: author review confirmation | rewritten | Q1, Q2, Q3 |
| `4521da3ed674` | `dbdbdecf1325` | report: add complete session export | rewritten | Q1, Q2, Q3 |

## Commit counts

- Original parent history: 48 commits.
- Commits retained by this repository after filtering: 29.
- Later split commits are not part of the filter-repo map; they are ordinary commits on top of the filtered history and are dated when made.
- No `submission` tag was created by the split; the parent tag is recorded above.
