# Noise buffering and amplification by binding reservoirs — code and data

Code and data reproducing every figure and every quoted number in:

> *The limits of decoy buffering: Binding reservoirs amplify extrinsic noise unless they scale with the source*
> — Shay Tal, submitted to PLOS Computational Biology.

[![DOI](https://zenodo.org/badge/DOI/ZENODO_DOI_HERE.svg)](https://doi.org/ZENODO_DOI_HERE)

## Quick start

```bash
conda env create -f environment.yml
conda activate decoy-buffering
python make_all.py
```

This regenerates all eight datasets in `data/`, all three figures in
`figures/`, `reported_values.csv`, and `S1_Data.xlsx`, then checks the
regenerated data against the published copies in `reference/`. Runtime is
dominated by the age-structured replication sweeps; budget a few hours on a
laptop. `python make_all.py --figures` rebuilds only the figures from data
that already exists.

## Checking a single number without running anything

`reported_values.csv` lists every number quoted in the manuscript next to the
value recomputed from the data, the relative difference, and the exact
criterion it came from. Regenerate it in a few seconds with:

```bash
python reported_values.py
```

It exits non-zero if any claim falls outside its tolerance. With the shipped
`reference/` data all claims pass.

## Layout

| path | contents |
|---|---|
| `decoy_sim.py` | the model: three solvers, described below |
| `scripts/01–08` | parameter sweeps, one per dataset, each standalone |
| `scripts/09–11` | the three manuscript figures |
| `reported_values.py` | recomputes every quoted number; writes `reported_values.csv` |
| `make_s1_data.py` | builds `S1_Data.xlsx`, one sheet per figure panel |
| `verify.py` | diffs regenerated `data/` against published `reference/` |
| `reference/` | the exact datasets behind the submitted figures |

Each script under `scripts/` is self-contained: it recomputes whatever it
needs rather than reading another script's output, so any one can be run on
its own. The cost is some redundant computation when running the whole
pipeline.

## The three solvers in `decoy_sim.py`

`adiabatic_cme` — exact solution of the master equation in the fast-binding
limit. The bound count is at conditional equilibrium with a closed-form law,
so the total pool is a Markov jump process whose stationary distribution
follows from a sparse linear solve. **No Monte Carlo error.** Used for all
steady-state sweeps. Truncations are checked to boundary mass < 1e-9.

`ssa_full` — exact stochastic simulation with explicit binding and unbinding,
with no adiabatic assumption. Used to validate the reduction above.

`ssa_cellcycle` — age-structured single-lineage simulation: decoy number
doubles at replication age, division applies binomial partitioning to free and
bound molecules, and cell age is sampled uniformly.

## Reproducibility caveats

The deterministic solvers reproduce to machine precision on any platform. The
stochastic components are seeded, but the random stream is not portable across
NumPy and numba versions, so `verify.py` applies a 2% relative tolerance
rather than requiring identical digits. Pin the environment with
`environment.yml` to stay closest to the published values. Versions used:
Python 3.12, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, matplotlib 3.11.1,
numba 0.67.0.

Figures are written as PNG at 300 dpi. The TIFF files submitted to the journal
are the same images retagged at 300–360 dpi so that their physical width falls
within the 7.5 inch limit without resampling.

## License

MIT — see `LICENSE`.

## Provenance and known differences

The sweep scripts were extracted from the analysis session that produced the
submitted figures, so they are the code that generated the published numbers
rather than a reimplementation. Two changes were made during verification,
both recorded here because they changed results:

1. `05_sweep_modulation.py` and `06_decompose_cellcycle.py` originally
   re-derived the tuned burst rates `kx` instead of reading them from
   `04_sweep_replication.py`'s output. In `05` the extraction had also
   mis-indexed the conditional-moment tables, so the post-replication buffer
   was `4M` rather than `2M`; this affected only the replicating runs, not the
   static controls. Both scripts now read `data/needsim4_isolated.csv`, and
   both datasets reproduce exactly.
2. `verify.py` was added, and is what caught the above.

`verify.py` currently reports all eight datasets reproducing at zero relative
deviation, except `needsim1.csv` at 7e-12 (sparse-solver round-off).

Regenerated figures are not byte-identical to the submitted PNGs: `bbox_inches
='tight'` crops to the rendered text extent, which shifts the canvas by a few
pixels across matplotlib/font versions. The plotted content is identical, as
it is produced by the same code from bit-identical data.

## Additions in this revision

| script | output |
|---|---|
| `scripts/12_scan_iteron.py` | `data/iteron_coscale.csv` — extrinsic gain for a fixed reservoir and for one that co-scales with the noise source (Fig 2d) |
| `scripts/13_export_age_profiles.py` | `data/fig3a_age_profiles.csv` — the age-resolved curves behind Fig 3a |

`figstyle.py` holds the shared plotting style, including `opaque_legend`, which
draws legends on a solid background so guide lines cannot be drawn across label
text, and `check_overlaps`, which reports text-vs-text collisions at render
time. All three figure scripts now **read** their inputs from `data/` rather
than recomputing them, so `python make_all.py --figures` re-renders the whole
figure set in seconds. The two scripts added here are included in `make_all.py`
and their outputs in `reference/`, so `verify.py` covers them.

## Iteron-regime revision

| script | output |
|---|---|
| `scripts/14_scan_iteron_biological.py` | `data/iteron_biological.csv` — Fig 2d: integer copy number, `M = n*m`, exact CME per copy-number state |
| `scripts/15_figureS1.py` | `figures/figS1_mechanism.png` — S1 Fig, the wide-range comparison and the capacity-ratio collapse |

Fig 2d plots only parameter sets with mean free initiator at or above 2.5
molecules and `k_d >= 3`; below that the exact gain becomes non-monotonic in
`m` through discreteness of the free pool, and `10_figure2.py` asserts
monotonicity of every curve it draws. `reported_values.py` audits the same
selection, including that assertion, so the exclusion stated in the caption is
checked rather than asserted.
