Code to generate data for the EIGSEP insturment paper [C. H. Bye et al., 2026](https://arxiv.org/abs/2602.02661). Submitted to RASTI.

The data and an archived version of this repository is available at [Zenodo](https://doi.org/10.5281/zenodo.18383267).

If you use this software, please cite it according to the CITATION.cff file or the following bibtex:

```
@misc{bye2026electromagneticallyisolatedglobalsignal,
      title={The Electromagnetically Isolated Global Signal Estimation Platform (EIGSEP)}, 
      author={Christian H. Bye and David R. DeBoer and Matt Dexter and Aaron Ewall-Wice and Adam Fahs and Pranav Karthik and Komal Kaur and Bahram Khalichi and Wei Liu and Raul A. Monsalve and Aaron R. Parsons and Reid Parsons and Richard R. Rodriguez and Richard J. Saeed and Charlie G. Tolley and Dominic Vazquez and Dirk Wright},
      year={2026},
      eprint={2602.02661},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2602.02661}, 
}
```

## Layout

One notebook per figure in the paper. Each is standalone: it loads the NPZ files
from the Zenodo deposit, placed in this directory, and writes its PDF here. No
notebook reads a path outside the repository, and none imports another.

| notebook | figure | PDF |
|---|---|---|
| `beam_comparison.ipynb` | 1 | `beam_comparison.pdf` |
| `s11-plot.ipynb` | 3 | `reflections.pdf` |
| `s11_sim_meas.ipynb` | 6 | `s11_sim_meas.pdf` |
| `beam.ipynb` | 7 | `beam_dBi.pdf` |
| `beam_modulation.ipynb` | 12 | `beam_modulation.pdf` |
| `horizon_perturbations.ipynb` | 13 | `horizon_perturbations_1col.pdf` |
| `horizon_shift.ipynb` | 14 | `horizon_shift.pdf` |
| `rfi_waterfall.ipynb` | 16 | `rfi_waterfall.pdf` |
| `virA_2dfilter.ipynb` | 18 | `point_src.pdf` |

`eigsep_style.py` is the only shared module. It carries the paper's rcParams and
the `figure`/`figsize`/`save` helpers that give every figure the box size the
manuscript expects at `width=\linewidth`. Figures are saved with an explicit
`figsize` and a bare `savefig` — never with `bbox_inches="tight"`, which crops to
the ink and makes the saved box unpredictable.

## Reproducing a figure

Download the deposit, unpack the NPZ files into this directory, and run the
notebook. No further data is needed — every input a notebook reads is either in
the deposit or in this repository.

Beyond `numpy` and `matplotlib`, the notebooks need `healpy` (`beam.ipynb`),
`hera_filters` (`s11-plot.ipynb`, `virA_2dfilter.ipynb`) and `astropy`, `scipy`
and `eigsep_data` (`virA_2dfilter.ipynb`).

## Tests

```
python -m pytest tests/
```

`test_eigsep_style.py` and `test_figure_geometry.py` check the style module and
that each committed PDF still has the box the manuscript expects.
`test_generator_mapping.py` checks which notebook produces which figure from
which data — read from the code, never inferred from filenames.
`test_npz_contract.py` checks that the deposited arrays reproduce the published
curves. The NPZ files are gitignored, so in a clone without the deposit the tests
that need them skip rather than fail.
