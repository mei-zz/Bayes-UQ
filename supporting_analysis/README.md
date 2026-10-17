# Supporting analysis sources

This directory contains the archived internal/gain-only diagnostic, no-null,
intervention-scale and internal utility-variance replay sources. These sources
require their recorded saved inputs and workspace dependencies. They are not
presented as a complete training package.

The current MIRAGE xi=0.27 run is reproduced using REVIEW_MINIMAL_REPRO. The
xi=0.25 coarse control is supplied as a frozen incumbent for the paired
comparison; its test rankings are not refitted. Rankings, actions, labels and
nominal cluster intervals for the reproduced run are supplied there.
The earlier MIRAGE gain-only diagnostic in analyze_q05_reliability.py is kept
here for provenance. Its outputs are not reported in the final Online Resource 1
(Tables S1--S12), consistent with the manuscript's Code availability statement.

REVIEW_MINIMAL_REPRO/synthetic_core supplies a runnable cached-feature model
example. REVIEW_MINIMAL_REPRO/run_reproduction.py independently verifies the
new MIRAGE utility decisions and reproduces the evaluation. See that package's
README for exact scope and commands.

`requirements_utility_replay.txt` covers the saved-prediction utility replay
(NumPy and Pandas only). The other archived scripts additionally require SciPy
and scikit-learn; their historical versions are not pinned here because those
scripts cannot run without the absent saved inputs and are included for source
provenance, not as a runnable workflow.
