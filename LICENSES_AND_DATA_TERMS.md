# License Boundaries and Third-Party Data Terms

## Project-authored software

Unless a file states otherwise, project-authored source code and documentation
are provided under the root [MIT License](LICENSE).

## Third-party software

Python packages used by the project retain their own licenses.  The MIT
License in this repository does not relicense NumPy, pandas, SciPy,
scikit-learn, XGBoost, CuPy, Matplotlib, pytest, or their dependencies.

## Third-party datasets

No license grant for UCI, HMEQ, PPDai, LendingClub, or source-material files is
made by this repository.  Third-party raw and processed datasets are excluded.
Users are responsible for obtaining authorized copies and following the
source-specific terms.

Dataset names, hashes, row counts, schema descriptions, target mappings, and
aggregate experimental results are included for reproducibility and
provenance.  Do not add dataset binaries or CSVs to this repository without a
separate redistribution-rights review.

## Generated research results

The repository includes model-independent aggregate reports and formal
simulation output CSVs derived from the registered inputs.  Before public
release, authors should confirm that the intended venue and each source's
terms permit the planned artifact contents.

## Publication checklist

- Confirm author/copyright ownership for all project code.
- Confirm whether each dataset permits redistribution of raw or processed
  rows; default to source links and hashes when uncertain.
- Remove author-identifying metadata if the target venue uses double-blind
  review.
- Update `CITATION.cff` with the final author list, title, venue, DOI, and
  release version.
- Create an immutable tagged release for the paper artifact.
