# Evidence-integrated refinement report

The pre-refinement gate first blocked manuscript editing because the controlled
mechanism runner used branch-dependent training seeds and recorded declarative
isolation flags. The repair was validated in a separate output block: all 320
selection and endpoint rows match the original evidence exactly, while every
branch within a task now shares training randomness and candidate/test-label
perturbation checks pass.

One substantive presentation iteration was then accepted. The internal score
trajectory is 70.15 to 80.15. These values compare manuscript versions under the
PaperOrchestra rubric and are not an estimate of DASFAA acceptance probability.
Iteration 1 was promoted to the canonical final manuscript. It is 16 pages, has
a 193-word abstract under the project tokenizer, compiles without unresolved
references or overfull boxes, and passes all 76 project tests.
