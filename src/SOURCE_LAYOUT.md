# Frozen source layout

The original frozen Python modules are deliberately retained as a flat compatibility tree so imports and numerical behavior are not changed during organization. The requested thematic folders (`common`, `models`, `policies`, `simulation`, `optimization`, `evaluation`, `statistics`, and phase folders) are reserved boundaries. `phase0_engine.py` is the FAVE reference implementation; no duplicate FAVE/LRBE implementation was introduced by this organization.
