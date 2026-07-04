## Summary

- 

## Evidence Impact

- [ ] No calibration or evaluation behavior changed.
- [ ] Metrics, factors, solvers, adapters, or reports changed and tests were updated.
- [ ] Schema-visible fields changed and committed schemas were regenerated with:

```bash
slac schema all --output-dir schemas
```

## Checks

- [ ] `ruff check .`
- [ ] `mypy src/slac`
- [ ] `pytest`

## Notes

- Dataset-provided calibration is treated as reference evidence, not ground truth.
- ROS and GPL integrations must stay outside slac core.
