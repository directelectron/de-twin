# Following real software

Each part of the twin can either simulate its state or mirror real software:

```
de-twin serve --shm --mirror-temchannel 10.0.0.5     # a real microscope (or the Dummy channel) drives the twin
de-twin serve --deapi --holder impulse               # a DENS Impulse holder, or DENS's own simulator, drives the twin
```

- `MirrorColumn` polls a DE-TEM-Channel with the same batch command DE-Server uses and
  forwards writes to it. It also reads a real corrector's last measured aberrations.
- `ImpulseFollower` reads temperature and bias through `impulsePy`
  (`pip install "de-twin[impulse]"`); without impulsePy or a responding holder it falls back
  to the simulated heating chip and says why.

The twin then renders what the real column and holder report.
