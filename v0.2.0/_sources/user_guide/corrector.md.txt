# Probe and image corrector

```
de-twin serve --soap 5002 --deapi 13240 --corrector probe     # or image | both; --corrector-cold
```

```python
from de_twin import DigitalTwin, ManualClock

twin = DigitalTwin(camera="DESim", corrector="probe", clock=ManualClock())
col = twin.column
col.set_tem_stem(1)                        # the probe corrector measures on the Ronchigram
corr = col.corrector
corr.pi4_angle_mrad()                      # ~25 mrad tuned (uncorrected column: ~6)
r = corr.measure("Standard")               # Zemlin tableau; takes ~9 s on the clock
r.aberrations["B2"], r.sigma_nm["B2"]      # measured value (nm) and 1-sigma confidence
corr.correct("B2"); corr.correct("A2", fraction=0.7)   # imperfect, with coupling
col.state().probe_aberrations              # ground-truth residual, used by the renderers
```

The column can have a CEOS-like probe corrector (CESCOR), an image corrector (CETCOR), or
both. The residual is the native lens (C3 about 1.2 mm, C5 a few mm, small B2/A2/S3/A3;
seeded), plus the corrector's settings, plus drift. Drift is a seeded random walk on the
twin's clock: C1, A1 and B2 drift fastest, C3/S3/A3 slowest. A good tune leaves C1/A1 of
1–2 nm, B2 of 4–10 nm, A2 of 7–20 nm, C3/S3/A3 of a few hundred nm, and C5 uncorrected.

The corrector behaves like the instrument's:

- **Changing the column detunes it.** An HT change detunes it badly; a probe-mode,
  function-mode or TEM/STEM change kicks C1/A1/B2/A2.
- **Measurements see the total aberration**, including the operator's defocus and
  stigmators. Each is a least-squares fit of the defocus and astigmatism induced at each
  tableau tilt, with per-image noise, a tilt-calibration error and a 470 nm range. Fast
  (9 mrad, up to B2), Standard (18 mrad, up to A4) and Enhanced (34 mrad, up to A5) tableaux
  behave accordingly: C5 leaks into a Standard fit, and an out-of-tune column needs a Fast
  tableau before a large one succeeds.
- **Corrections are imperfect**, with gain and rotation errors and coupling into related
  aberrations.
- **Everything takes time on the clock.** A tableau takes 1 s plus 0.4 s per image; a
  correction takes 0.5 s.

Tuning code drives it through the interfaces it would use on a real instrument. On the
DE-TEM-Channel SOAP face these are the `getCorrector*` calls, `getAberration(s)`, the
`corrector*` commands and `get/setCorrectorBeamTilt`, with DE-TEM-Channel's post-and-poll
semantics: commands answer `success` = accepted; `getCorrectorStatus` returns 0 idle,
1 running, 2 done, 3 failed or 6 absent; results come back as `"C1=x,y;A1=x,y;..."` in
metres. `SoapTemChannelClient` has matching `corrector_*` methods, `MirrorColumn` reads a real
channel's corrector into the state, and the deapi face shows `Instrument Corrector …`
read-only properties.
