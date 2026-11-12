# What is simulated

- **Column**: behaves like DE-TEM-Channel's JEOL Dummy.
  - JEOL function modes, TEM/STEM, and magnification and camera-length detent ladders.
  - Stage motion at realistic speeds, with a busy status while moving.
  - Spot size and probe mode set the beam current; intensity sets the illuminated area.
  - Defocus, astigmatism, beam/image shift, beam tilt and diffraction shift.
  - Axial aberrations in CEOS notation (C1 to C5), with an optional probe and/or image
    corrector (see {doc}`corrector`).
  - Beam blank, screen and column valves.
- **Specimen**: every holder and preparation of DE-Server's former VirtualSpecimen,
  deterministic from a seed.
  - Holders: mesh and waffle grids, FIB lift-out lamellae, in-situ MEMS heating chip.
  - Preparations: nanoparticles, proteins in ice, thin films, bulk structures (strain,
    p-n junction, precipitates).
  - Every droplet or grain is a real crystal with a full 3D orientation (orix): random for
    particles, a fibre texture for thin films, near-single-crystal for lamellae.
  - In-situ crystallisation driven by the holder's temperature history.
- **Image formation**:
  - TEM: wave-optical, with a CTF: Thon rings, astigmatism, and the beam-tilt × defocus shift.
  - SAED, NBD and CBED: kinematic diffraction from diffsims structure factors. Stage tilt
    moves grains through Bragg conditions in every pattern and in bright-field contrast,
    and disks have the column's convergence angle.
  - STEM: coherent 4D-STEM (probe wavefunction × transmission function, partial coherence)
    suitable for ptychography and tcBF, a fast kinematic model for large patterns, and
    virtual detectors.
  - Dose is physical: electrons per pixel per second from beam current and illuminated area.
  - Ground truth for scoring automation: noiseless flux, orientation maps (`twin.ground_truth`,
    an orix `CrystalMap`), and the ptychography object and probe.
- **Detector**: every DE camera model (DE16/64, Celeritas, Apollo, …).
  - Poisson electrons, charge spread and gain non-uniformity.
  - Dark offset, fixed pattern, read noise and dark current.
  - Hot/dead pixels and bad columns, saturation, binning and counting.
  - Dark and gain references are therefore necessary, and they work.
- **Holder**: a simulated DENS-style heating chip (optionally biasing), or a follower of a
  real DENS Impulse holder or DENS's own simulator.
