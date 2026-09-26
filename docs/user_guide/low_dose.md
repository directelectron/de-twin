# Low dose

```python
from de_twin import DigitalTwin, ManualClock

twin = DigitalTwin("Apoferritin in ice", camera="DESim", clock=ManualClock())
col = twin.column
col.set("Magnification", 30000)
col.set("LowDose", True)                  # the current state becomes Record
col.set("LowDoseAreas", {"View": {"magnification": 3000, "defocus_offset_um": -150.0}})
col.set("LowDoseArea", "View")            # Search | View | Focus | Record
col.get("LowDoseAreas")                   # every area's settings
```

The column has a low-dose mode, as TFS and JEOL columns do. There are four areas: Search,
View, Focus and Record (`Exposure` is accepted for Record). Each one stores:

- magnification;
- spot size;
- intensity;
- defocus offset;
- image shift;
- beam shift.

How the areas behave:

- **Defaults.** Switching low dose on for the first time seeds the areas from the current
  state:
  - Record: the current state.
  - Focus: Record image-shifted by 1.5 units along x (the tilt axis).
  - View: about 8× lower magnification, 200 µm underfocus.
  - Search: about 50× lower magnification, 200 µm underfocus.
- **Switching** stores the live settings into the area you leave, as a real column does when
  you adjust an area while in it. It then applies the area you enter. A switch is
  all-or-nothing: if a setting is refused, the column is left as it was.
- **Defocus** is kept relative to Record's focus. Refocusing in Record carries View's large
  offset along with it.
- **`LowDoseAreas`** is validated before anything changes. Editing the area you are in keeps
  your live changes.
- **Diffraction.** In diffraction the magnification is left alone, so areas can be switched
  there too.
- **Switching off** returns to Record. Switching back on later keeps what you set in the
  meantime, as Record.

On a realistic column (`OpticsConfig.realistic`) the areas do not line up by themselves. Each
magnification has its own rotation, pixel size, image-shift matrix and offset, and View's
high defocus changes scale and rotation again. `tests/test_low_dose.py` runs SerialEM's
View-to-Record alignment against the twin, and finds the true area offset to 1.5 px.
