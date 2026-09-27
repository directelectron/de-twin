# Serving the twin to existing software

```
de-twin serve --soap 5002 --deapi 13240
```

One twin, several faces:

| Face | Who talks to it | How |
|---|---|---|
| DE-TEM-Channel SOAP on `:5002` | DE-Server, de_microscope, de_autopilot, de_ground_crew | point the TEM-Channel host at this machine (`DE_TEM_HOST=127.0.0.1`) |
| deapi fake DE-Server on `:13240` | anything using deapi (`Client`) | `DE_CAMERA_HOST=127.0.0.1:13240`, `client.usingMmf = False` |
| shared memory | a real DE-Server | see {doc}`deserver` |

Moving the stage over SOAP changes the images acquired over deapi, exactly as it would on
the real instrument.

## The DE-TEM-Channel face

Every getter, setter, `getProperty` key and `executeBatchCommands` key of DE-TEM-Channel is
answered from the twin's column, in the formats DE-Server's gSOAP client parses. Setters
follow the channel's post-and-poll semantics (operation status while the stage moves), and
the corrector operations are answered when the column has a corrector.

## The deapi face

The deapi face behaves like DE-Server:

- Dark and Gain exposure-mode acquisitions become references, and other acquisitions are
  corrected with them.
- `Simulator Auto References` (default On) provides references on first use; turn it Off to
  exercise reference workflows.
- `Instrument …` properties mirror the column.
- `Simulator Virtual Specimen Temperature (C)` drives the holder, and
  `Simulator Virtual Specimen` switches the specimen preset.

`python -m de_twin.faces.deapi_server <port>` keeps deapi's own fake-server command line
(positional port, "started" banner), so launchers written for deapi's fake server can start
the twin instead.
