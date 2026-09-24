=============
API reference
=============

The twin
========

.. currentmodule:: de_twin

.. autosummary::
   :toctree: generated
   :nosignatures:

   twin.DigitalTwin
   clock.Clock
   clock.ManualClock
   processing.Processor

Shared state
============

.. autosummary::
   :toctree: generated
   :nosignatures:

   state.MicroscopeState
   state.AcquisitionRequest
   state.ScanRequest
   state.HolderState
   state.FrameMeta
   state.ExposureMode
   state.RenderMode

Column and corrector
====================

.. autosummary::
   :toctree: generated
   :nosignatures:

   column.Column
   column.MirrorColumn
   column.ColumnAdapter
   column.SoapTemChannelClient
   column.corrector.Corrector
   optics.aberrations.Aberrations

Specimen and crystals
=====================

.. autosummary::
   :toctree: generated
   :nosignatures:

   specimen.Specimen
   specimen.SpecimenConfig
   specimen.from_name
   crystal.library.CrystalLibrary

Detector
========

.. autosummary::
   :toctree: generated
   :nosignatures:

   detector.CameraModel
   detector.Detector
   detector.camera

Holders
=======

.. autosummary::
   :toctree: generated
   :nosignatures:

   holder.connect_holder
   holder.SimHeatingHolder
   holder.ImpulseFollower

Optics and rendering
====================

.. autosummary::
   :toctree: generated
   :nosignatures:

   optics.derive_optics
   optics.OpticsConfig
   optics.Calibration
   render.Renderer
   render.RenderConfig

Faces (serving the twin)
========================

.. autosummary::
   :toctree: generated
   :nosignatures:

   faces.temchannel_soap.TemChannelServer
   faces.deapi_server.TwinDeapiServer
   faces.shm_face.ShmFace
   transport.shm.FrameProducer
   transport.shm.FrameConsumer
