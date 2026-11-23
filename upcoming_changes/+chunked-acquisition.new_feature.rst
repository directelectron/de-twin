The deapi face takes in an acquisition's frames a chunk at a time
(``DigitalTwin.frame_chunks``): its virtual detectors are one matrix product per chunk
(the masks as a matrix times the stack) and its sum one reduction, instead of Python
float64 sums per frame. A recording replays several times faster through it.
