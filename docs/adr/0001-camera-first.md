# Camera first; sensor hardware only if photos fall short

The project classifies a coffee's origin from phone photos alone. The multi-sensor rig researched at the start (NIR spectral sensors, a gas sensor, an RF dielectric cell on two ESP32-C5 boards) is a contingency, built only if camera-only results prove insufficient, so no firmware, toolchain or hardware devcontainer is set up ahead of that. The same research explains why photos are hard: bean appearance follows altitude, variety and grade more than country, and the strong origin signals (isotope ratios, trace elements) need lab equipment.

Origin: owner decision, 2026-07-30; research in `hardware/Computer vision models for coffee bean origin classification - Claude.pdf`.
