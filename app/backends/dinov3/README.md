# Vendored DINOv3 inference code

The DINOv3 ViT backbone used by the Kaloscope 3.0 backend (`app/backends/kaloscope.py`).

Taken from the `v3/` package of
[DraconicDragon/Kaloscope-artist-style-classifier](https://huggingface.co/spaces/DraconicDragon/Kaloscope-artist-style-classifier),
which vendors it from [spawner1145/comfyui-kaloscope](https://github.com/spawner1145/comfyui-kaloscope/tree/1f1f12d4103fd10c6f890ecf197a0e280d1979d5/kaloscope_dinov3)
at commit `1f1f12d`. Only `hub`, `layers`, `models` and `utils` are kept, and the
`from v3...` imports were made relative. Nothing else was changed.

Keep `LICENSE.md` with these files: the DINOv3 License applies to this code and to the Kaloscope 3.0 weights.
