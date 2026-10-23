# GF180 liberty provenance

The fourth library is the GlobalFoundries GF180MCU 7t5v0 standard-cell library,
TT corner, taken **from the OpenROAD-flow-scripts repository at the same commit
this project pins** (3451ca1), so no separate PDK download is required:

    path in repo : flow/platforms/gf180/lib/gf180mcu_fd_sc_mcu7t5v0__tt_025C_1v80.lib.gz
    blob         : e3d1e682284795f36c2130a8a9573958577b2d46
    fetched with : git cat-file blob e3d1e68... > <dst>.lib.gz   (SSH remote, ~3 MB)
    unpacked size: 20,410,760 bytes

Cells seen in the mapped netlists are `gf180mcu_fd_sc_mcu7t5v0__*`
(e.g. `__nand2_1`, `__aoi21_1`, `__dffq_1`); the only non-mapped cell type after
`abc -liberty` is Yosys's `$scopeinfo` metadata, exactly as for the other three
libraries (558 occurrences on AES-T100 for every library).

Synthesis recipe is identical to `src/batch_multilib.py` with
`TECHS["gf180"]`; the gate-level re-mapping uses the same library-independent
functional models (`src/e3x/stub_models.v`) built from the Trust-Hub
Nangate-original cells, so no new cell models were needed for the new library.
