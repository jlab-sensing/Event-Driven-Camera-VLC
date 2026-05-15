# Local Metavision SDK Reference

This folder is a local searchable reference for the public Prophesee Metavision SDK documentation, focused on the EVK4 camera and this repo's visible light communication experiments.

Generated: 2026-05-14  
Starting URL: https://docs.prophesee.ai/stable/index.html

This is a working cache, not a rule limiting future internet use. When exact wording, newest API behavior, or version-specific details matter, check the live official docs too.

## What is here

- [url_index.md](url_index.md): official URLs grouped by topic.
- [evk4_notes.md](evk4_notes.md): project-focused notes for EVK4 VLC work.
- [downloaded_pages/](downloaded_pages/): readable local markdown notes from relevant public documentation pages.
- [manuals/](manuals/): local hardware manuals, including the EVK4 HD camera manual.

## Useful searches

Run these from the repo root:

```powershell
rg -n "EVK4|IMX636|event camera|Metavision" docs/metavision
rg -n "bias_diff|bias_fo|bias_hpf|bias_refr|bias_diff_on|bias_diff_off" docs/metavision
rg -n "RAW|DAT|HDF5|EVT2|EVT3|file format|recording" docs/metavision
rg -n "EventsIterator|RawReader|Camera|DeviceDiscovery|CameraStreamSlicer" docs/metavision
rg -n "ROI|RONI|pixel selection|event rate|LED|VLC" docs/metavision
```

## Fast Map

- Event-camera concepts: [downloaded_pages/01_event_based_concepts.md](downloaded_pages/01_event_based_concepts.md)
- EVK4 and IMX636 hardware: [downloaded_pages/02_evk4_and_imx636.md](downloaded_pages/02_evk4_and_imx636.md)
- EVK4 HD camera manual: [manuals/EVK4_HD_Prophesee_Evaluation_Kit_Camera_Manual.pdf](manuals/EVK4_HD_Prophesee_Evaluation_Kit_Camera_Manual.pdf)
- SDK setup and modules: [downloaded_pages/03_sdk_basics_installation_modules.md](downloaded_pages/03_sdk_basics_installation_modules.md)
- Python API overview: [downloaded_pages/04_python_api_overview.md](downloaded_pages/04_python_api_overview.md)
- Open camera and read events: [downloaded_pages/05_open_camera_read_events.md](downloaded_pages/05_open_camera_read_events.md)
- Record RAW files: [downloaded_pages/06_recording_raw_files.md](downloaded_pages/06_recording_raw_files.md)
- Biases: [downloaded_pages/07_biases.md](downloaded_pages/07_biases.md)
- ROI/RONI and pixel selection: [downloaded_pages/08_roi_pixel_selection.md](downloaded_pages/08_roi_pixel_selection.md)
- Camera settings and filters: [downloaded_pages/09_camera_settings_filters.md](downloaded_pages/09_camera_settings_filters.md)
- File formats and encodings: [downloaded_pages/10_file_formats_encoding.md](downloaded_pages/10_file_formats_encoding.md)
- Metavision Studio and CLI tools: [downloaded_pages/11_studio_and_tools.md](downloaded_pages/11_studio_and_tools.md)
- Training links: [downloaded_pages/12_training_links.md](downloaded_pages/12_training_links.md)
