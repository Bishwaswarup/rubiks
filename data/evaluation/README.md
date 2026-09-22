# Evaluation image set

This small, manually labelled set records one solved face for each colour of
the stickerless 3x3 cube used during development. It is an **evaluation and
calibration example**, not a training set: the program uses rule-based HSV and
BGR colour measurements and does not train a machine-learning model.

The original photographs included the contributor's face and room background.
Only crops that exclude the face are included here; the originals are not
distributed. They still show hands, so do not describe this set as fully
anonymous without reviewing the applicable consent and privacy requirements.
The photos were taken under mixed indoor fluorescent and daylight illumination
with the development camera. The labels in `labels.csv` are the known uniform
face colours, repeated in row-major order for all nine stickers.

## Scope and limitations

- 6 images; one cube, one camera, one lighting setting.
- Useful as a reproducible smoke-test example and calibration record.
- Not sufficient to claim general colour-recognition accuracy.

For a publication-quality benchmark, extend this set with multiple cube
surfaces (stickers and stickerless), cameras, backgrounds, distances, angles,
and lighting conditions. Record each condition and a verified 3x3 label grid.
