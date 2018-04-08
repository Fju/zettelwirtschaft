# Zettelwirtschaft

Real-time YOLOv2 object detector using Tensorflow (in Python) for recognizing parts of a super market receipt.

## Command-Line options

When executing `main.py`, you can use the following command-line options

| option | type | description | default |
|--------|------|-------------|---------|
| `--batch-size` | integer | Size of training batch | `16` |
| `--investigate` | flag <sup>[1](#footnote1)</sup> | Shows GUI to let the user view all entries of the current batch used for training | `False` |
| `--restore` | flag <sup>[1](#footnote1)</sup> | If `True` restores the latest checkpoint (if it exists) and keeps on training | `False` |
| `--no-training` | flag <sup>[1](#footnote1)</sup> | If `True` no training process will be started, instead the latest checkpoint will be validated | `False` |



<a id="footnote1">1</a>: always `False` when not present, if used the option will be `True` respectively

__Example:__ `$ python main.py --batch-size 10 --investigate`