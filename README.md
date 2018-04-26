# Zettelwirtschaft

Real-time YOLOv2 object detector using Tensorflow (in Python) for recognizing parts of a super market receipt.

## Dataset

I currently maintain a dataset for training and validation of the model that contains 35 images. I collected these supermarket receipts, took pictures of them and labeled them respectively. Due to privacy I can't publish the dataset here. If you want to evaluate your own model, you have to collect and label your dataset by yourself.
You can, however, contribute to this project by sending labeled examples to me. The bigger the dataset the model is trained with, the better it will perform!

Here's how I formatted my label list:
```
kb_0001;424,47,136,48;659,270,108,59;573,700,108,35;lidl;0.99;15.11.17
```
Attributes are separated by semi-colons (`;`) and represent following properties of the receipt (left to right):

1. __filename__: the filename of the image (`.png` format)
2. __logo position__: coordinates and size (`x,y,width,height`) of the bounding box where the logo is located in the image
3. __price position__: coordinates and size (`x,y,width,height`) of the bounding box where the price is located in the image
4. __date position__: coordinates and size (`x,y,width,height`) of the bounding box where the date is located in the image
5. __supermarket__: name of the supermarket according to the logo
6. __price__: total of the receipt, decimal number ignoring currency (for now)
7. __date__: date when the receipt was printed (european format: `DD.MM.YYYY`)


## Command-Line options

When executing `main.py`, you can use the following command-line options

| option | type | description | default |
|--------|------|-------------|---------|
| `--batch-size` | integer | Size of training batch | `16` |
| `--summarize` | flag | If `True` writes a summary into the summary directory. Summaries can be reviewed using `tensorboard`. This option may get replaced in the future by an option where the frequency of logs can be specified. |
| `--investigate` | flag  | Shows GUI to let the user view all entries of the current batch used for training. | `False` |
| `--restore` | flag  | If `True` restores the latest checkpoint (if it exists) and keeps on training. | `False` |
| `--no-training` | flag | If `True` no training process will be started, instead the latest checkpoint will be validated. | `False` |

Note: flag is a boolean, that is always `False` when not present as an argument, if present the option will be `True` respectively.


__Example:__ 
```
$ python main.py --batch-size 10 --investigate
```
