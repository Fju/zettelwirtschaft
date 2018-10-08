# Zettelwirtschaft

A neural network to detect the total price printed on a super-market receipt. The goal is to create an real-time and accurate model that can be integrated into mobile applications.

## Training and validation data

I currently maintain a dataset for training and validation of the model that contains 35 images. I collected these supermarket receipts, took pictures of them and labeled them respectively. I will add more and more labeled receipts in the future since 35 samples is too few for a robust model. When the dataset is big enough I may provide a link to download it.
In the meantime I crop and scale the samples randomly to augment more data.

## Arguments

When executing `main.py`, you can use the following command-line options

| option | type | description | default |
|--------|------|-------------|---------|
| `--list-devices` | flag | Lists available processing units (GPU's, CPU's, etc.). Exits immediately after list is printed. | `False` |
| `--model_name` | string | The model's name needed for restoring and saving checkpoints. | `"default"` |
| `--config` | string | Filepath to the JSON configuration file. If the file doesn't exists under the specified path, a blank configuration file will be generated automatically | `"config.json"` |
| `--continue` | flag | If present it will try to restore the latest checkpoint (`model_name` determines which model will be restored) | `False` |