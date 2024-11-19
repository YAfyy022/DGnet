# DG-Net

## Training DG-Net
If you want to train the DG-Net, please download the pretrained model of [ResNet-50](https://drive.google.com/open?id=1raU0m3zA52dh5ayQc3kB-7Ddusa0lOT-) and move it to **/pretrained** before run ``python train.py``. You may need to change the configurations in ``config.py`` if your GPU memory is not enough. During training, the log file and checkpoint file will be saved in ``model_path`` directory. 

## Evaluation
If you want to test the DG-Net, just run ``python test.py``. You need to specify the ``model_path`` and  ``dataset_path`` in ``test.py`` for testing.

## Model
We also provide the checkpoint model trained by ourselves, you can download if from Google Drive for [21-Plant](https://drive.google.com/file/d/1V_3EKkQc0vMF4bYXywDrMNI-xCuFSFm0/view?usp=sharing), [18-Plant](https://drive.google.com/file/d/1Uf34JUsEHvxcwLfhpTt3JApV66MFKXJT/view?usp=sharing)  and [FGVC-Aircraft](https://drive.google.com/file/d/1_A5IW4zVHK9iy7UEam2x0JSOGIGQTOHF/view?usp=sharing) . If you test on our provided model, you will get 77.5%, 74.1% and 94.4% test accuracy, respectively.



