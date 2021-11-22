# Image Background Removal Service

This service is based on the pre-trained [MODNet](https://github.com/ZHKKKe/MODNet) model.

The folder `model/pretrained` contains the official pre-trained models of MODNet. You can download them from this [link](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing).

## Setup

    conda create --name ibrs python=3.9
    conda activate ibrs
    conda install --yes --file requirements.txt

## Test

    python -c "import inference; inference.remove_background('images/lea.jpg')"

