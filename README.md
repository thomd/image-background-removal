# Image Background Removal Service

This service is based on the pre-trained [MODNet](https://github.com/ZHKKKe/MODNet) model.

The folder `model/pretrained` contains the official pre-trained models of MODNet. You can download them from this [link](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing).

## Setup

    conda create --name ibrs python=3.9
    conda activate ibrs
    conda install --yes --file requirements.txt

## Test

    python -c "import inference; inference.remove_background('images/lea.jpg')"

## Service Endpoints

Depending on the `Accept` request header, the service returns an PNG image or an DataURI:

    curl -F "file=@image.jpg" -H 'Accept: image/png' 'http://localhost:8000/file'
    curl -F "file=@image.jpg" -H 'Accept: application/json' 'http://localhost:8000/file'

The service does **not** persist images, but does allow to request the ...


## TODO

[ ] rate limiter
[ ] demo webpage
[ ] error handling
