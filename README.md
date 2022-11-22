# Pure tesseract branch
[Go to main branch](https://github.com/kundestyrt-gruppe-5-2022/passport-mrz-reader)

This branch contains an implementation of OCR using only Tesseract and image manipulation libraries such as OpenCV.

## How to run the code

### Install Tesseract
To be able to run the code, Tesseract must be installed. See https://github.com/tesseract-ocr/tessdoc/blob/main/Downloads.md for how to download Teserract. 
Tesseract should be added to the PATH variable.

### Move traineddata file
Then [this traineddata file](model/tesseract/mrz.traineddata) should be moved to 
the tessdata folder where your Tesseract program is stored. The file is collected from https://github.com/DoubangoTelecom/tesseractMRZ and is used according to the following [license](TRAINEDDATA_LICENSE).

### Install python packages
Create virtual environment (first install)
```sh
python -m venv venv
```
Activate virtual environment (every time)
```sh
source venv/bin/activate # Unix-like
venv/Scripts/activate # Windows
```
Set-up (first install)
```sh
pip install -e .
```
Install dependencies
```sh
pip install -r requirements.txt # After remote dependency changes
```
Save your dependency changes
```sh
pip freeze > requirements.txt # After local dependency changes
```
Run main method of a specific file (example):
```sh
python -m passport_mrz_reader.pure_tesseract.tesseract_predict
```
Enable Jupyter widgets
```sh
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```
Open Jupyter notebook (VSCode should also work)
```sh
jupyter notebook
```

Deactivate virtual environment (if desired)
```sh
deactivate
```

### Run tests

Tests can be run using the following command:

`python -m unittest discover -s passport_mrz_reader/tests`

## File structure

[data](data):
- Contains the labeled dataset and MRZ images

[model](model):
- Contains the different trained models, one folder for every approach
  - [pure tesseract](model/tesseract): Contains a .traineddata file which should be moved to the folder where Tesseract is installed.

[src](src):
- [common](passport_mrz_reader/common):
  - Contains functionality that is shared across all implementations
- [pure tesseract](passport_mrz_reader/pure_tesseract): Contains an implementation using only tesseract and image processing libraries such as Open CV to perform OCR on passport MRZ.
