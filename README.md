# openvino and fastapi

This repository contains deep learning models optimized for inference using OpenVino and served as REST APIs using FastAPI

### Steps to run the code and API

1. Create a virtual environment 
 * ``` python -m venv <VENV_NAME> ```
2. Update pip and Install requirements (install fastapi and uvicorn)
  * ``` python pip install --upgrade ```
  * ``` pip install requirements.txt ```
3. Locate your .h5 model and run following command that will save your model as saved_model format for tensorflow
  * ``` python h5_to_saved_model.py ```
4. Modify the bees_openvino_fastapi.py as per you requirements and run following command
  * ``` uvicorn bees_openvino_fastapi:app --reload ```
5. Open the link in the browser and start testing your API.

### Todo

- [ ] Work on optimization of open source off-the-shelf object detection models 
- [ ] Write a detailed tutorial 

### In Progress

- [ ] Working on PyTorch and other popular frameworks model optimization. 

### Done ✓

- [x] Trained a Keras like model on Ants vs Bees (dataset available at [this link](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html))
- [x] Converted .h5 model to tensorflow saved_model
- [x] Optimzied model using OpenVino model optimizer to get model.xml
- [x] Developed a FastAPI for serving this model
