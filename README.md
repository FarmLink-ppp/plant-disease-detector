# Plant Disease Detection

This repository contains a machine learning model for detecting plant diseases using images. The model is trained on a dataset of plant images and can classify them into different disease categories.

## Dataset

The dataset used for training the model is the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## Setup without Docker

1. Clone the repository:

   ```bash
   git clone https://github.com/FarmLink-ppp/plant-disease-detector.git
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Open plant-disease.ipynb in Kaggle or Google Colab, run the code to train the model, and save the model as `plant_disease_model.pth`

6. Move the `plant_disease_model.pth` file to the project root directory.

7. Run the following command to start fastapi:

   ```bash
   uvicorn main:app --reload
   ```

8. Open your browser and go to `http://127.0.0.1:8000/docs` to see the API documentation and test the endpoints.

## Setup with Docker

1. Clone the repository:

   ```bash
   git clone https://github.com/FarmLink-ppp/plant-disease-detector.git
   ```

2. Open plant-disease.ipynb in Kaggle or Google Colab, run the code to train the model, and save the model as `plant_disease_model.pth`

3. Move the `plant_disease_model.pth` file to the project root directory.

4. You have to install nvidia container toolkit to run the docker image with GPU support.

   - For windows: Follow the instructions [here](https://docs.docker.com/desktop/features/gpu/)
   - For linux: Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.htmlhttps://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

5. Build and run the container using Docker Compose:

   ```bash
   docker-compose up --build
   ```

6. Open your browser and go to `http://127.0.0.1:8000/docs` to see the API documentation and test the endpoints.

## Usage

You can use the API to upload an image of a plant leaf and get the predicted disease category. The API accepts a POST request with the image file or a query with image_url and returns the predicted disease category.
