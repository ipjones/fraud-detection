# fraud-detection
Fraud Detection Exercise

# Setup
The exercise was performed in Python 3.12. An exhaustive list of the python modules used in the project can be found in requirements.txt. In addition to these modules, you will need OpenMP and Docker.

`brew install libmp`

`brew install docker`

# Design Choices
In the interest of productionization and scalability, I have chosen to implement model training as two separate "packages". The first, model training, takes care of preparing the training and validation data as well as training the model. In this case, I have used a short included data splitting script to ensure logical isolation between the training/validation data and the test data.

The second "package" is model inference. This package handles loading a pre-trained model and running inference against a separate set of data, in this case the pre-defined test set. This separation of training and inference concerns ensures that inference can be performed quickly.

In addition to these two "packages", I have included a Dockerfile to package all of the required code into a docker image. Docker containers are an excellent example of containerization, which allows model inference to be scaled up on demand.

In this particular case, the data is pre-loaded onto the Docker container, which is not practical if the model is to be used for on-demand inference. In such a case, the inference container can implement a listener that services inference requests from the model as they are needed.

# Running the project
To run the project, install the requirements and OpenMP. Uncompress the training data. Afterwards, the project can be run locally with:

`python train_and_infer.py`