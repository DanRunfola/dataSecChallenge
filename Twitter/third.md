This guide provides a basic implementation of Binary Classification Model using PyTorch. Before writing any python script we need to setup a conda Environment and instal all necessary libraries we need.

# Setting Up Enviroment

It is highly recommended that you use conda environments to avoid conflicts with other packages. Assuming you have anaconda in your local computer follow the below steps:

Use the following line to create your conda environment and type y when prompted Proceed ([y]/n)?

```sh
conda create -n [ENVNAME]
```
(note: in the place of [ENVNAME] you guys can give whatever the name you want eg: DLEnv)

Use the following line to activate your new environment:
```sh
conda activate [ENVNAME]
```
Use the following line to deactivate your environment:
```sh
conda deactivate
```

# Installing Libraries

We have to activate conda environment before installing any libraries. There are bunch of libraries we need to install like pandas,numpy,sklearn,nltk, pytorch and transformers.

Once you've activated your environment, you can install any packages using standard commands:
```sh
#For pandas,numpy and sklearn:
conda install pandas
conda install numpy
conda install scikit-learn

#For nltk:
conda install -c anaconda nltk

#For pytorch:
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

#For transformers:
conda install -c huggingface transformers

```

If you are not using any conda environment then you can install any packages you need for a program using standard pip install commands:
```sh
pip install pandas
pip install numpy
pip install -U scikit-learn
pip install -U nltk
pip install transformers
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```


# The Python Script
 
