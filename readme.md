# Spectral Reconstruction

This repository contains the code used to participate at the [CVPR 2018 challenge on spectral reconstruction](http://icvl.cs.bgu.ac.il/ntire-2018/).
The basic task is to reconstruct 31-channel multispectral images from RGB-images.

Next to the source files, this repository also provides the zip-file standalone folder "reproduction". 
The executable inside, "reproduce.exe", runs out of the box under Windows (tested for Windows 10 x64) and is capable of performing the task of spectral reconstruction using our models trained for the challenge.
Details on the executable are given in the respective [section](#exeInfo).

The code itself is written in python. 
Deep learning is implemented and applied with the help of pytorch.

## Using the Code
A more detailed documentation is about to follow... .

There are 3 important scripts to be found: 
* reproduce.py allows to load either of our two models pretrained for their respective challenge tracks, i.e. Clean or RealWorld. 
The loaded model is then used to reconstruct every rgb image within a directory of choice.
* train.py allows you to run the training yourself.
* evaluate.py loads a specificied trained model. 
The model is used to reconstruct a set of images. 
If the groundtruth is available, the evaluation as performed in the challenge is performed and respective error metrics stored.

It is important to note that the code was meant to be run under Linux. 
One reason amongst other things is pytorch. 
While there is also a pytorch version available for Windows, which can be found [here](https://github.com/peterjc123/pytorch-scripts), the scripts were never tested in detail under Windows. 
In theory, it should be running just fine.

## <a name="exeInfo"></a> Using the Executable
The executable is created from the script "reproduce.py" and is meant to be run under Windows.
It allows to perform the task of spectral reconstruction using either of the two models specifically trained for the two respective challenge tracks on any image.
This is the most simple way to reproduce our results or apply the pretrained models to other images.
For the sake of simplicity and to limit possible requirements, all computations are performed on the cpu only. 
Therefore, the reconstruction takes some time, e.g. the 5 images of the challenge test set took on our system having 3.4GHz roughly 25min.
In order to take advantage of GPU support, take a look at the script "reproduce.py" instead.

Before you use the executable, it is necessary to specify a path to the folder containing the RGB-images to be processed.

During execution, the spectral reconstruction of the input images will be initially stored as .npy files. 
After all images have been processed, the corresponding files using hdf5 storage will be created, which are compatible to the challenge data format.

### Configuration

The file "config.ini" can be found inside the folder reproduce and allows the user to configure 3 options:
* The option "track" controls which pretrained network is to be loaded and used for the task of spectral reconstruction. 
There are two pretrained networks available in form of "Clean" and "RealWorld". 
Their main difference is the data the underlying network has been trained upon.
These are also the precise models used to generate the results submitted to the challenge.
The default is 'track = RealWorld'.
* The option "path2images" specifies the folder containing the images to be processed. 
It will be searched for .jpg and .png images only. 
All found images will be reconstructed.
The default is 'path2images = ./images_real/'.
* The option "path2store" specifies the folder where the reconstruction will be written to. 
The reconstructed files will have the same name as their corresponding RGB images.
The default is 'path2store = ./reconstruction/'.

## Authors

Tarek Stiebel

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments