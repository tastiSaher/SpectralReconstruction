# Spectral Reconstruction

This repository contains the code used to participate at the [CVPR 2018 challenge on spectral reconstruction](http://icvl.cs.bgu.ac.il/ntire-2018/).
The basic task is to reconstruct 31-channel multispectral images from RGB-images.

Next to the source files, this repository also provides the zip-file "reproduce.zip". 
The executable inside, "reproduce.exe", runs out of the box under Windows (tested for Windows 10 x64) and is capable of performing the task of spectral reconstruction using either of the two models trained for the challenge.
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
The executable was created from the script "reproduce.py" using pyinstaller and is meant to be run under Windows.
It allows to perform the task of spectral reconstruction using either of the two models specifically trained for the two respective challenge tracks on any image.
This is the most simple way to reproduce our results or apply the trained models to other images.
For the sake of simplicity and to limit possible requirements, all computations are performed on the cpu only when using the executable. 
Therefore, the reconstruction may take some time, e.g. the 5 images of the challenge may take up to 30min to be processed.
In order to take advantage of GPU support, take a look at the script "reproduce.py" instead, in which case the processing time for the same 5 images reduces to a couple of seconds.

During execution, the spectral reconstruction of the input images will be initially stored as .npy files. 
After all images have been processed, the corresponding files using hdf5 storage will be created, which are compatible to the challenge data format. Although the created files using hdf5 storage might suggest that thery are matlab files, i.e. end with ".mat", they are not  since they lack the necessary header. You may treat them as ".h5" files.

### How to reproduce our final results for dummies
0. Make sure your operating system is Windows 10 (64bit).
1. Download the file "reproduce.zip" and extract all its contents to a folder of your choice, "root_dir". 
2. Copy the images from the track "Clean" you would like to reconstruct into the folder "root_dir/images_clean/".
3. Copy the images from the track "RealWorld" you would like to reconstruct into the folder "root_dir/images_real/".

#### Track Clean
If you did not change the file "root_dir/config.ini", simply run the executable "root_dir/reproduce.exe".
The reconstructed images will be stored inside "root_dir/reconstruction/" when done.

#### Track RealWorld
4. Open the file "root_dir/config.ini".
5. Change the third line to
```
track = RealWorld
```
6. Change the fifth line to
```
path2images = ./images_real/
```
7. Save and close the file "root_dir/config.ini".
8. If the folder "root_dir/reconstruction/" exists, make sure it is empty.
9. Run the executable "root_dir/reproduce.exe" and wait.
10. The reconstructed images will be stored inside "root_dir/reconstruction/".

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
