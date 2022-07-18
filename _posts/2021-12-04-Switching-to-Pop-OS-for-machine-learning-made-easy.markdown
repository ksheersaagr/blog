---
layout: post
title: "Switching to Pop! OS for machine learning made easy"  
author: krunal kshirsagar
toc: true
youtubeId: QE0lyWodWdU
---

 Some of the major components of Pop!_OS are written in Rust. Its 21.04 version was released on 29 June 2021 and is based upon Ubuntu 21.04. It included the COSMIC (Computer Operating System Main Interface Components) desktop, based on GNOME, but with a custom dock and shortcut controls. Also, it is said that in the future updates System76 will be ditching GNOME and will introduce a new Desktop Environment all together written in Rust (expect it to be heck of alot faster!) based on Pop! OS instead. Hmm..Going the apple way it seems but remaining open source of course. However it won't be an easy task, wish them luck and I hope it comes out.

References: [[here]](https://www.reddit.com/r/linux_gaming/comments/qo4bdo/pop_os_to_use_it_own_desktop_environment_written/)[[here]](https://www.reddit.com/r/linux/comments/qododc/pop_os_to_build_a_new_independent_desktop/)

# Table of Contents

1. [Setup](#setup)
    - [Download](#download)
    - [Making a bootable USB drive](#bootable-drive)
    - [Installing Pop! OS](#installing-os)
    - [Initial Update & Upgrade](#update-upgrade) 
    - [Enable maximise button](#maximise-button)
    - [Dash to Dock](#dash-dock)
    - [Audio](#audio)
        - [Installing Restricted Formats](#restricted-formats)
        - [Sound Issues](#sound-issues)
    - [Backup & Restore(Not Mandatory)](#backup-restore)
2. [Installing Conda](#installing-conda)
    - [Useful conda commands](#conda-commands)
3. [Installing Julia](#julia-lang)
    - [Installing IJulia kernel](#ijulia-kernel)
    - [Installing CUDA.jl](#cuda-jl)
4. [Installing nodejs(Optional)](#installing-nodejs)
5. [Installing Git](#installing-git)
6. [Installing Pytorch](#installing-pytorch)
    - [Installing Pytorch Geometric](#pytorch-geometric)
7. [Installing Cuda](#installing-cuda)
    - [Installing Cuda toolkit](#installing-cuda-toolkit)
    - [Installing CuDNN](#installing-cudnn)
8. [Installing JAX](#installing-jax)
9. [Installing VS Code](#installing-vs-code)



## 1. <a name="setup">Setup</a>

### <a name="download"> Download </a>
- Download Pop! OS from **[here](https://pop.system76.com/)**. click **Download**, then choose from the current release (default) or if you have Nvidia GPU then please download the **NVIDIA** version(Pop! OS provide Nvidia drivers for GPU support out of the box).
- Download PowerISO from **[here](https://www.poweriso.com/download.php)**.


### <a name="bootable-drive">Making a bootable USB drive </a>
Install the poweriso.exe file. After installing, open the poweriso software then go to **Tools** => **Create bootable USB drive** then hit OK, then **select the source file(Pop! OS image file with .iso extension)**, **select destination USB drive** and hit **Start**. It will take couple of minutes to complete the task.

### <a name="installing-os">Installing Pop! OS</a>
1. Insert the bootable usb into the usb slot and reboot the system.
2. Keep hitting **F10 or F11** key depending on your systems motherboard (mine is of MSI - so it's F11 key) while your system restarts. 
3. In the bootloader menu select your bootable usb drive.
4. Next select the options of your choice - mainly the language, time zone & keyboard layout etc.(better keep it to default)
5. Once you are in the installation window you'll see 2 options:
    1. Clean Install.(**Warning: This will wipe all of your system's data**)
    2. Custom (Advanced).
    choose as per your need, I chose the 1st option to clear everything and install a fresh copy of Pop! OS.
6. Then select the disk drive that you want to install the Pop! OS a fresh and hit **Erase and Install.** 
7. Drive Encryption: select if you want to encrypt your drive. (This will impact the performance of your system) - I chose don't encrypt.
8. And then the usual stuff like select the wifi, time zone, keyboard layout, privacy location settings and connect account etc - choose as you need to and you are done with the installation.

### <a name="update-upgrade">Initial Update & Upgrade</a>
In order to upgrade your system, execute the following commands in the terminal:
```
    sudo apt update
    sudo apt full-upgrade
```

### <a name="maximise-button">Enable maximise button</a>
Since it's a highly customisable OS, some functionality might not be available out of the box. For instance, the maximise button isn't available for you to maximise the window size instead you have to double click on top of the window pane in order to maximise the window size.

Let's add the maximise button with the following command:

`sudo apt install gnome-tweaks`

Now go to applications and select **tweaks**. When it opens click on the **Window Titlebars** tab to the left and then enable **Maximise** toggle option under the **Titlebar Buttons** section.

Check out all the different settings and configuration options to get the most out of your system.

### <a name="dash-dock">Dash to Dock</a>
There are a few gripes that I have but nothing that can't be resolved with some more time under my belt with the new OS. My biggest and probably the most worrisome is the **taskbar**. I don't really like that I can't see all of my running apps on the bar. It makes switching a bit of a pain, although alt+tab isn't the worst in the world, I would like another option if I'm having a derp moment. Please find how to customise the taskbar **[here](https://support.system76.com/articles/dash-to-dock/).**

**Note: once you are done with the installation you should find Dash-to-Dock settings by right-clicking on the `show application` menu.**

### <a name="audio">Audio</a>
#### <a name="restricted-formats">Installing Restricted Formats</a>
Pop! OS comes with few of the open source softwares. However, you will find some codecs or media format missing. Execute following commands in the terminal:

`sudo apt-get install ubuntu-restricted-extras`

Oh! and in the package configuration window use **Tab** button to select OK and Yes options.

#### <a name="sound-issues">Sound Issues</a>
If you come across a sound issue, please run the following commands in the terminal and reboot the system:
```
    1. sudo apt purge timidity-daemon
    2. sudo apt-get install --reinstall alsa-base pulseaudio
    3. sudo alsa force-reload
```

### <a name="backup-restore">Backup & Restore</a>
Install Timeshift by running the following command in the terminal:

`sudo apt install timeshift`

After installing follow the instructions in the video below:

<div class="video">
{% include youtubePlayer.html id=page.youtubeId %}
<div style="text-align:center">Timeshift tutorial.</div>
</div>

and you are done with the initial setup.

## 2. <a name="installing-conda">Installing Conda</a>
1. Download the latest `.sh` file for linux from **[here](https://www.anaconda.com/products/individual)**.

2. Open a new terminal and go to your Downloads folder:

    `cd ~/Downloads`

3. Run the following command in the terminal:

    `bash ~/Downloads/Anaconda3-2021.11-Linux-x86_64.sh`

    **Note: please change the command as per the `.sh` file name.** 

4. Now keep pressing **Enter** till it asks -
    1. Do you accept the license terms? -> Type **yes** and hit **Enter**.
    2. Do you wish the installer to initialize Anaconda3 by running conda init? -> Type yes and hit **Enter**.

5. Then re-open the terminal and enter the following command:

    `source ~/.bashrc`

6. Again open a new terminal and enter:

    `conda -V`

if it returns the conda version then the installation is successful.

### <a name="conda-commands">Useful conda commands</a>
``` 
    1. conda activate
    2. conda init
    3. jupyter notebook 
```

## 3. <a name="julia-lang">Installing Julia</a>
1. Download the `tar.gz` file from **[here](https://julialang.org/downloads/)**.
**Under the `Current stable release:` section, select the `64-bit` version from `Generic Linux on x86`**.

2. Open a new terminal and go to your Downloads folder:

    `cd ~/Downloads`

3. Extract the `.tar.gz`:

    `tar -xvzf julia-1.6.4-linux-x86_64.tar.gz`

4. Copy the extracted folder to `/opt`:

    `sudo cp -r julia-1.6.4 /opt/`

5. Create a symbolic link to julia inside the /usr/local/bin folder:

    `sudo ln -s /opt/julia-1.6.4/bin/julia /usr/local/bin/julia`

6. Re-open the terminal and execute the command:

    `julia`

### <a name="ijulia-kernel">Installing IJulia kernel</a>
In order to run Julia in Jupyter notebook, you need to install the IJulia kernel by entering the following command in the Julia REPL terminal:
``` 
    1. using Pkg
    2. Pkg.add("IJulia")
```
Once installed, re-open the terminal and open jupyter notebook by entering `jupyter notebook` command, now in the browser, click **new** option and you'll see the Julia kernel mentioned in the dropdown menu, hit the Julia kernel option and you will be redirected to a new notebook with IJulia kernel. Now you can write Julia programs in Jupyter notebooks.

### <a name="cuda-jl">Installing CUDA.jl</a>
CUDA.jl package is the main programming interface for working with NVIDIA CUDA GPUs using Julia. Install the package by entering the following command in the Julia REPL terminal:
```
    1. using Pkg
    2. Pkg.add("CUDA")
```


## 4. <a name="installing-nodejs">Installing nodejs</a>
Install nodejs by executing following commands in the terminal:
```
    1. sudo apt update
    2. sudo apt install nodejs npm
```
To check whether nodejs is properly installed or not, launch REPL terrminal by pressing **`Ctrl+C` twice** & execute the commands given below:
```
    1. nodejs
    2. node -V && npm -V
```
If it returns version then nodejs is properly installed.


## 5. <a name="installing-git">Installing Git</a>
Run the following commands in the terminal and you are good to go:
```
    1. sudo apt-get install git-all
    2. git --version
```

## 6. <a name="installing-pytorch">Installing Pytorch</a>
Head out to **[pytorch.org](https://pytorch.org)** and you'll see a command customised according to your environment under the **Run this Command** section which should look like this:

**`conda install pytorch torchvision cudatoolkit=11.3 -c pytorch`**

or you can customised it according to your needs.

### <a name="pytorch-geometric">Installing Pytorch Geometric</a>
Run the following command to install:

**`conda install pyg -c pyg -c conda-forge`**

## 7. <a name="installing-cuda">Installing Cuda</a>

<!--
Install the Cuda toolkit by entering the following command in the terminal:

**`sudo apt install nvidia-cuda-toolkit`**

after installing check if it's installed properly by entering the following command in the terminal:

```
1. nvidia-smi
2. nvcc --version
```
if it returns the version then it's properly installed.

- check if the pytorch recognises the cuda is installed or not and is available by running the following command in the terminal:

    ```
    1. python
    2. import pytorch 
    3. torch.cuda.is_available()
    ```
if it returns **`True`** then it's installed properly.
-->
### <a name="installing-cuda-toolkit">Installing Cuda toolkit</a>

1. Click **[here](https://developer.nvidia.com/cuda-downloads)** to redirect to the official CUDA Toolkit website.

2. Now under **Select Target Platform**, choose **Linux** then select **x86_64** as Architecture, then select **Ubuntu** as Distribution, select **20.04** as Version, lastly select **runfile[local]** as Installer type.

3. Scroll to the **Base Installler** section and under **Installation Instructions** copy the 1st command that will look like this:

    **`wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda_11.5.1_495.29.05_linux.run`**

    Now paste & run the above command in the terminal.

4. Now run the **`ls`** command in the terminal, you will see **`cuda_11.5.1_495.29.05_linux.run`** mentioned in the output, copy the name of the file. After that run the following command in the terminal to give execute permission:

    **`sudo chmod +x cuda_11.5.1_495.29.05_linux.run`**

5. Then copy the 2nd command from the **Installation Instructions**, paste & run it in the terminal:

    **`sudo sh cuda_11.5.1_495.29.05_linux.run`**

    After that select **Continue** and hit **Enter**. Then write **accept** and hit **Enter** again.

    Next you'll be in the CUDA Installer window, in that select **CUDA Toolkit 11.5** and deselect every other options and select **Install**. You'll see the summary after successful installation **(It will be mentioned that `Toolkit:  Installed in /usr/local/cuda-11.5/`)**.

6. Open the .bashrc by typing the following command in the terminal:

    **`nano ~/.bashrc`**

    - Set the PATH by copying the following line to the file:

    ```
    export PATH=/usr/local/cuda-11.5/bin${PATH:+:${PATH}}

    export LD_LIBRARY_PATH=/usr/local/cuda-11.5/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    export CUDADIR=/usr/local/cuda-11.5
    ```
    use **`Ctrl+Alt+S`** to save, then use **`Ctrl+X`** to Exit, then it will ask to save modified buffer? - type **`Y` for yes** and then it'll ask to save the file hit **`Enter`**. 

    After that paste following command in the terminal to force load and re-read the .bashrc file:

    **`source ~/.bashrc`**

    You are done with CUDA Toolkit Installation.

7. Some useful stuff about Cuda toolkit:

    ```
    1. https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

    2. https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi

    3. https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

    4. https://askubuntu.com/questions/1211919/error-installing-cuda-toolkit-existing-package-manager-installation-of-the-driv

    ```


### <a name="installing-cudnn">Installing CuDNN</a>

1. Click **[here](https://developer.nvidia.com/rdp/cudnn-archive)** to redirect to the official cuDNN archive website.

2. Select the latest cuDNN installation. In my case I chose the **cuDNN v8.3.0(November 3rd, 2021), for CUDA 11.5**
It will ask you to sign up to Nvidia developers program, sign up using any dummy email address then click the link in your email address to verify the sign up and then submit the form. After that the file will be downloaded.

3. Navigate to where you have downloaded the cudnn tar file. In my case I've downloaded it to the Downlods directory. So change directory to Downloads by **`cd Downloads/`** after that unzip the CuDNN package .tgz file using the following command in the terminal:

    **`tar -xvf cudnn-11.5-linux-x64-v8.3.0.98.tgz`**

4. Copy the following files into the CUDA toolkit directory by entering the following command in the terminal:

    ```
    1. sudo cp cuda/include/cudnn*.h /usr/local/cuda/include

    2. sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64

    3. sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    ```
5. Reboot the system and enter the following command in the terminal:

    ```
    1. nvidia-smi

    2. nvcc --version
    ```
6. Some useful stuffs about CudNN:
    - Check CudNN by running the following command:

        `cat ${CUDNN_H_PATH} | grep CUDNN_MAJOR -A 2`
                        
                        OR

        `sudo cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A2`

                        OR

        `CUDNN_H_PATH=$(whereis cudnn.h)`
        
    
    - More References:

        ```

        1. https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux

        2. https://support.system76.com/articles/cuda/

        3. https://newbedev.com/how-to-verify-cudnn-installation

        ```
    
## 8. <a name="installing-jax">Installing JAX</a>

```
1. pip install --upgrade pip

2. pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

```

**Warning: This will install jax compatible with latest version of CUDA & CuDNN. As I'm writing this post, the latest JAX version is compatible with CUDA 11 & CuDNN 8.2 or newer respectively.**

- Check if JAX is installed properly with GPU support by running the following commands in the terminal:

    ```
    1. python

    2. from jax.lib import xla_bridge

    3. print(xla_bridge.get_backend().platform)

    ```

    **If installed properly with GPU support, it will print `GPU` in the terminal.**


- Reference:

    ```
    
    https://github.com/google/jax#installation
    ```

## 9. <a name="installing-vs-code">Installing VS Code</a>

1. Go to Pop! shop from the application menu and search for the VS Code

2. Select VS Code and hit install.

Note that in VS Code, search for any extensions you require by pressing **`Ctrl+Shift+X`**.

**Note: New stuff will be added to this blog post if I discover some useful things related to Pop! OS for machine learning.**

<!-- 
<div class="imgcap">
<img src="{{ site.baseurl }}/images/2019-12-01-facial-keypoint-detection-using-cnn-pytorch-dots-represents-keypoints.png">
<div class="thecap">Examples of RL in the wild. <b>From left to right</b>: Deep Q Learning network playing ATARI, AlphaGo, Berkeley robot stacking Legos, physically-simulated quadruped leaping over terrain.</div>
</div>
-->