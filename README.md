# Containerized TensorFlow GPU development environment

A containerized TensorFlow AI/ML development environment with NVIDIA GPU support. This repository provides a consistent development environment with wide GPU compatibility using VS Code Dev Containers, eliminating the need for manual environment setup. It provides the following:

- CUDA 12.5
- cuDNN 9.1
- Python 3.10
- TensorFlow 2.16
- Keras 3.3
- NumPy 1.24
- Pandas 2.2
- Scikit-learn 1.4
- SciPy 1.13
- Matplotlib 3.10
- jupyterlab 2.3

And much more - the underlying docker container is a lightly customized version of NVIDIA's official TensorFlow 24.06 container. For full details see:

- George's TensorFlow 2.16 container on DockerHub: [gperdrizet/tensorflow-gpu](https://hub.docker.com/r/gperdrizet/tensorflow-gpu)
- NVIDIA's [TensorFlow Release 24.06](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-24-06.html)

The container needs Docker and a compatible NVIDIA driver installed to run. It should support NVIDIA GPUs back to Pascal on both Windows and Linux (MacOS users should be able to run it without GPU support - see below). 

## 1. Prerequisites

Before you can use this development container, you need to have:

1. **A Pascal or later NVIDIA GPU**
2. **NVIDIA driver >=545** (slightly older drivers may work - try it before you make major changes to your host system)
3. **Docker** (see platform-specific instructions below)
4. **Visual Studio Code** - [Download here](https://code.visualstudio.com)
5. **Dev Containers extension** - Install from the VS Code Extensions marketplace (`ms-vscode-remote.remote-containers`)

---

## 2. Docker installation instructions

### 2.1. Windows

Full installation documentation here: [Install Docker Desktop on Windows](https://docs.docker.com/desktop/setup/install/windows-install)

1. **Check system requirements**
   - Windows 10 64-bit: Pro, Enterprise, or Education (Build 19041 or higher), or Windows 11
   - Windows 11 64-bit: Enterprise, Pro, or Education version 23H2 (build 22631) or higher
   - WSL 2.1.5 or later
   - 64-bit processor with Second Level Address Translation (SLAT)
   - 4GB system RAM
   - Enable hardware virtualization in BIOS/UEFI. For more information, see Virtualization.

2. **Install or update WSL**

    Full documentation here: [WSL: Verification and setup](https://docs.docker.com/desktop/setup/install/windows-install#wsl-verification-and-setup)

    - To check your WSL version, open PowerShell as Administrator and run:

    ```powershell
    wsl --version
    ```

    - Then install or update as needed:

    ```powershell
    wsl --install
    ```

    ```powershell
    wsl --update
    ```
   - Restart your computer when prompted

3. **Download and install Docker Desktop**
   - Download Docker Desktop from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
   - Run the installer and follow the prompts
   - Ensure "Use WSL 2 instead of Hyper-V" is selected during installation

4. **Start Docker Desktop**
   - Launch Docker Desktop from the Start menu
   - Wait for Docker to fully start (the whale icon in the system tray will stop animating)

5. **Verify installation**
   - Open a terminal (PowerShell or Command Prompt) and run:
     ```powershell
     docker --version
     ```

    The command should complete without error.

> **Note for GPU support:** To use GPU acceleration on Windows, you'll need an NVIDIA GPU with updated drivers.

---

### 2.2. Linux

For Linux set-up instructions, refer to [Install Docker Desktop on Linux](https://docs.docker.com/desktop/setup/install/linux).

> **Note for GPU support:** To use GPU acceleration on Linux, you'll need an NVIDIA GPU with updated drivers and the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## 3. Getting started

Once you have a current NVIDIA driver (>=545) and Docker with GPU support. Starting a CUDA/TensorFlow development environment from this repository is easy. Use it like a template to start new projects:

1. **Fork this repository** to create your own copy:
   - Click the **"Fork"** button in the top-right corner of this repository's GitHub page
   - Select your GitHub account as the destination
   - This creates your own copy where you can save your work and push changes

2. **Clone your forked repository:**

   On your local machine run:
   
   ```bash
   git clone https://github.com/<your-username>/tensorflow-GPU.git
   cd tensorflow-GPU
   ```
   > Replace `<your-username>` with your GitHub username

4. **Open the folder in VS Code:**
   ```bash
   code .
   ```

5. **Launch the dev container:**
   - When prompted, click **"Reopen in Container"**
   - Or press `F1` → Type "Dev Containers: Reopen in Container" → Press Enter

6. **Wait for the container to build** (this may take a few minutes on first run)

7. **Start coding!** Open and run `notebooks/test.ipynb` to verify your environment is working.

## 4. Keeping Your Fork Updated

To sync your fork with the latest changes from this original repository:

```bash
# Add the original repo as an upstream remote (only needed once)
git remote add upstream https://github.com/gperdrizet/tensorflow-GPU.git

# Fetch and merge updates
git fetch upstream
git merge upstream/main
```

---

## Troubleshooting

- **Docker not starting:** Ensure virtualization is enabled in your BIOS settings
- **Permission denied errors on Linux:** Make sure you've added your user to the docker group and logged out/in
- **GPU not detected:** Verify NVIDIA drivers and Container Toolkit (if needed) are properly installed
- **Container build fails:** Ensure you have a stable internet connection for downloading the TensorFlow image