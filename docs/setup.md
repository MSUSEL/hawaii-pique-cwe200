### Deployment
Here is how you can get it to work via ssh though your local terminal.

1. `ssh -L 4200:localhost:4200 -L 5400:localhost:5400 exouser@<ServerIP>`
2. `git clone https://github.com/MSUSEL/hawaii-pique-cwe200.git` 
3. `cd` into root
4. If labeling - `git checkout labeling-branch`
5. `docker build -t cwe-200 --build-arg JAVA_VERSION=<VERSION> .`
6. `docker run -p 4200:4200 -p 5400:5400 cwe-200`
7. Navigate to `http://localhost:4200/` on your local machine 

### Development Using VS Code
Here is how you can set up the development environment to use the GPUs in the cloud, but develop locally. 

1. Download [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Download [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
3. Navigate to the **Remote Explorer** extension on the left side of VS Code
4. In the drop down on the top left select **Remotes (Tunnels/SSH)**
5. Once it loads there should be a drop down up top, put in your remote information `exouser@<ServerIP>` then password
6. When it connects make sure the terminal is up and check to see that the user is the remote user. If you, you are connected.
7. If you are setting up for the first time, do `git clone https://github.com/MSUSEL/hawaii-pique-cwe200.git`
8. `cd` into root
9. Then in the drop down on the top left select **Dev Containers**
10. Click the **+** and select **Open Current Folder in Container**
11. You will be asked to connect once again using the same credentials. 
12. Once, you are connected it will build the image and launch the container  if this is the first time.  

