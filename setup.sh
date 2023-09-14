docker build --no-cache -t clothing-computer-vision .
docker run --name clothing-computer-vision clothing-computer-vision:latest 
docker exec clothing-computer-vision /bin/bash
