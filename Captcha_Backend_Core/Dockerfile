
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
WORKDIR /app
RUN apt-get update
RUN apt-get install libre2-dev git wget vim python3-enchant libenchant-dev -y
RUN python -m pip install --upgrade pip 
# RUN apt-get install libre2-5
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1 -y
RUN apt-get install enchant -y
RUN pip3 install einops
RUN pip3 install gdown
RUN pip3 install matplotlib
RUN pip3 install imgaug
RUN pip3 install PyYAML
RUN pip3 install prefetch_generator
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

####  Adding OCR models ####
RUN mkdir -p /Reaction/vietocr/weights
ADD https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4 /app/vietocr/weights
ADD https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA /app/vietocr/weights

