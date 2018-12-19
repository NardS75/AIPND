FROM python:3.6
RUN mkdir /app
RUN mkdir /app/checkpoints
COPY helper.py /app
COPY model_helper_keras.py /app
COPY model_helper_pytorch.py /app
COPY websvc.py /app
COPY requirements.txt /app
COPY cat_to_name.json /app
COPY checkpoints/cp_vgg16_e10_lr0.05.h5 /app/checkpoints
COPY checkpoints/cp_vgg16_e10_lr0.05.json /app/checkpoints
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "websvc.py"]