FROM python:3.6
RUN mkdir /app
RUN mkdir /app/checkpoints
COPY helper.py /app
COPY model_helper_keras.py /app
COPY model_helper_pytorch.py /app
COPY websvc.py /app
COPY requirements.txt /app
COPY cat_to_name.json /app
COPY checkpoints/cp_densenet201_e3_lr0.05.pth /app/checkpoints
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "websvc.py"]