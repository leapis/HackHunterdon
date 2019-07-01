FROM python:3

WORKDIR /main

COPY . /main

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 80

ENV NAME World

CMD ["python", "main.py"]
