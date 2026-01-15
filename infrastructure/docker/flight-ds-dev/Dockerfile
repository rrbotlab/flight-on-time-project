FROM python:3.10.12

ADD entry.sh /entry.sh
RUN chmod 755 /entry.sh

RUN git clone https://github.com/dragoscalin33/flight-on-time-ds.git /app
WORKDIR /app/data-science
RUN pip install -r requirements.txt

CMD ["/entry.sh"]

# docker build -t rrbotlab/flight-ds:dev-v1 .

