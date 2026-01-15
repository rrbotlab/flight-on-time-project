FROM eclipse-temurin:21-jdk

ADD entry.sh /entry.sh
RUN chmod 755 /entry.sh

RUN apt update && \
    apt upgrade -y && \
    apt install maven -y && \
    apt install git -y

RUN git clone -b dev https://github.com/rrbotlab/flight-on-time-project.git

WORKDIR /flight-on-time-project/back-end

RUN mvn clean package

CMD ["/entry.sh"]

# docker build -t rrbotlab/flight-back:dev-v3 .

