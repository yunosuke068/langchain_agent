FROM python:3.11
USER root

RUN apt-get update
RUN apt-get -y install locales && \
  localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
# ENV TZ JST-9
ENV TERM xterm
ENV TZ Asia/Tokyo


COPY . /app/personal_agent

WORKDIR /app/personal_agent

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt --no-cache-dir

# アプリ起動
CMD ["streamlit", "run", "discussion_agent_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
