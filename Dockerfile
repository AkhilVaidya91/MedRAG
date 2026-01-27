FROM python:3.13

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY ./requirements.txt requirements.txt
# COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app
RUN chown -R user:user /app
# COPY --chown=user . /app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]