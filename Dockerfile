FROM python:slim

LABEL org.opencontainers.image.source=https://github.com/OAR-atmospheric-observations/IRS-Noise-Filter

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install .

CMD [ "python", "./run_irs_nf.py", "--help"]