#python image
FROM python:3.11-slim

#running the workind dir
WORKDIR /app

# main
COPY main.py .
COPY Diabaties_model.pkl .
RUN pip install flask scikit-learn numpy xgboost
EXPOSE 5000
CMD ["python" , "main.py"]
