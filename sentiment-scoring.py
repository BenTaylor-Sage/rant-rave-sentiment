# Databricks notebook source
import torch
import mlflow.pytorch
from pyspark.sql.functions import col, udf, lit, to_utc_timestamp, split, collect_list
from pyspark.sql import Row
from pyspark.sql.types import *
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from pyspark.sql.types import *
import re
import mlflow.pytorch.pickle_module
import time
import datetime

# COMMAND ----------

model_version = "1.2"

# COMMAND ----------

def get_randr_data():
  sf_key_passphrase = dbutils.secrets.get(scope="snowflake", key="pk_pass_phrase")
  sf_service_account = dbutils.secrets.get(scope="snowflake", key="account_name")
  sf_cert_location = "/dbfs/sage/certs/snowflake_private_key.p8"

  p_key = None

  with open(sf_cert_location, "rb") as key_file:
    p_key = serialization.load_pem_private_key( \
      key_file.read(), \
      password=sf_key_passphrase.encode(), \
      backend=default_backend() \
    )
    key_file.close()

  pkb = p_key.private_bytes(encoding=serialization.Encoding.PEM, \
    format=serialization.PrivateFormat.PKCS8, \
    encryption_algorithm=serialization.NoEncryption() \
  )

  pkb = pkb.decode("UTF-8")
  pkb = re.sub("-*(BEGIN|END) PRIVATE KEY-*\n","",pkb).replace("\n","")
  
  sfOptions = {
  "sfURL" : "sage.eu-west-1.privatelink.snowflakecomputing.com",
  "sfRole": "PROD_BSU_BIGDATA_SUPER_USER",
  "sfUser" : sf_service_account,
  "sfDatabase" : "PROD_DB",
  "sfSchema" : "MODELED",
  "sfWarehouse" : "PROD_ELT_RAW_WH",
  "pem_private_key": pkb
}

  df = spark.read.format("snowflake") \
    .options(**sfOptions) \
    .option("query",  "SELECT ID, MESSAGE FROM RANTANDRAVE_TRANSACTIONNPS WHERE MODELED_CURRENT_RECORD = 'Yes'") \
    .load() \
    .withColumn('SourceSystem', lit("PROD_DB.MODELED.RANTANDRAVE_TRANSACTIONNPS"))

  df2 = spark.read.format("snowflake") \
    .options(**sfOptions) \
    .option("query",  "SELECT ID, MESSAGE FROM RANTANDRAVE_CUSTOMERRELATIONSHIPNPS WHERE MODELED_CURRENT_RECORD = 'Yes'") \
    .load() \
    .withColumn('SourceSystem', lit("PROD_DB.MODELED.RANTANDRAVE_CUSTOMERRELATIONSHIPNPS"))

  df3 = spark.read.format("snowflake") \
    .options(**sfOptions) \
    .option("query",  "SELECT ID, MESSAGE FROM RANTANDRAVE_BUSINESSPARTNERSRELATIONSHIPNPS WHERE MODELED_CURRENT_RECORD = 'Yes'") \
    .load() \
    .withColumn('SourceSystem', lit("PROD_DB.MODELED.RANTANDRAVE_BUSINESSPARTNERSRELATIONSHIPNPS"))
  
  dfsent = spark.read.format("snowflake") \
    .options(**sfOptions) \
    .option("query", "SELECT FEEDBACK_ITEM_ID, VERSION FROM RANTANDRAVE_SENTIMENTSCORE WHERE VERSION = " + model_version) \
    .load()

  df = df.dropna()
  df2 = df2.dropna()
  df3 = df3.dropna()

  sourcedf = df.union(df2).union(df3)
  sourecdf = sourcedf.distinct()
  
  finaldf = sourcedf.join(dfsent,sourcedf.ID == dfsent.FEEDBACK_ITEM_ID,"leftanti")
  
  return finaldf

# COMMAND ----------

def sentiment_score(text):
  tokens = r_and_r_tokenizer.encode(text, return_tensors='pt', truncation=True)
  result = model(tokens)
  return int(torch.argmax(result.logits))+1

# COMMAND ----------

predict_udf = udf(lambda x:sentiment_score(x))

# COMMAND ----------

timestamp = datetime.datetime.fromtimestamp(time.time())

r_and_r_tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model_state_dict = mlflow.pytorch.load_state_dict('dbfs:/databricks/mlflow-tracking/129067/ef164ae8561b4dd8b7cde3303352f0dc/artifacts/bert-randr-sentiment')
model.load_state_dict(model_state_dict)

# COMMAND ----------

data = get_randr_data()

# COMMAND ----------

display(data)

# COMMAND ----------

pandas_data = data.toPandas()
max_size = 512
pandas_data['MESSAGE'] = pandas_data['MESSAGE'].str.split(n=max_size).str[:max_size].str.join(' ')

# COMMAND ----------

pdf = pandas_data.head()
pdata = spark.createDataFrame(pdf)
display(pdata)

# COMMAND ----------

data = spark.createDataFrame(pandas_data)

# COMMAND ----------

data_predicted = (data
                  .withColumn('ID', data['ID'])
                  .withColumn("SentimentScore", predict_udf(data["MESSAGE"]))
                  .withColumn('Version', lit(model_version))
                  .withColumn('DateScored', lit(timestamp))
                  .drop('MESSAGE')
                 )

# COMMAND ----------

display(data_predicted)

# COMMAND ----------

save_path = '/mnt/da-global-raw-prod/Rant_and_Rave/Sentiment'
data_predicted.repartition(1).write.format('parquet').mode('append').save(save_path)
