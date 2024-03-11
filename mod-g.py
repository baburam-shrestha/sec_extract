import os
import yaml
import re
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from scrapingbee import ScrapingBeeClient
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow
from pandas import DataFrame
from logger_file import logger

import pyspark
from pyspark.sql import SparkSession, DataFrame, functions as F, types as T, Row
from pyspark.sql.window import Window
from pyspark import SparkContext

logger.info(f"Logger initialized successfully in file")

spark = SparkSession.builder.appName("Sec-Gov-Scraping").getOrCreate()
sc = SparkContext.getOrCreate()


def save_dataframepqt(df: DataFrame, path: str):
    df = df.coalesce(1)
    file_exists = os.path.exists(f"{path}")
    if file_exists:
        df.write.mode("append").parquet(path)
    else:
        df.write.mode("overwrite").parquet(path)


def scrape_document(cik_name: str, date: str, cik_num: str, accsNum: str, document: str, client:str, max_retries=3):
    retries = 0
    page_splitted = []
    pages_with_tables = []
    page_number = 0
    while retries < max_retries:
        try:
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{accsNum}/{document}"
            response = client.get(url)
            status = response.raise_for_status()
            if status == 200:
                html = str(response.content)
                soup = BeautifulSoup(html, 'html.parser')
                hr_tags = soup.find_all('hr')
                page_splitted = []
                pages_with_tables = []
                page_number = 0
                if not hr_tags:
                    pages_with_tables.append({
                            "cik_name": cik_name,
                            "reporting_date": date,
                            "url":url,
                            "page_number": page_number,
                            "page_content": soup
                        })
                    return pages_with_tables
                else:
                    first_hr = hr_tags[0]
                    content_before_first_hr = []
                    sibling = first_hr.find_previous_sibling()
    
                    while sibling and sibling.name != 'hr':
                        content_before_first_hr.append(str(sibling))
                        sibling = sibling.find_previous_sibling()
    
                    page_splitted.append(''.join(reversed(content_before_first_hr)))
                    for tag in hr_tags:
                        content = []
                        sibling = tag.find_next_sibling()
                        
                        while sibling and sibling.name != 'hr':
                            content.append(str(sibling).replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' '))
                            sibling = sibling.find_next_sibling()
    
                        page_splitted.append(''.join(content))
    
                    for page_content in page_splitted:
                        page_number += 1
                        soup = BeautifulSoup(page_content, "html.parser")
                        table_tags = soup.find_all("table")
    
                        if table_tags:
                            for tag in soup.find_all():
                                if "style" in tag.attrs:
                                    del tag.attrs["style"]
    
                            page_content = str(soup)
    
                            pages_with_tables.append({
                                "cik_name": cik_name,
                                "reporting_date": date,
                                "url":url,
                                "page_number": page_number,
                                "page_content": page_content
                            })
                    return pages_with_tables
        except requests.exceptions.HTTPError as errh:
            logger.info("Http Error:", errh)
            time.sleep(2)
        except requests.exceptions.ConnectionError as errc:
            logger.info("Error Connecting:", errc)
            time.sleep(2)
        except requests.exceptions.Timeout as errt:
            logger.info("Timeout Error:", errt)
            time.sleep(2)
        except requests.exceptions.RequestException as err:
            logger.info("OOps: Something Else", err)
        retries += 1
        time.sleep(5)
    return None


def scrap_table(dict_records: dict):
    cik_num = dict_records["cik_number"]
    date = dict_records["reportDate"]
    accsNum = dict_records["accessionNumber"]
    document = dict_records["primaryDocument"]
    cik_name = dict_records["cik_name"]
    tables = scrape_document(cik_name, date, cik_num, accsNum, document, client)
    logger.info(f"Scrapped the document from {document} inside the CIK name {cik_name}")
    data = []
    for table in tables:
        data.append(Row(**table))
    return data


def process_chunked_df(processed_rows):
    if len(processed_rows) > 0:
        logger.info(len(processed_rows))
        table_data = spark.sparkContext.parallelize(processed_rows)
        table_df = spark.createDataFrame(table_data)
        table_df = table_df.select(
            F.col("_1.cik_name").alias("cik_name"),
            F.col("_1.reporting_date").alias("reporting_date"),
            F.col("_1.url").alias("url"),
            F.col("_1.page_number").alias("page_number"),
            F.col("_1.page_content").alias("page_content"),
        )
        table_df = table_df.filter(F.col("page_content").isNotNull())
        save_dataframepqt(table_df, "data/output/table_contents/new_table_contents")
        logger.info(
            f"Saved the document to 'data/output/table_contents/new_table_contents'"
        )


if __name__ == "__main__":
    api_key = ''
    client = ScrapingBeeClient(api_key=api_key)
    logger.info(f"Scraping Table content Started")
    companie1_df = spark.read.parquet(
        "data/output/companies_details/company_details_file"
    )
    companie2_df = spark.read.parquet(
        "data/output/companies_details/company_details_fil"
    )
    companies_df = companie1_df.union(companie2_df)
    companies_df = companies_df.withColumn(
        "row_id", F.row_number().over(Window.orderBy("cik_number"))
    )
    total_items = companies_df.count()
    batch_size = 330
    num_batches = (total_items + batch_size - 1) // batch_size
    logger.info(f"Total Number of Batches: {num_batches}")
    for i in range(0,num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_items)
        logger.info(f"Batche of index: {start_idx}-{end_idx} and {i}/{num_batches} and started.")
        chunked_df = companies_df.filter(
            (F.col("row_id") >= start_idx) & (F.col("row_id") < end_idx)
        )
        with ThreadPoolExecutor(max_workers=9) as executor:
            chunked_df = chunked_df.drop("row_id")
            pd_df = chunked_df.toPandas()
            processed_rows = list(
                executor.map(scrap_table, pd_df.to_dict(orient="records"))
            )
        logger.info(len(processed_rows))
        process_chunked_df(processed_rows)
    logger.info(f"Scraping Table content Completed")
