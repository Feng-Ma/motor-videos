import json
import datetime

import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from .motor_videos import MotorVideos


class FlujoSemanal:
    """
    Clase que sirve para leer y procesar los datos de tendencias de videos y guardar los resultados en distintos
    luegares

    Attributes
    ----------
    config: dict
        configuracion de la clase
    spark: sparksession
        objeto de sparksession para generar y gestionar los dataframes

    Methods
    -------
    refinar_datos():
        Refinar los datos de la capa bronze y guardar el resultado en la capa silver

    silver_to_gold():
        Procesar los datos de la capa silver y guardar el resultado en la capa gold

    calcular_tendencias():
        Leer los datos de la capa silver y calcular las tendencias de la última semana

    actualizar_weekly_trends_country():
        Actualizar la tabla GOLD.WEEKLY_TRENDS_COUNTRY con los datos que ha recibido

    actualizar_weekly_trends():
        Actualizar la tabla GOLD.WEEKLY_TRENDS con los datos que ha recibido

    actualizar_10_weeks_trends():
        Actualizar la tabla GOLD.10_WEEKS_TRENDS con los datos que ha recibido

    actualizar_last_prediction():
        Actualizar la tabla GOLD.LAST_PREDICTION con los datos que ha recibido

    actualizar_new_prediction():
        Actualizar la tabla GOLD.NEW_PREDICTION con los datos que ha recibido

    evaluate_last_prediction():
        Evaluar la prediccion anterior y actualizar la tabla GOLD.EVALUATIONS con el resultado de la evaluacion
    """

    def __init__(self, config_file: str):
        """
        Construccion del object de la clase FlujoSemanal inicializando los atributos
        :param config_file: ruta del fichero de configuracion
        """
        self.spark = SparkSession.builder.getOrCreate()
        with open(config_file) as config_file:
            config = json.load(config_file)
        self.config = config
        logger.add(f'{self.config["logs_folder"]}/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log',
                   format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                   mode="a",
                   enqueue=True)

    def refinar_datos(self, source_path: str, output_path: str):
        """
        Leer el fichero de la capa bronze, refinar los datos y guardarlos en la capa silver
        :param source_path: ruta del fichero de origen
        :param output_path: ruta del fichero de destino
        :return: None
        """
        try:
            data = self.spark.read.option("header", "true").option("delimiter", ",").csv(source_path)
            data_pd = data.select("*").toPandas()

            motor_videos = MotorVideos(self.config)
            refined_data = motor_videos.filtrar_datos(data_pd)
            logger.info("Ha refinado los datos de bronze con éxitos")

            refined_data_spark = self.spark.createDataFrame(refined_data)
            refined_data_spark.write.mode("overwrite").format("delta").save(f"{output_path}/last_week")

            fechas = refined_data_spark.select("snapshot_date").distinct().orderBy("snapshot_date")
            first_date = fechas.first().snapshot_date
            last_date = fechas.orderBy("snapshot_date", ascending=False).first().snapshot_date
            refined_data_spark.write.mode("overwrite").format("delta").save(f"{output_path}/{first_date} - {last_date}")
            logger.info("Ha guardado los datos refinados en la capa silver con éxitos")
        except Exception as e:
            logger.error(f"Error al procesar los datos de la capa bronze: {str(e)}")
            raise e

    def calcular_tendencias(self, source_path: str) -> pd.DataFrame | None:
        """
        Leer el fichero que ha especificado por source_path, calcular y devolver un DataFrame de pandas
        que tiene la frecuencia de etiquetas
        :param source_path: ruta del fichero
        :return: DataFrame de pandas que tiene la frecuencia de etiquetas
        """
        result = None
        try:
            data = self.spark.read.format("delta").load(source_path)
            motor_videos = MotorVideos(self.config)
            result = motor_videos.contar_etiquetas_pais(data.select("*").toPandas())
        except Exception as e:
            logger.error(f"Error al calcular la frecuencia de etiquetas: {str(e)}")
            raise e

        return result

    def actualizar_weekly_trends_country(self, datos: pd.DataFrame):
        """
        Actualizar la tabla GOLD.WEEKLY_TRENDS_COUNTRY con los datos del DataFrame que ha recibido
        :param datos: datos que quiere actualizar
        :return: None
        """
        try:
            schema = StructType([StructField('periodo', StringType(), True),
                                 StructField('etiquetas', StringType(), True),
                                 StructField('pais', StringType(), True),
                                 StructField('frecuencias', IntegerType(), True)])
            result = self.spark.createDataFrame(datos[['periodo', 'etiquetas', 'pais', 'frecuencias']], schema=schema)
            result.write.insertInto('GOLD.WEEKLY_TRENDS_COUNTRY')
            logger.info("Ha actualizado la tabla GOLD.WEEKLY_TRENDS_COUNTRY con éxitos")
        except Exception as e:
            logger.error(f"Error al actualizar la tabla GOLD.WEEKLY_TRENDS_COUNTRY: {str(e)}")
            raise e

    def actualizar_weekly_trends(self, datos: pd.DataFrame):
        """
        Actualizar la tabla GOLD.WEEKLY_TRENDS con los datos del DataFrame que ha recibido
        :param datos: datos que quiere actualizar
        :return: None
        """
        try:
            schema = StructType([StructField('periodo', StringType(), True),
                                 StructField('etiquetas', StringType(), True),
                                 StructField('frecuencias', IntegerType(), True)])
            result = self.spark.createDataFrame(datos, schema=schema)
            result.write.insertInto('GOLD.WEEKLY_TRENDS')
            logger.info("Ha actualizado la tabla GOLD.WEEKLY_TRENDS con éxitos")
        except Exception as e:
            logger.error(f"Error al actualizar la tabla GOLD.WEEKLY_TRENDS: {str(e)}")
            raise e

    def actualizar_10_weeks_trends(self, datos: pd.DataFrame):
        """
        Actualizar la tabla GOLD.10_WEEKS_TRENDS con los datos del DataFrame que ha recibido
        :param datos: datos que quiere actualizar
        :return: None
        """
        try:
            motor_videos = MotorVideos(self.config)
            data = motor_videos.evolucion_tendencias(datos)
            schema = StructType([StructField('periodo', StringType(), True),
                                 StructField('etiquetas', StringType(), True),
                                 StructField('frecuencias', FloatType(), True)])
            result = self.spark.createDataFrame(data, schema=schema)
            result.write.insertInto('GOLD.10_WEEKS_TRENDS', overwrite=True)
            logger.info("Ha actualizado la tabla GOLD.10_WEEKS_TRENDS con éxitos")
        except Exception as e:
            logger.error(f"Error al actualizar la tabla GOLD.10_WEEKS_TRENDS: {str(e)}")
            raise e

    def actualizar_last_prediction(self):
        """
        Actualizar la tabla GOLD.LAST_PREDICTION con los datos de la tabla GOLD.NEW_PREDICTION
        :return: None
        """
        try:
            predictions = self.spark.sql("SELECT * FROM GOLD.NEW_PREDICTION")
            predictions.write.insertInto('GOLD.LAST_PREDICTION', overwrite=True)
            logger.info("Ha actualizado la tabla GOLD.LAST_PREDICTION con éxitos")
        except Exception as e:
            logger.error(f"Error al actualizar la tabla GOLD.LAST_PREDICTION: {str(e)}")
            raise e

    def actualizar_new_prediction(self, datos: pd.DataFrame):
        """
        Actualizar la tabla GOLD.10_WEEKS_TRENDS con los datos del DataFrame que ha recibido
        :param datos: datos que quiere actualizar
        :return: None
        """
        try:
            schema = StructType([StructField('etiquetas', StringType(), True),
                                 StructField('frecuencias', FloatType(), True)])
            result = self.spark.createDataFrame(datos, schema=schema)
            result.write.insertInto('GOLD.NEW_PREDICTION', overwrite=True)
            logger.info("Ha actualizado la tabla GOLD.NEW_PREDICTION con éxitos")
        except Exception as e:
            logger.error(f"Error al actualizar la tabla GOLD.NEW_PREDICTION: {str(e)}")
            raise e

    def evaluate_last_prediction(self):
        """
        Evaluar las predicciones de la tabla GOLD.LAST_PREDICTION comparando con las tendencias de
        la ultima semana y guardar el resultado en la tabla GOLD.EVALUATIONS
        :return: None
        """
        try:
            tendencias = self.spark.sql("select * from GOLD.WEEKLY_TRENDS order by periodo desc, frecuencias desc limit 30")
            predicciones = self.spark.sql("select * from GOLD.LAST_PREDICTION")
            result = MotorVideos.evaluar_prediccion(tendencias.toPandas(), predicciones.toPandas())

            schema = StructType([StructField('periodo', StringType(), True),
                                 StructField('evaluacion', FloatType(), True)])
            evaluacion = self.spark.createDataFrame(result, schema=schema)
            evaluacion.write.insertInto('GOLD.EVALUATIONS')
            logger.info("Ha actualizado la tabla GOLD.EVALUATIONS con éxitos")
        except Exception as e:
            logger.error(f"Error al evaluar la prediccion anterior: {str(e)}")
            raise e
