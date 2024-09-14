from typing import List
import difflib
import datetime

from loguru import logger
import pandas as pd


class MotorVideos:
    """
    Clase que sirve para ofrecer varias funciones que son capaces de transformar los datos recibidos en formatos
    distintos según las necesidades

    Attributes
    ----------
    config: dict
        configuracion de la clase

    Methods
    -------
    filtrar_datos():
        devuelve un nuevo DataFrame que solo tiene las columnas necesarias y se eliminan las filas con valores invalidos

    eliminar_etiquetas_similares():
        recibe una lista de etiquetas y se devuelve una lista nueva eliminando las etiquetas similares

    contar_etiquetas_pais():
        recibe un DataFrame con datos de vídeos y se devuelve un DataFrame nuevo que tiene todas las etiquetas y
        sus frecuencias en cada pais

    preparar_datos_a_entrenar():
        recibe un DataFrame que tiene las etiquetas más populares de cada semana y lo convierte en un DataFrame nuevo
        que tiene un formato adecuado para el entrenamiento de modelo

    evolucion_tendencias():
        recibe un DataFrame que tiene datos de las etiquetas de las últimas semanas, los transforma en un formato
        nuevo que tiene las etiquetas más populares y sus frecuencias en cada semana

    evaluar_prediccion():
        evaluar la prediccion calculando el porcentaje de etiquetas que existen en las tendencias reales
    """

    def __init__(self, config: dict):
        """
        Construccion del object de la clase MotorVideos inicializando los atributos

        :param config: configuracion del objeto motor
        """
        self.config = config

        logger.add(f'{self.config["logs_folder"]}/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log',
                   format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                   mode="a",
                   enqueue=True)

    def filtrar_datos(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve un nuevo DataFrame que solo tiene las columnas necesarias y elimina las filas con valores invalidos

        :param data: datos que quiere procesar
        :return: DataFrame de pandas que tiene solo las columnas que ha indicado en el fichero config.json y elimina
        las filas que tienen valores invalidos
        """
        result = None
        try:
            result = data[self.config["data_columns"]]
            result = result[result["country"].isin(self.config["paises"])]
            result = result.dropna()
            result.reset_index(drop=True, inplace=True)
            logger.info("Ha filtrado los datos con éxitos")
        except Exception as e:
            logger.error(f"Error al intentar filtrar los datos: {str(e)}")
            raise e

        return result

    def eliminar_etiquetas_similares(self, etiquetas: List[str]) -> List[str]:
        """
        Elimina las etiquetas similares que existen en una lista

        :param etiquetas: lista de etiquetas
        :return: lista de etiquetas teniendo las etiquetas similares eliminadas
        """
        result = []
        try:
            for etiqueta in etiquetas:
                etiqueta = etiqueta.lower()
                etiqueta.replace("'", "")
                es_similar = False
                for string in result:
                    calculo = difflib.SequenceMatcher(None, etiqueta, string).ratio()
                    if calculo > self.config["porc_similitud"]:
                        es_similar = True
                        break
                if not es_similar:
                    result.append(etiqueta)
            logger.info("Ha eliminado las etiquetas similares con éxitos")
        except Exception as e:
            logger.error(f"Error al intentar eliminar las etiquetas similares: {str(e)}")
            raise e

        return result

    def contar_etiquetas_pais(self, datos: pd.DataFrame) -> pd.DataFrame:
        """
        Recibe un DataFrame con datos de vídeos y se devuelve un DataFrame nuevo que tiene todas las etiquetas y
        sus frecuencias en cada pais

        :param datos: DataFrame que tiene datos de los vídeos
        :return: DataFrame nuevo que tiene todas las etiquetas y sus frecuencias en cada pais
        """
        result = None
        try:
            datos["cleaned_tags"] = datos["video_tags"].apply(
                lambda x: self.eliminar_etiquetas_similares(x.split(", ")))
            datas_json = {}
            for i in datos.index:
                pais = datos.iloc[i].country
                for tag in datos.iloc[i].cleaned_tags:
                    tag = tag.lower()
                    if tag not in datas_json.keys():
                        datas_json[tag] = {"total": 0}
                        for item in self.config["paises"]:
                            datas_json[tag][item] = 0
                    if pais in self.config["paises"]:
                        datas_json[tag][pais] = datas_json[tag][pais] + 1
                        datas_json[tag]["total"] = datas_json[tag]["total"] + 1

            datas = pd.DataFrame.from_dict(datas_json)
            datas = (datas.T.sort_values(by=["total"], ascending=False)
                     .head(self.config["num_etiquetas"]).reset_index(names="etiquetas"))
            temp = pd.DataFrame(columns=["etiquetas", "pais", "frecuencias", "total"])
            for i in datas.index:
                row = {"etiquetas": datas.iloc[i]["etiquetas"],
                       "total": datas.iloc[i]["total"]}
                for pais in self.config["paises"]:
                    row["pais"] = pais
                    row["frecuencias"] = datas.iloc[i][pais]
                    temp.loc[len(temp)] = row
            fechas = list(datos.snapshot_date.unique())
            fechas.sort()
            temp_periodo = pd.DataFrame([[f"{fechas[0]} - {fechas[-1]}"]],
                                        columns=["periodo"])
            result = temp_periodo.join(temp, how="cross")
            logger.info("Ha contado la frecuencia de todas las etiquetas con éxito")
        except Exception as e:
            logger.error(f"Error al contar etiquetas: {str(e)}")
            raise e

        return result

    @staticmethod
    def preparar_datos_a_entrenar(datos: pd.DataFrame) -> pd.DataFrame:
        """
        Recibe un DataFrame que tiene las etiquetas más populares de cada semana y lo convierte en un DataFrame nuevo
        que tiene un formato adecuado para el entrenamiento de modelo.

        :param datos: DataFrame que tiene las etiquetas más populares de cada semana y sus frecuencias
        :return: DataFrame con nuevo formato, tiene las etiquetas como el index, y las columnas son los periodos,
        los valores son las frecuencias de cada etiqueta en cada periodo
        """
        result = pd.DataFrame()
        try:
            for periodo in datos.periodo.unique():
                temp = datos[datos["periodo"] == periodo][["etiquetas", "frecuencias"]]
                temp.set_index("etiquetas", inplace=True)
                temp = temp.T
                temp.insert(0, "periodo", [periodo])
                result = pd.concat([result, temp])
            result.sort_values(by=["periodo"], inplace=True)
            result.set_index("periodo", inplace=True)
            result.fillna(0, inplace=True)
            result = result.T
            logger.info("Ha transformar los datos para el entrenamiento de modelos con éxitos")
        except Exception as e:
            logger.error(f"Error al transformar los datos para el entrenamiento de modelos: {str(e)}")
            raise e

        return result

    def evolucion_tendencias(self, datos: pd.DataFrame) -> pd.DataFrame:
        """
        Recibe un DataFrame que tiene datos de las etiquetas de las últimas semanas, los transforma en un formato
        nuevo que tiene las etiquetas más populares y sus frecuencias en cada semana

        :param datos: DataFrame que tiene datos de las etiquetas de las últimas semanas
        :return: DataFrame con las etiquetas más populares y sus frecuencias en cada semana
        """
        result = None
        try:
            resultado = datos
            resultado["total"] = resultado.sum(axis=1)
            resultado.sort_values(by=["total"], ascending=False, inplace=True)
            resultado = resultado.iloc[:self.config["top_n"], :-1]
            result = pd.DataFrame(columns=["periodo", "etiquetas", "frecuencias"])
            for periodo in resultado.columns.values:
                row = {"periodo": periodo}
                for i in range(len(resultado.index)):
                    row["etiquetas"] = resultado.index.values[i]
                    row["frecuencias"] = resultado.iloc[i][periodo]
                    result.loc[len(result)] = row
            logger.info("Ha calculado la evolucion de tendencias con éxitos")
        except Exception as e:
            logger.error(f"Error al calcular la evolucion de tendencias: {str(e)}")
            raise e

        return result

    @staticmethod
    def evaluar_prediccion(tendencias: pd.DataFrame, prediccion: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluar la prediccion calculando el porcentaje de etiquetas que existen en las tendencias reales

        :param tendencias: DataFrame de las etiquetas ordenadas por sus frecuencias descendentemente
        :param prediccion: DataFrame de las etiquetas ordenadas por sus frecuencias descendentemente
        :return: DataFrame que indica el periodo de las tendencias y el resultado de la evaluación
        """
        count = 0
        result = None
        try:
            for etiqueta in prediccion.get("etiquetas").tolist():
                if etiqueta in tendencias.get("etiquetas").tolist():
                    count = count + 1
            result = pd.DataFrame([[tendencias.iloc[0].periodo, count / len(tendencias.etiquetas.values)]],
                                  columns=['periodo', 'evaluacion'])
            logger.info("Ha evaluado la predicción con éxito")
        except Exception as e:
            logger.error(f"Error al evaluar la predicción de la semana pasada: {str(e)}")
            raise e

        return result
