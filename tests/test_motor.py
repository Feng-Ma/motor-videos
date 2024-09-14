from motor_videos.motor_videos import MotorVideos
import pandas as pd
import json


def test_filtrar_datos():
    """
    Testea que la funcion transformar_datos() devuelve un dataframe con las columnas que ha indicado en
    el fichero config.json
    :return: None
    """
    datos = pd.DataFrame([['Hermitcraft 10: Episode 19 - THE RETURN!', 'Grian', '2024-06-30',
                           'US', 'hermitcraft season 10, grian, episode 18, hermitcraft'],
                          ['Lore Revelations in the Shadow of the Erdtree', 'VaatiVidya', '2024-06-30',
                           'ES', 'guide, walkthrough, lore, dark, souls']],
                         columns=["title", "channel_name", "snapshot_date", "country", "video_tags"])

    resultado_esperado = pd.DataFrame([['2024-06-30', 'US', 'hermitcraft season 10, grian, episode 18, hermitcraft']],
                                      columns=["snapshot_date", "country", "video_tags"])

    with open("resources/test_config.json") as config_file:
        config = json.load(config_file)
    motor_videos = MotorVideos(config)
    resultado = motor_videos.filtrar_datos(datos)

    assert (resultado.equals(resultado_esperado))


def test_eliminar_etiquetas_similares():
    """
    Testea que la funcion eliminar_etiquetas_simialres() puede eliminar las etiquetas que tienen el porcentaje
    de similitud mayor al valor que indica en el fichero config.json
    :return: None
    """
    datos = ("Finesse2Tymes, Finesse 2 Tymes, Finesse 2Tymes, Finesse2Times, Finesse 2 Times, Finesse 2Times").split(
        ",")
    resultado_esperado = ["finesse2tymes"]

    with open("resources/test_config.json") as config_file:
        config = json.load(config_file)
    motor_videos = MotorVideos(config)
    resultado = motor_videos.eliminar_etiquetas_similares(datos)

    assert (resultado == resultado_esperado)


def test_contar_etiquetas_pais():
    """
    Testea que la función contar_etiquetas_pais() puede contar la frecuencia de cada etiqueta correctamente
    :return: None
    """
    datos = pd.DataFrame([['2024-06-30', 'US', 'Football, music'],
                          ['2024-06-25', 'US', 'football, Music'],
                          ['2024-06-27', 'GB', 'football']],
                         columns=["snapshot_date", "country", "video_tags"])

    resultado_esperado = pd.DataFrame([["2024-06-25 - 2024-06-30", "football", "US", 2, 3],
                                       ["2024-06-25 - 2024-06-30", "football", "CA", 0, 3],
                                       ["2024-06-25 - 2024-06-30", "football", "GB", 1, 3],
                                       ["2024-06-25 - 2024-06-30", "football", "AU", 0, 3],
                                       ["2024-06-25 - 2024-06-30", "football", "NZ", 0, 3],
                                       ["2024-06-25 - 2024-06-30", "football", "IE", 0, 3],
                                       ["2024-06-25 - 2024-06-30", "music", "US", 2, 2],
                                       ["2024-06-25 - 2024-06-30", "music", "CA", 0, 2],
                                       ["2024-06-25 - 2024-06-30", "music", "GB", 0, 2],
                                       ["2024-06-25 - 2024-06-30", "music", "AU", 0, 2],
                                       ["2024-06-25 - 2024-06-30", "music", "NZ", 0, 2],
                                       ["2024-06-25 - 2024-06-30", "music", "IE", 0, 2]],
                                      columns=["periodo", "etiquetas", "pais", "frecuencias", "total"])
    with open("resources/test_config.json") as config_file:
        config = json.load(config_file)
    motor_videos = MotorVideos(config)
    resultado = motor_videos.contar_etiquetas_pais(datos)
    print(resultado)
    print(resultado_esperado)
    assert resultado.equals(resultado_esperado)


def test_preparar_datos_a_entrenar():
    """
    Testea que la función preparar_datos_a_entrenar() puede transformar el DataFrame que ha recibido
    en un formato más adecuado para el entrenamiento de modelos
    :return: None
    """
    datos = pd.DataFrame([['2024-06-24 - 2024-06-30', 'football', 30],
                          ['2024-06-24 - 2024-06-30', 'games', 25],
                          ['2024-06-17 - 2024-06-23', 'music', 20],
                          ['2024-06-17 - 2024-06-23', 'football', 40]],
                         columns=["periodo", "etiquetas", "frecuencias"])

    resultado_esperado = pd.DataFrame([["football", 40.0, 30.0],
                                       ["games", 0.0, 25.0],
                                       ["music", 20.0, 0.0]],
                                      columns=["etiquetas", "2024-06-17 - 2024-06-23", "2024-06-24 - 2024-06-30"])
    resultado_esperado.set_index("etiquetas", inplace=True)
    with open("resources/test_config.json") as config_file:
        config = json.load(config_file)
    motor_videos = MotorVideos(config)
    resultado = motor_videos.preparar_datos_a_entrenar(datos)

    assert resultado.equals(resultado_esperado)


def test_evolucion_tendencias():
    """
    Testea que la funcion evaolucion_tendencias() puede transformar el DataFrame que ha recibido en un formato
    más adecuado que indica el cambio de frecuencia de cada etiqueta
    :return: None
    """
    datos = pd.DataFrame([["football", 40, 30],
                          ["games", 0, 25],
                          ["music", 20, 10]],
                         columns=["etiquetas", "2024-06-17 - 2024-06-23", "2024-06-24 - 2024-06-30"])
    datos.set_index("etiquetas", inplace=True)

    resultado_esperado = pd.DataFrame([["2024-06-17 - 2024-06-23", "football", 40],
                                       ["2024-06-17 - 2024-06-23", "music", 20],
                                       ["2024-06-17 - 2024-06-23", "games", 0],
                                       ["2024-06-24 - 2024-06-30", "football", 30],
                                       ["2024-06-24 - 2024-06-30", "music", 10],
                                       ["2024-06-24 - 2024-06-30", "games", 25]],
                                      columns=["periodo", "etiquetas", "frecuencias"])
    with open("resources/test_config.json") as config_file:
        config = json.load(config_file)
    motor_videos = MotorVideos(config)
    resultado = motor_videos.evolucion_tendencias(datos)

    assert resultado.equals(resultado_esperado)


def test_evaluar_prediccion():
    """
    Testea que la funcion evaluar_prediccion() pueda evaluar la prediccion correctamente
    :return: None
    """
    tendencias = pd.DataFrame([["2024-06-17 - 2024-06-23", "football", 40],
                          ["2024-06-17 - 2024-06-23", "games", 25],
                          ["2024-06-17 - 2024-06-23", "music", 20],
                          ["2024-06-17 - 2024-06-23", "news", 15]],
                         columns=["periodo", "etiquetas", "frecuencias"])
    prediccion = pd.DataFrame([["2024-06-17 - 2024-06-23", "comedy", 50],
                               ["2024-06-17 - 2024-06-23", "football", 45],
                               ["2024-06-17 - 2024-06-23", "top 10", 22],
                               ["2024-06-17 - 2024-06-23", "news", 17]],
                              columns=["periodo", "etiquetas", "frecuencias"])

    resultado_esperado = pd.DataFrame([["2024-06-17 - 2024-06-23", 0.5]],
                                      columns=['periodo', 'evaluacion'])

    with open("resources/test_config.json") as config_file:
        config = json.load(config_file)
    motor_videos = MotorVideos(config)
    resultado = motor_videos.evaluar_prediccion(tendencias, prediccion)

    assert resultado.equals(resultado_esperado)
