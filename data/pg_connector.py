import pandas as pd
import logging

def get_data(link: str) -> pd.DataFrame:

    logging.info('Extracting df')
    df = pd.read_csv(link)
    logging.info('Df is extracted')

    return df