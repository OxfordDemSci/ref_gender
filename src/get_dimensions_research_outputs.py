import pandas as pd
import json
from pathlib import Path
import dimcli
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
import io
import os
import time
import logging
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm  # Standard tqdm works fine in helpers
import numpy as np
import pandas as pd


def login_dimcli():
    def get_apikey(key_file_path):
        if not key_file_path.is_file():
            raise FileNotFoundError(f"API key file not found at {key_file_path.resolve()}")
        with key_file_path.open("r") as file:
            api_key = file.read().strip()
        return api_key

    dimcli.login(key=get_apikey(Path("../keys/dimensions_apikey.txt")),
                 endpoint="https://app.dimensions.ai/api/dsl/v2")
    dsl = dimcli.Dsl()
    return dsl


def get_pubs(string_representation, field, limit, logger=None):
    query = f"""search publications
    where {field} in {string_representation}
    return publications[authors + authors_count + category_for_2020 +
                        dimensions_url + doi + isbn +
                        id + year] limit {limit}"""
    capture = io.StringIO()
    with redirect_stdout(capture), redirect_stderr(capture):
        result = dsl.query(query)
    captured_output = capture.getvalue()
    if logger is not None:
        logger.info("Captured DSL query output: %s", captured_output)
    return result.as_dataframe()


def get_raw_data(limit, fields, df, timestamp):
    max_retries = 100
    for field in fields:
        field_log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"../logging/dimensions/api/{field}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        logfile = f"{log_dir}/logfile_{field_log_time}.txt"
        logger = logging.getLogger(field)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.FileHandler(logfile)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(f"Starting work on the {field} field")
        if field == 'doi':
            data_to_get = df[df['DOI'].notnull()]['DOI'].tolist()
        elif field == 'isbn':
            data_to_get = df[df['ISBN'].notnull()]['ISBN'].tolist()
        else:
            print('Some weird field has been passed in')
            break
        for i in tqdm(range(0, len(data_to_get), limit), desc=f"Processing {field} chunks"):
            fpath = f'../data/dimensions_outputs/api/raw/{field}/{timestamp}/df_{i}_to_{i + limit}.csv'
            if not os.path.exists(fpath):
                directory = os.path.dirname(fpath)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                logger.info(f"Processing file: {fpath}")
                chunk = [int(x) if isinstance(x, np.integer) else x for x in data_to_get[i:i + limit]]
                string_representation = json.dumps(chunk)
                for attempt in range(max_retries):
                    try:
                        df_chunk = get_pubs(string_representation, field, limit, logger)
                        logger.info("Collected chunk %d (attempt %d/%d).",
                                     i, attempt+1, max_retries)
                        break
                    except Exception as e:
                        logger.error("Error encountered for chunk %d (attempt %d/%d): %s",
                                     i, attempt+1, max_retries, e)
                        if attempt < max_retries - 1:
                            sleep_time = 2 ** attempt  # Exponential backoff.
                            logger.info("Retrying in %d seconds...", sleep_time)
                            time.sleep(sleep_time)
                        else:
                            logger.info("Max retries reached. Skipping this chunk.")
                            print('Warning! A chunk couldnt be collected at all?! INVESTIGATE!')
                            df_chunk = None
                if df_chunk is not None:
                    df_chunk.to_csv(fpath)
    logger.info("Tada!")


if __name__ == "__main__":
    df = pd.read_excel('../data/raw/raw_ref_outputs_data.xlsx', skiprows=4)
    dsl = login_dimcli()
    timestamp = datetime.now().strftime("%Y%m")
    get_raw_data(100, ['doi', 'isbn'], df, timestamp)