import requests
import pandas as pd
import io

def get_ecb_data(flowRef: str, key: str, parameters: dict = None) -> pd.DataFrame:
    """
    Download ECB SDW data using the official API syntax:
        protocol://wsEntryPoint/resource/flowRef/key?parameters

    Example:
        get_ecb_data(
            flowRef="MNA",
            key="Q.N.I9.W2.S1.S1.B.B1GQ.Z.Z.Z.EUR.LR.N",
            parameters={"format": "csvdata"}
        )

    Returns:
        A pandas DataFrame with the ECB data.
    """
    base_url = "https://data-api.ecb.europa.eu/service/data"
    url = f"{base_url}/{flowRef}/{key}"

    # Add query parameters
    if parameters is not None:
        # convert dict -> query string
        query = "&".join([f"{k}={v}" for k, v in parameters.items()])
        url = f"{url}?{query}"
    else:
        url = f"{url}?format=csvdata"

    print(f"Requesting URL: {url}")

    response = requests.get(url)
    response.raise_for_status()  # raise error if request failed
    df = pd.read_csv(io.StringIO(response.text))

    print(f"Received {len(df)} rows and {len(df.columns)} columns")
    return df
