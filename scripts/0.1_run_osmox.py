import os
import subprocess

import requests
from pyrosm import get_data

from acbm.cli import acbm_cli
from acbm.config import load_and_setup_config


@acbm_cli
def main(config_file):
    config = load_and_setup_config(config_file)
    logger = config.get_logger("preprocessing", __file__)
    logger.info("Getting OSM data")
    try:
        fp = get_data(config.region, directory=config.osmox_path)
    except Exception as e:
        logger.error(e)
        logger.info(f"Trying manual download of region: {config.region}")
        url = f"http://download.geofabrik.de/europe/united-kingdom/england/{config.region}-latest.osm.pbf"
        filename = url.split("/")[-1]
        fp = os.path.join(config.osmox_path, filename)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(fp, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Region ({config.region}) successfully downloaded to: {fp}")
    logger.info("Running osmox")
    subprocess.run(
        [
            "osmox",
            "run",
            # TODO: add to lib as string
            config.root_path / "osmox/config_osmox.json",
            fp,
            config.osmox_path / config.region,
            "-f",
            "geoparquet",
            "-crs",
            # TODO: check if this can be specified as the output CRS
            # See: https://github.com/arup-group/osmox/blob/82602d411374ebc9fd33443f8f7c9816b63715ec/docs/osmox_run.md#L35-L38
            "epsg:27700",
            "-l",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
