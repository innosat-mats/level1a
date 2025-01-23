from typing import Dict
from pandas import DataFrame


# Map all channels to string names
channel_num_to_str: Dict[int, str] = {
    1: "IR1",
    4: "IR2",
    3: "IR3",
    2: "IR4",
    5: "UV1",
    6: "UV2",
    7: "NADIR",
}

channel_to_tempertature: Dict[str, str] = {
    "UV1": "HTR8B",
    "UV2": "HTR8A",
}


def add_ccd_item_attributes(ccd_data: DataFrame) -> None:
    """Add some attributes to CCD data that we need.
    Note that this function assumes the data has up to date names for columns,
    not the names used in the old rac extract file (prior to May 2020).
    Conversion to the old standard can be performed using
    `rename_ccd_item_attributes`, but that has to be done _after_ applying this
    function.

    Args:
        ccd_data (DataFrame):   CCD data to which to add attributes.

    Returns:
        None:   Operation is performed in place.
    """

    ccd_data["channel"] = [channel_num_to_str[c] for c in ccd_data["CCDSEL"]]
    ccd_data["flipped"] = False

    # CCDitem["id"] should not be needed in operational retrieval. Keeping it
    # because protocol reading / CodeCalibrationReport needs it.  LM220908
    ccd_data["id"] = [
        f"{nanos}_{ccd}" for nanos, ccd in zip(ccd_data.EXPNanoseconds, ccd_data.CCDSEL)
    ]

    # Add temperature info fom OBC, the temperature info from the rac files are
    # better since they are based on the thermistors on the UV channels
    ADC_temp_in_mV = ccd_data["TEMP"] / 32768 * 2048
    ADC_temp_in_degreeC = 1.0 / 0.85 * ADC_temp_in_mV - 296
    ccd_data["temperature_ADC"] = ADC_temp_in_degreeC

    # This needs to be updated when a better temperature estimate has been
    # designed. For now a de facto implementation of
    # get_temperature.add_temperature_info()

    temperatures = [
        (
            ccd_item[channel_to_tempertature[ccd_item["channel"]]]
            if ccd_item["channel"].startswith("UV")
            else (ccd_item["HTR8A"] + ccd_item["HTR8B"]) * 0.5
        )
        for _, ccd_item in ccd_data.iterrows()
    ]

    ccd_data["temperature"] = temperatures
    ccd_data["temperature_HTR"] = 0.5 * (ccd_data["HTR8A"] + ccd_data["HTR8B"])
