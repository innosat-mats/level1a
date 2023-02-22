from typing import Tuple
import datetime as dt
import pathlib

from numpy.linalg import norm
from skyfield.api import load, wgs84, Time  # type: ignore
from skyfield.errors import EphemerisRangeError  # type: ignore
from skyfield.positionlib import Geocentric  # type: ignore
from skyfield.framelib import itrs  # type: ignore
from skyfield.units import Distance  # type: ignore
import numpy as np

DEGREES_PER_HOUR = 15
SECONDS_PER_HOUR = 3600

PLANETS = load(str(pathlib.Path(__file__).parent.resolve() / "de421.bsp"))


def eci_to_latlon(
    time: Time,
    eci_pos: np.ndarray,
):
    """Function giving the GPS position in lat/lon/alt.

    Arguments:
        time (Time):        skyfield Time to use
        eci_pos (ndarray):  ECI position

    Returns:
        float:  latitude of satellite (degrees)
        float:  longitude of satellite (degrees)
        float:  altitude (meters)
    """
    pos = Geocentric(position_au=Distance(m=eci_pos).au, t=time)
    lat, lon, alt = pos.frame_latlon(itrs)
    return lat.degrees, lon.degrees, alt.m


def solar_angles(
    time: Time,
    sat_lat: float,
    sat_lon: float,
    sat_alt: float,
    tp_lat: float,
    tp_lon: float,
    tp_alt: float,
) -> Tuple[float, float, float]:
    """Function giving various solar angles.

    Arguments:
        time (Time):        skyfield Time to use
        sat_lat (float):    latitude of satellite (degrees)
        sat_lon (float):    longitude of satellite (degrees)
        sat_alt (float):    altitude (meters)
        tp_lat (float):     latitude of satellite (degrees)
        tp_lon (float):     longitude of satellite (degrees)
        tp_alt (float):     altitude (metres)

    Returns:
        float:  solar zenith angle at satellite position (degrees)
        float:  solar zenith angle at TP position (degrees)
        float:  solar scattering angle at TP position (degrees)
    """
    earth, sun = PLANETS['earth'], PLANETS['sun']

    try:
        sat_pos = earth + wgs84.latlon(sat_lat, sat_lon, elevation_m=sat_alt)
        sun_dir = sat_pos.at(time).observe(sun).apparent()
        obs = sun_dir.altaz()
        nadir_sza = (90 - obs[0].degrees)

        tp_pos = earth + wgs84.latlon(tp_lat, tp_lon, elevation_m=tp_alt)
        fov = (tp_pos - sat_pos).at(time).position.m
        fov = fov / norm(fov)
        sun_dir = tp_pos.at(time).observe(sun).apparent()
        obs = sun_dir.altaz()
        tp_sza = 90 - obs[0].degrees
        tp_ssa = np.rad2deg(np.arccos(
            np.dot(fov, sun_dir.position.m / norm(sun_dir.position.m))
        ))

        return nadir_sza, tp_sza, tp_ssa
    except EphemerisRangeError:
        return np.nan, np.nan, np.nan


def local_time(time: Time, lon: float) -> str:
    """Function giving lokal time at longitude

    Arguments:
        time (Time):        skyfield Time to use
        lon (float):    longitude of interest (degrees)

    Returns:
        str:    local time at longitude (ISO 8601 time string)
    """
    try:
        return (
            time.utc_datetime()
            + dt.timedelta(seconds=lon * SECONDS_PER_HOUR / DEGREES_PER_HOUR)
        ).strftime('%H:%M:%S')
    except ValueError:
        return ""
