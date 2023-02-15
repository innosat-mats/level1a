from typing import Tuple
import datetime as dt

from numpy.linalg import norm
from skyfield.api import load, wgs84, Time
from skyfield.positionlib import Geocentric, ICRF
from skyfield.framelib import itrs
from skyfield.units import Distance
import numpy as np

DEGREES_PER_HOUR = 15
SECONDS_PER_HOUR = 3600


def satellite_position(
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
    sat_pos = Geocentric(position_au=Distance(m=eci_pos).au, t=time)
    lat, lon, alt = sat_pos.frame_latlon(itrs)
    return lat.degrees, lon.degrees, alt.m


def tangent_point_position(
    time: Time,
    eci: np.ndarray,
):
    """
    Function giving the GPS Tangent Point in lat/lon/alt/.
   
    Arguments:
        time (Time):    skyfield Time to use
        eci (ndarray):  tangent point ECI

    Returns:
        float:  latitude of satellite (degrees)
        float:  longitude of satellite (degrees)
        float:  altitude (meters)
    """
    tp_pos = Geocentric(position_au=Distance(m=eci).au, t=time)
    lat, lon, alt = tp_pos.frame_latlon(itrs)
    return lat.degrees, lon.degrees, alt.m


def angles(
    time: Time,
    sat_lat: float,
    sat_lon: float,
    sat_alt: float,
    tp_lat: float,
    tp_lon: float,
    tp_alt: float,
) -> Tuple[float, float, float, str]:
    """
    Function giving various angles.
   
    Arguments:
        time (Time):        skyfield Time to use
        sat_lat (float):    latitude of satellite (degrees)
        sat_lon (float):    longitude of satellite (degrees)
        sat_alt (float):    altitude (meters)
        tp_lat (float):     latitude of satellite (degrees)
        tp_lon (float):     longitude of satellite (degrees)
        tp_alt (float):     altitude (metres)

    Returns:
        float:  solar zenith angle at satelite position (degrees)
        float:  solar zenith angle at TP position (degrees)
        float:  solar scattering angle at TP position (degrees)
        str:    local time at the TP (ISO 8601 format time)
    """
    planets = load('de421.bsp')
    earth, sun = planets['earth'], planets['sun']
   
    sat_pos = earth + wgs84.latlon(sat_lat, sat_lon, elevation_m=sat_alt)
    sun_dir = sat_pos.at(time).observe(sun).apparent()
    obs = sun_dir.altaz()
    nadir_sza = (90 - obs[0].degrees)

    tp_pos = earth + wgs84.latlon(tp_lat, tp_lon, elevation_m=tp_alt)
    tp_lt = (
        time.utc_datetime()
        + dt.timedelta(seconds=tp_lon * SECONDS_PER_HOUR / DEGREES_PER_HOUR)
    ).strftime('%H:%M:%S')

    fov = (tp_pos - sat_pos).at(time).position.m
    fov = fov / norm(fov)
    sun_dir = tp_pos.at(t).observe(sun).apparent()
    obs = sun_dir.altaz()
    tp_sza = 90 - obs[0].degrees
    tp_ssa = np.rad2deg(np.arccos(
        np.dot(fov, sun_dir.position.m / norm(sun_dir.position.m))
    ))

    return nadir_sza, tp_sza, tp_ssa, tp_lt
