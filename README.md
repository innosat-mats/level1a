# MATS satellite level 0 to 1a code

## Deployments status
[![DOI](https://zenodo.org/badge/565748106.svg)](https://zenodo.org/badge/latestdoi/565748106)
![CI](https://github.com/innosat-mats/level1a/actions/workflows/ci.yml/badge.svg?branch=deploy)
![CD](https://github.com/innosat-mats/level1a/actions/workflows/cd.yml/badge.svg)


## Overview
This code is used to generate the level 1a data from the level 0 data of the MATS satellite mission.

## Prerequisites
The code is supposed to run in a AWS (Amazon web services) environment and
depends on other componets that downloads satellite data from its source.
Files that are supposed to be processed are read from a queue.

## Language
Python 3.9

## File description
A processing activity starts in:
- level1a/handlers/level1a.py
  - lambda_handler function


## References
Megner, L., Gumbel, J., Christensen, O. M., Linder, B., Murtagh, D. P., Ivchenko, N., Krasauskas, L., Hedin, J., Dillner, J., Giono, G., Olentsenko, G., Kern, L., and Stegman, J.: The MATS satellite: Limb image data processing and calibration, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2025-265, 2025.

## Contact information
L. Megner, Department of Meteorology, Stockholm University, SE-106 91 Stockholm, Sweden, linda.megner@misu.su.se



