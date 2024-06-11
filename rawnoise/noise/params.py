import os
import json
from dataclasses import dataclass
from collections import namedtuple

@dataclass
class GainParams:
    Kmin: float
    Kmax: float


@dataclass
class ScaleParams:
    slope: float
    sigma: float
    bias: float


@dataclass
class TukeyParams:
    scale: ScaleParams
    shape: list[float]
    color_bias: list[tuple[float, float, float, float]]



def load_gain_params(params: dict) -> GainParams:
    return GainParams(Kmin=params['Kmin'], Kmax=params['Kmax'])


def load_tukey_params(params: dict) -> TukeyParams:
    return TukeyParams(
        ScaleParams(
            slope=params['Profile-1']['G_scale']['slope'],
            sigma=params['Profile-1']['G_scale']['sigma'],
            bias=params['Profile-1']['G_scale']['bias']
        ),
        shape=params['G_shape'],
        color_bias=params['color_bias']
    )

def load_params_from_json(file: str | os.PathLike):
    with open(file, 'r') as jf:
        params = json.load(jf)

    tukey_params = load_tukey_params(params)
    gain_params = load_gain_params(params)

    return {
        'tukey': tukey_params,
        'gain': gain_params
    }
