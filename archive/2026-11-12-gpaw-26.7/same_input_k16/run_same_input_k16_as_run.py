#!/usr/bin/env python3
"""Run GPAW 26.7.0 response on the archived, hash-verified 16^3 GaAs GPWs."""

import gc
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import gpaw
import numpy as np
from gpaw.response.df import DielectricFunction

OLD = Path('/Users/dela/sbup3-dielectric-pilot-20261111')
CASE = Path('/Users/dela/sbup3-gpaw-26.7-20261112/same_input_k16')
STATUS = CASE / 'status.json'
GPWS = {
    'full': OLD / 'tetra_k16/gaas_response_gamma_k16_b48.gpw',
    'sym': OLD / 'tetra_k16_symmetry/gaas_response_gamma_k16_sym_b48.gpw',
}
FREQUENCIES = {'type': 'nonlinear', 'domega0': 0.005,
               'omega2': 5.0, 'omegamax': 8.0}


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(status):
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(status, indent=2) + '\n')
    tmp.replace(STATUS)


def main():
    CASE.mkdir(exist_ok=False)
    old_full = json.loads((OLD / 'tetra_k16/status.json').read_text())
    old_sym = json.loads((OLD / 'tetra_k16_symmetry/status.json').read_text())
    if old_full['state'] != 'completed' or old_sym['state'] != 'stopped_numerical_mismatch':
        raise ValueError('Prior 16-grid status is unexpected')
    old_hashes = {'full': old_full['response_gpw_sha256'],
                  'sym': old_sym['response_gpw_sha256']}
    for key, path in GPWS.items():
        if sha(path) != old_hashes[key]:
            raise ValueError(f'{key} GPW hash mismatch')
    s = {'state': 'running', 'started_utc': now(),
         'gpaw_version': gpaw.__version__,
         'source_sha256': sha(__file__),
         'input_gpw_sha256': old_hashes,
         'settings': {'mesh': [16, 16, 16], 'bands': 48, 'eta_eV': 0.05,
                      'ecut_eV': 150, 'intraband': False,
                      'qsymmetry': True, 'frequencies': FREQUENCIES},
         'results': {}}
    save(s)
    try:
        os.chdir(CASE)
        for key in ('full', 'sym'):
            for mode, integration in (('point', 'point integration'),
                                      ('tetrahedron', 'tetrahedron integration')):
                s['stage'] = f'{key}_{mode}'
                save(s)
                output = CASE / f'k16_{key}_{mode}_gpaw267.csv'
                df = DielectricFunction(calc=str(GPWS[key]), eta=0.05,
                                        ecut=150.0, intraband=False,
                                        frequencies=FREQUENCIES,
                                        integrationmode=integration,
                                        qsymmetry=True)
                df.get_dielectric_function(filename=str(output))
                data = np.loadtxt(output, delimiter=',')
                if data.shape != (964, 5) or not np.isfinite(data).all():
                    raise ValueError(f'{key}_{mode}: invalid {data.shape} spectrum')
                s['results'][f'{key}_{mode}'] = {
                    'path': str(output), 'sha256': sha(output),
                    'shape': list(data.shape),
                    'energy_min_eV': float(data[:, 0].min()),
                    'energy_max_eV': float(data[:, 0].max())}
                save(s)
                del df
                gc.collect()
        s['state'] = 'completed'
        s.pop('stage', None)
    except Exception as e:
        s['state'] = 'failed'
        s['error'] = repr(e)
        s['traceback'] = traceback.format_exc()
        raise
    finally:
        s['finished_utc'] = now()
        save(s)


if __name__ == '__main__':
    main()
