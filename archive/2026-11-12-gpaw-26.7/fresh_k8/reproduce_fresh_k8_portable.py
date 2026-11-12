#!/usr/bin/env python3
"""Self-contained GaAs response symmetry comparison for GPAW 26.7.0.

This is a finite-grid investigation, not an assertion of a software defect.
Run in an empty directory with GPAW setups installed.
"""

import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import ase
import gpaw
import numpy as np
import scipy
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
from gpaw.bztools import optimal_monkhorst_pack_grid
from gpaw.response.df import DielectricFunction

ROOT = Path.cwd() / 'gpaw_tetra_symmetry_reproducer'
STATUS = ROOT / 'status.json'
FREQUENCIES = {'type': 'nonlinear', 'domega0': 0.005,
               'omega2': 5.0, 'omegamax': 8.0}


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(status):
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(status, indent=2) + '\n')
    temp.replace(STATUS)


def main():
    ROOT.mkdir(exist_ok=False)
    os.chdir(ROOT)
    s = {'state': 'running', 'started_utc': now(),
         'versions': {'gpaw': gpaw.__version__, 'ase': ase.__version__,
                      'numpy': np.__version__, 'scipy': scipy.__version__},
         'source_sha256': sha(__file__),
         'settings': {'xc': 'PBE', 'pw_ecut_eV': 200, 'gs_kpts': [6, 6, 6],
                      'response_kpt_density': 4.0, 'response_bands': 48,
                      'response_ecut_eV': 150, 'eta_eV': 0.05,
                      'intraband': False, 'frequencies': FREQUENCIES},
         'outputs': {}}
    save(s)
    try:
        atoms = bulk('GaAs', 'zincblende', a=5.6533)
        s['stage'] = 'ground_state'
        save(s)
        atoms.calc = GPAW(mode=PW(200), xc='PBE',
                          kpts={'size': (6, 6, 6), 'gamma': True},
                          nbands=48, occupations=FermiDirac(0.01),
                          symmetry='off', txt='gs.txt')
        atoms.get_potential_energy()
        atoms.calc.write('gs.gpw', mode='all')
        s['outputs']['gs_gpw_sha256'] = sha('gs.gpw')
        grid = optimal_monkhorst_pack_grid(
            atoms, kptdensity=4.0, force_even=True, force_gamma=True,
            contains_ibz_vertices=True, nmaxperdim=24)
        s['response_grid'] = {'size': [int(x) for x in grid['size']],
                              'gamma': bool(grid['gamma'])}
        if s['response_grid'] != {'size': [8, 8, 8], 'gamma': True}:
            raise ValueError(f'Unexpected response grid: {grid}')
        save(s)
        for name, sym in (('full', 'off'),
                          ('sym', {'point_group': True, 'time_reversal': True})):
            s['stage'] = f'{name}_fixed_density'
            save(s)
            response = GPAW('gs.gpw', txt=f'{name}_fixed.txt').fixed_density(
                kpts=grid, nbands=48, symmetry=sym)
            response.get_potential_energy()
            response.diagonalize_full_hamiltonian(nbands=48)
            gpw = f'{name}.gpw'
            response.write(gpw, mode='all')
            s['outputs'][f'{name}_gpw_sha256'] = sha(gpw)
            s['outputs'][f'{name}_irreducible_kpoints'] = int(response.wfs.kd.nibzkpts)
            save(s)
            del response
            for mode, integration in (('point', 'point integration'),
                                      ('tetrahedron', 'tetrahedron integration')):
                s['stage'] = f'{name}_{mode}'
                save(s)
                csv = f'{name}_{mode}.csv'
                df = DielectricFunction(
                    calc=gpw, eta=0.05, ecut=150.0, intraband=False,
                    frequencies=FREQUENCIES, integrationmode=integration,
                    qsymmetry=True, txt=f'{name}_{mode}.txt')
                df.get_dielectric_function(filename=csv)
                data = np.loadtxt(csv, delimiter=',')
                if data.shape != (964, 5) or not np.isfinite(data).all():
                    raise ValueError(f'{name} {mode}: invalid array {data.shape}')
                s['outputs'][f'{name}_{mode}'] = {'path': str(ROOT / csv),
                                                  'sha256': sha(csv)}
                save(s)
                del df
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
