# Repository agent instructions

## Python environment

- Always use the dedicated Conda environment named `car_track` for Python,
  notebook, dependency, and test commands in this repository.
- Environment location: `C:\Users\11320\anaconda3\envs\car_track`.
- Prefer the environment's executable directly so commands do not accidentally
  use the base environment:
  `C:\Users\11320\anaconda3\envs\car_track\python.exe`.
- Interactive users may instead run `conda activate car_track` before working.
- Do not install project packages into the base Conda environment. Update
  `requirements.txt` when dependencies change, then install them into
  `car_track`.

## Standard checks

- Install or refresh dependencies with:
  `C:\Users\11320\anaconda3\envs\car_track\python.exe -m pip install -r requirements.txt`
- Run tests with:
  `C:\Users\11320\anaconda3\envs\car_track\python.exe -m pytest -q`
- Before executing a notebook, confirm its selected kernel/interpreter resolves
  to `C:\Users\11320\anaconda3\envs\car_track\python.exe`.
- If a sandboxed Matplotlib command cannot write its user-level font cache, set
  `MPLCONFIGDIR` to a writable temporary directory for that command.
