from tqdm import tqdm

import h5utils
from config import State, Config
from name_maps import domain_type_map, integrator_map, system_map, decay_map
import h5py
import numpy as np
from verbprint import vprint

"""
Domain types: uniform, exponential
"""

STORAGE_CHUNK_SIZE = 1024 * 1024 * 1024 // 8 // 5


class IntegrationRun:
    def __init__(
        self,
        name,
        config: Config,
        t0,
        t1,
        t_eval=None,
        decay_type="linear",
        solver="Radau",
    ):
        self.name = name
        self.config = config
        self.state = self.config.state

        if t_eval is None:
            self.t_eval = np.linspace(t0, t1, config.output_points)
        else:
            self.t_eval = t_eval

        self.decay_function = decay_map(decay_type, self.state.decay_rate)
        self.system = lambda t, y: system_map(decay_type)(
            t, y, self.state.M1, self.state.M2, self.state.decay_rate
        )
        self.t0 = t0
        self.t1 = t1
        self.writer = h5py.File(f"{self.config.name}/{self.config.name}.h5", "a")
        self.solver = integrator_map(solver)(
            fun=self.system,
            t0=self.t0,
            t_bound=self.t1,
            y0=[self.state.a, self.state.e],
            rtol=1e-9,
            atol=1e-12,
            dense_output=True,
        )

    def run(self):
        with self.writer as writer:
            vprint(f"creating datasets in h5 {self.config.name}: {self.name}/.... ")

            times_ds = h5utils.create_overwrite_dataset(
                writer,
                f"{self.name}/times",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=True,
                compression="gzip",
            )
            a_ds = h5utils.create_overwrite_dataset(
                writer,
                f"{self.name}/a",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=True,
                compression="gzip",
            )
            e_ds = h5utils.create_overwrite_dataset(
                writer,
                f"{self.name}/e",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=True,
                compression="gzip",
            )
            m1_ds = h5utils.create_overwrite_dataset(
                writer,
                f"{self.name}/m1",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=True,
                compression="gzip",
            )
            m2_ds = h5utils.create_overwrite_dataset(
                writer,
                f"{self.name}/m2",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=True,
                compression="gzip",
            )

            vprint("done creating datasets!")
            vprint("creating buffers and t_eval domain")

            a_buffer = np.zeros(STORAGE_CHUNK_SIZE, dtype="f8")
            e_buffer = np.zeros(STORAGE_CHUNK_SIZE, dtype="f8")
            m1_buffer = np.zeros(STORAGE_CHUNK_SIZE, dtype="f8")
            m2_buffer = np.zeros(STORAGE_CHUNK_SIZE, dtype="f8")
            t_buffer = np.zeros(STORAGE_CHUNK_SIZE, dtype="f8")

            t_eval = self.t_eval

            vprint("done creating buffers and t_eval domain!")
            vprint(
                f"t_eval: {t_eval}, t0 = {self.t0}, t1 = {self.t1}, len={t_eval.shape}"
            )

            buffer_ptr = 0
            t_eval_ptr = 0

            def mass_scale_ok(t):
                return self.decay_function.value(t) > 1e-6

            def flush_buffers():
                nonlocal buffer_ptr

                def flush_buffer(buffer, data):
                    cursize = buffer.shape[0]
                    new_size = cursize + buffer_ptr
                    buffer.resize(new_size, axis=0)
                    buffer[cursize:new_size] = data[:buffer_ptr]

                flush_buffer(a_ds, np.clip(a_buffer, 0, np.inf))
                flush_buffer(e_ds, np.clip(e_buffer, 1e-18, 1))
                flush_buffer(m1_ds, m1_buffer)
                flush_buffer(m2_ds, m2_buffer)
                flush_buffer(times_ds, t_buffer)

            with tqdm(total=t_eval.shape[0], desc=f"{self.name} ODE: ") as pbar:
                while self.solver.t < self.t1:
                    if not mass_scale_ok(self.solver.t):
                        vprint("mass scaling below threshold, stopping integration")
                        break

                    if t_eval_ptr >= len(t_eval):
                        vprint("t_eval outside of t_eval, stopping integration")
                        break

                    t_target = t_eval[t_eval_ptr]

                    self.solver.step()
                    vprint(
                        f"t: {self.solver.t}, t_arget={t_target}, y = {self.solver.y}"
                    )
                    while t_target <= self.solver.t:
                        t_target = t_eval[t_eval_ptr]
                        istate = self.solver.dense_output()(t_target)

                        t_buffer[buffer_ptr] = t_target
                        a_buffer[buffer_ptr] = istate[0]
                        e_buffer[buffer_ptr] = istate[1]
                        m1_buffer[buffer_ptr] = (
                            self.state.M1 * self.decay_function.value(t_target)
                        )
                        m2_buffer[buffer_ptr] = (
                            self.state.M2 * self.decay_function.value(t_target)
                        )
                        pbar.update(1)

                        t_eval_ptr += 1
                        buffer_ptr += 1

                        if buffer_ptr >= STORAGE_CHUNK_SIZE:
                            vprint("flushing buffers!")
                            flush_buffers()
                            buffer_ptr = 0
                            vprint("done flushing buffers!")

                        if t_eval_ptr >= len(t_eval):
                            vprint("t_eval outside of t_eval, stopping integration")
                            break

                    if self.solver.status != "running" and self.solver.t < self.t1:
                        vprint(f"solver oopsie, reason: {self.solver.status}")
                        vprint(
                            f"computed untill t={self.solver.t}, stored = {t_eval_ptr}"
                        )
                        break

            flush_buffers()

            return State.from_si(
                self.state.M1 * self.decay_function.value(self.solver.t),
                self.state.M2 * self.decay_function.value(self.solver.t),
                self.solver.y[0],
                self.solver.y[1],
                self.state.decay_rate,
            )
