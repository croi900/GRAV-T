import os
from pathlib import Path

import h5py
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from h5utils import create_overwrite_dataset
from plotter import Plotter
from polarizations import compute_phi_r_ode


def polar_to_cartesian(phi, r, m1, m2):
    """
    Convert polar coordinates (phi, r) to Cartesian positions for each star.
    Uses center-of-mass frame.
    """

    x_rel = r * np.cos(phi)
    y_rel = r * np.sin(phi)

    total_mass = m1 + m2
    ratio_1 = m2 / total_mass
    ratio_2 = m1 / total_mass

    x1 = -ratio_1 * x_rel
    y1 = -ratio_1 * y_rel
    x2 = ratio_2 * x_rel
    y2 = ratio_2 * y_rel

    return x1, y1, x2, y2


class OrbitPlotter(Plotter):
    def __init__(self, config):
        super().__init__(config)
        self.width = config.width
        self.height = config.height
        self.tail_length = min(config.tail_length, 500)

    def _get_video_path(self, run_name):
        video_path = f"{self.config.name}/ode_plots/{run_name}/orbit.mp4"
        directory = Path(video_path).parent
        directory.mkdir(parents=True, exist_ok=True)
        return video_path

    def plot(self, run_name):
        with h5py.File(self.h5path, "r") as f:
            t_full = np.array(f[f"{run_name}/times"])
            a_full = np.array(f[f"{run_name}/a"])
            e_full = np.array(f[f"{run_name}/e"])
            m1_full = np.array(f[f"{run_name}/m1"])
            m2_full = np.array(f[f"{run_name}/m2"])

        target_frames = 60 * self.config.fps
        stride = max(1, len(t_full) // target_frames)

        t = t_full[::stride]
        a = a_full[::stride]
        e = e_full[::stride]
        m1 = m1_full[::stride]
        m2 = m2_full[::stride]

        n_frames = len(t)
        print(
            f"Generating {n_frames} frames from {len(t_full)} data points (stride={stride})"
        )

        print("Computing orbital phase evolution...")
        phi, r = compute_phi_r_ode(t, a, e, m1 + m2)

        x1, y1, x2, y2 = polar_to_cartesian(phi, r, m1, m2)

        with h5py.File(self.h5path, "a") as f:
            for name, data in [
                ("x1", x1),
                ("y1", y1),
                ("x2", x2),
                ("y2", y2),
                ("phi", phi),
                ("r", r),
            ]:
                ds = create_overwrite_dataset(
                    f,
                    f"{run_name}/{name}",
                    shape=(len(data),),
                    maxshape=(None,),
                    dtype="f8",
                    chunks=True,
                    compression="gzip",
                )
                ds[:] = data

        video_path = self._get_video_path(run_name)
        print(f"Orbit animation will be saved to: {video_path}")
        self._create_animation(t, a, e, x1, y1, x2, y2, run_name, video_path)

    def _create_animation(self, t, a, e, x1, y1, x2, y2, run_name, video_path):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0a0a0f")

        ax.set_aspect("equal")
        ax.axis("off")

        star1 = ax.scatter(
            [],
            [],
            c="#4a9eff",
            s=200,
            zorder=10,
            label="Star 1",
            edgecolors="white",
            linewidths=0.5,
        )
        star2 = ax.scatter(
            [],
            [],
            c="#ff6b4a",
            s=200,
            zorder=10,
            label="Star 2",
            edgecolors="white",
            linewidths=0.5,
        )

        com = ax.scatter(
            [0], [0], c="#ffffff", s=30, marker="+", alpha=0.5, zorder=5, label="CoM"
        )

        (trail1,) = ax.plot([], [], color="#4a9eff", alpha=0.4, linewidth=1.5, zorder=3)
        (trail2,) = ax.plot([], [], color="#ff6b4a", alpha=0.4, linewidth=1.5, zorder=3)

        time_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=12,
            color="white",
            family="monospace",
            verticalalignment="top",
            alpha=0.9,
        )
        params_text = ax.text(
            0.02,
            0.02,
            "",
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            family="monospace",
            verticalalignment="bottom",
            alpha=0.7,
        )
        scale_text = ax.text(
            0.98,
            0.02,
            "",
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            family="monospace",
            verticalalignment="bottom",
            horizontalalignment="right",
            alpha=0.7,
        )

        title_text = ax.text(
            0.5,
            0.98,
            f"Binary Inspiral: {run_name}",
            transform=ax.transAxes,
            fontsize=14,
            color="white",
            horizontalalignment="center",
            verticalalignment="top",
            alpha=0.9,
            fontweight="bold",
        )

        ax.legend(loc="upper right", framealpha=0.3, fontsize=9)

        max_extent = max(
            np.max(np.abs(x1)),
            np.max(np.abs(y1)),
            np.max(np.abs(x2)),
            np.max(np.abs(y2)),
        )

        current_limit = [max_extent * 3.0]

        n_frames = len(t)
        tail_len = self.tail_length

        def init():
            star1.set_offsets(np.empty((0, 2)))
            star2.set_offsets(np.empty((0, 2)))
            trail1.set_data([], [])
            trail2.set_data([], [])
            return star1, star2, trail1, trail2, time_text, params_text, scale_text

        def update(frame):
            curr_x1, curr_y1 = x1[frame], y1[frame]
            curr_x2, curr_y2 = x2[frame], y2[frame]

            star1.set_offsets([[curr_x1, curr_y1]])
            star2.set_offsets([[curr_x2, curr_y2]])

            start_idx = max(0, frame - tail_len)
            trail1.set_data(x1[start_idx : frame + 1], y1[start_idx : frame + 1])
            trail2.set_data(x2[start_idx : frame + 1], y2[start_idx : frame + 1])

            max_dist = max(abs(curr_x1), abs(curr_y1), abs(curr_x2), abs(curr_y2))
            target_limit = max(max_dist * 3.0, 1e3)

            alpha = 0.02
            current_limit[0] = current_limit[0] * (1 - alpha) + target_limit * alpha

            limit = current_limit[0]
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit * 9 / 16, limit * 9 / 16)

            time_text.set_text(f"Time: {t[frame]:.20e} s")
            params_text.set_text(f"a = {a[frame]:.20e} m  |  e = {e[frame]:.20f}")

            if limit > 1e9:
                scale_str = f"{limit / 1e9:.2f} Gm"
            elif limit > 1e6:
                scale_str = f"{limit / 1e6:.2f} Mm"
            elif limit > 1e3:
                scale_str = f"{limit / 1e3:.2f} km"
            else:
                scale_str = f"{limit:.2f} m"
            scale_text.set_text(f"Scale: ±{scale_str}")

            return star1, star2, trail1, trail2, time_text, params_text, scale_text

        print(f"Creating animation with {n_frames} frames...")

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=n_frames,
            blit=False,
            interval=1000 / self.config.fps,
        )

        print(f"Saving to {video_path}...")

        from matplotlib.animation import FFMpegWriter

        writer = FFMpegWriter(
            fps=self.config.fps, metadata=dict(artist="GW Simulator"), bitrate=5000
        )

        with tqdm.tqdm(total=n_frames, desc="Rendering") as pbar:
            anim.save(
                video_path,
                writer=writer,
                progress_callback=lambda i, n: pbar.update(1) if i > 0 else None,
            )

        plt.close(fig)
        print(f"Animation saved to {video_path}")
