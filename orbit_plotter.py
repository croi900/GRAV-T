import os
from pathlib import Path

import h5py
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import cm

from h5utils import create_overwrite_dataset
from plotter import Plotter
from polarizations import compute_phi_r_ode

from roche_geometry import compute_roche_grid, compute_L1_position, roche_potential, roche_lobe_radius_eggleton, roche_coefficients
from nozzle_flow import solve_nozzle_flow_ivp, NozzleFlowParams, compute_isothermal_sound_speed, compute_nozzle_cross_section

from constants import m_H, k_B

def get_corotating_positions(r, m1, m2):
    """
    Compute star positions in the co-rotating (binary) frame.
    Stars are fixed on the x-axis.
    Donor (M1) at negative x, Accretor (M2) at positive x.
    """
    total_mass = m1 + m2
    ratio_1 = m2 / total_mass
    ratio_2 = m1 / total_mass

    # Fixed on X-axis (relative to CoM)
    x1 = -ratio_1 * r
    y1 = np.zeros_like(r)
    x2 = ratio_2 * r
    y2 = np.zeros_like(r)

    return x1, y1, x2, y2


class OrbitPlotter(Plotter):
    def __init__(self, config):
        super().__init__(config)
        self.width = config.width
        self.height = config.height
        self.tail_length = min(config.tail_length, 500)
        
        self.is_hydro = hasattr(config, 'hydro') and (config.decay_type == "hydrodynamics" or config.hydro.R_donor > 0)
        
        self.roche_segs = None
        self.phi_crit = None
        self.donor_poly_norm = None
        
        self.stream_norm_points = None # (N, 2) x,y
        self.stream_norm_widths = None # (N)
        self.stream_densities = None   # (N) for color
        
        # Flow particle system
        self.flow_particles = None      # scatter artist
        self.particle_positions = []    # list of (x_frac, y_offset) tuples
        self.n_particle_rows = 5        # number of rows (2D spread)
        self.particles_per_row = 8      # particles per row
        self.particle_speed = 0.015     # fraction of path per frame

    def _get_video_path(self, run_name):
        video_path = f"{self.config.name}/ode_plots/{run_name}/orbit.mp4"
        directory = Path(video_path).parent
        directory.mkdir(parents=True, exist_ok=True)
        return video_path

    def _compute_roche_geometry(self, q):
        """Pre-compute Roche lobe contours."""
        grid_res = 400
        x = np.linspace(-1.5, 1.5, grid_res)
        y = np.linspace(-1.2, 1.2, grid_res)
        z = compute_roche_grid(q, x[0], x[-1], y[0], y[-1], grid_res, grid_res)
        
        xl1 = compute_L1_position(q)
        phi_crit = roche_potential(xl1, 0, 0, q)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        cs_crit = ax.contour(x, y, z, levels=[phi_crit])
        segs_crit = []
        if hasattr(cs_crit, 'allsegs') and len(cs_crit.allsegs) > 0:
             for path in cs_crit.allsegs[0]:
                 segs_crit.append(path)
        
        plt.close(fig)
        return segs_crit, None, phi_crit, xl1

    def _compute_stream_geometry(self, m1, m2, a, q):
        """
        Compute the 1D nozzle flow profile (density, width) and map to geometry.
        Returns normalized arrays (scaled by a=1).
        """
        R_donor = self.config.hydro.R_donor
        rho_ph = self.config.hydro.rho_ph
        T_ph = self.config.hydro.T_ph
        mu = self.config.hydro.mu
        # Gamma_Edd might depend on context, using default 0.1 for visualization
        Gamma_Edd = 0.8 
        
        L1_x = compute_L1_position(q)
        R_L = roche_lobe_radius_eggleton(q) * a
        B, C = roche_coefficients(q, L1_x)
        
        params = NozzleFlowParams(
            M_donor=m1, M_accretor=m2, a=a, R_donor=R_donor,
            rho_ph=rho_ph, T_ph=T_ph, Gamma_Edd=Gamma_Edd,
            L1_x=L1_x, R_L=R_L, B=B, C=C
        )
        
        # Solve nozzle flow (Photosphere -> L1)
        try:
            x_sol, y_sol, _ = solve_nozzle_flow_ivp(params, n_points=50, mu=mu)

            c_T = compute_isothermal_sound_speed(T_ph, mu)
            c_T_sq = c_T**2
            rho_sol = y_sol[1] / c_T_sq
            
            widths = []
            for val_x in x_sol:
                Q = compute_nozzle_cross_section(val_x, c_T_sq, params)
                widths.append(2.0 * np.sqrt(Q / np.pi)) # Diameter
            widths = np.array(widths)
            
        except Exception as e:
            print(f"Warning: Stream solver check failed ({e}), using fallback geometry.")
            # Fallback linear from Ph to L1
            x_sol = np.linspace(R_donor/a, L1_x, 20)
            rho_sol = np.linspace(rho_ph, rho_ph*0.1, 20)
            widths = np.linspace(R_donor/a * 0.1, R_donor/a * 0.01, 20) * a

        # Extrapolate Outer Stream (L1 -> Accretor Surface)
        x_start_tail = x_sol[-1]
        x_end_tail = 0.95 # Near accretor
        
        n_tail = 30
        x_tail = np.linspace(x_start_tail, x_end_tail, n_tail)

        rho_tail = np.linspace(rho_sol[-1], rho_sol[-1]*0.1, n_tail)
        
        w_tail = np.linspace(widths[-1], widths[-1]*2.0, n_tail)
        
        full_x = np.concatenate([x_sol, x_tail[1:]])
        full_rho = np.concatenate([rho_sol, rho_tail[1:]])
        full_w = np.concatenate([widths, w_tail[1:]]) / a # Normalize width to a
        
        points = np.column_stack((full_x, np.zeros_like(full_x)))
        
        return points, full_w, full_rho

    def _init_particles(self):
        """Initialize particle system with staggered positions."""
        self.particle_positions = []
        n_total = self.n_particle_rows * self.particles_per_row
        
        # Create particles with staggered x positions and spread y offsets
        for row in range(self.n_particle_rows):
            # Y offset normalized [-0.5, 0.5]
            y_offset = (row / (self.n_particle_rows - 1) - 0.5) if self.n_particle_rows > 1 else 0.0
            for col in range(self.particles_per_row):
                # Stagger x positions per row for natural spread
                x_frac = (col + 0.5 * (row % 2)) / self.particles_per_row
                self.particle_positions.append([x_frac, y_offset])
        
        self.particle_positions = np.array(self.particle_positions)

    def _update_particles(self, x_phys, w_phys):
        """
        Update particle positions using velocity field integration.
        
        Particles move from donor (x_frac=0) to accretor (x_frac=1).
        When they reach the end, they respawn at the beginning.
        
        Args:
            x_phys: Physical x coordinates along stream (donor to accretor)
            w_phys: Physical stream widths at each x position
        """
        if len(self.particle_positions) == 0:
            self._init_particles()
        
        # Advance all particles along the path
        self.particle_positions[:, 0] += self.particle_speed
        
        # Respawn particles that reach the end
        overflow_mask = self.particle_positions[:, 0] >= 1.0
        if np.any(overflow_mask):
            # Reset to beginning with slight randomness for natural appearance
            n_overflow = np.sum(overflow_mask)
            self.particle_positions[overflow_mask, 0] = np.random.uniform(0.0, 0.05, n_overflow)
            # Randomize y offset slightly for respawned particles
            for i in np.where(overflow_mask)[0]:
                row = i // self.particles_per_row
                base_y = (row / (self.n_particle_rows - 1) - 0.5) if self.n_particle_rows > 1 else 0.0
                self.particle_positions[i, 1] = base_y + np.random.uniform(-0.05, 0.05)
        
        # Map particle positions to physical coordinates
        t_path = np.linspace(0, 1, len(x_phys))
        
        # Interpolate x position along stream
        particle_x = np.interp(self.particle_positions[:, 0], t_path, x_phys)
        
        # Interpolate stream width and apply y offset
        particle_w = np.interp(self.particle_positions[:, 0], t_path, w_phys)
        particle_y = self.particle_positions[:, 1] * particle_w * 0.8  # 80% of width to stay inside
        
        # Update scatter plot
        self.flow_particles.set_offsets(np.column_stack((particle_x, particle_y)))
        
        # Size particles based on their position (bigger near donor, smaller near accretor)
        sizes = 60 + 40 * (1 - self.particle_positions[:, 0])  # 60-100 range
        self.flow_particles.set_sizes(sizes)
        
        # Color particles based on position (transition from white to orange)
        colors = np.zeros((len(self.particle_positions), 4))
        for i, x_frac in enumerate(self.particle_positions[:, 0]):
            # White -> Yellow -> Orange transition
            colors[i] = (1.0, 1.0 - 0.3*x_frac, 1.0 - 0.8*x_frac, 0.9)
        self.flow_particles.set_facecolors(colors)

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
        
        # Orbital evolution
        phi, r = compute_phi_r_ode(t, a, e, m1 + m2)
        x1, y1, x2, y2 = get_corotating_positions(r, m1, m2)

        if n_frames > 0:
            q_initial = m1[0] / m2[0]
            print(f"Pre-computing geometry for q = {q_initial:.4f}")
            # Modified return signature (removed poly)
            self.roche_segs, _, self.phi_crit, self.xl1_norm = self._compute_roche_geometry(q_initial)
            
            if self.is_hydro:
                try:
                    self.stream_norm_points, self.stream_norm_widths, self.stream_densities = \
                        self._compute_stream_geometry(m1[0], m2[0], a[0], q_initial)
                except Exception as e:
                    print(f"Stream geom failed: {e}")

        with h5py.File(self.h5path, "a") as f:
            for name, data in [("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2), ("phi", phi), ("r", r)]:
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
        self._create_animation(t, a, e, x1, y1, x2, y2, m1, m2, run_name, video_path)

    def _create_animation(self, t, a, e, x1, y1, x2, y2, m1, m2, run_name, video_path):
        n_frames = len(t)
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0a0a0f")
        ax.set_aspect("equal")
        ax.axis("off")

        m1_init = m1[0]
        m2_init = m2[0]
        q_init = m1_init / m2_init
        rl1_norm = roche_lobe_radius_eggleton(q_init)
        rl2_norm = roche_lobe_radius_eggleton(1.0/q_init)

        star1 = Circle((0,0), radius=0, color="#4a9eff", zorder=3, alpha=0.9) # Donor
        star2 = Circle((0,0), radius=0, color="#ff6b4a", zorder=3, alpha=1.0) # Accretor
        ax.add_patch(star1); ax.add_patch(star2)
        
        roche_lines = LineCollection([], colors='#00ffff', linestyles='--', alpha=0.6, linewidths=1.2, zorder=2)
        ax.add_collection(roche_lines)
        
        stream_poly = Polygon([[0,0]], closed=True, facecolor='orange', edgecolor='yellow', 
                              alpha=0.7, linewidth=0.5, zorder=4)
        ax.add_patch(stream_poly)
        
        # Flow Particles (proper injection/integration system)
        self.flow_particles = ax.scatter([], [], s=80, c='white', alpha=0.9, zorder=6, 
                                         marker=(3, 0, 0), edgecolors='orange', linewidths=0.8)
        # Initialize particle system: each particle has (x_frac along path, y_offset in [-0.5, 0.5])
        self._init_particles()

        time_txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, color="white", family="monospace", fontsize=11)
        sep_txt = ax.text(0.02, 0.92, "", transform=ax.transAxes, color="white", family="monospace", fontsize=10)
        da_txt = ax.text(0.02, 0.88, "", transform=ax.transAxes, color="#ff6666", family="monospace", fontsize=10)
        dm_txt = ax.text(0.02, 0.84, "", transform=ax.transAxes, color="#66ff66", family="monospace", fontsize=10)
        
        title = ax.text(0.5, 0.98, f"Simulation: {run_name}", transform=ax.transAxes, 
                        color="white", ha="center", va="top", fontsize=14, fontweight="bold")

        # Mass Depletion Bar (Bottom left)
        bar_x, bar_y = 0.02, 0.05
        bar_w, bar_h = 0.15, 0.02
        mass_bar_bg = plt.Rectangle((bar_x, bar_y), bar_w, bar_h, transform=ax.transAxes,
                                     facecolor='#333333', edgecolor='white', linewidth=1, zorder=10)
        mass_bar_fill = plt.Rectangle((bar_x, bar_y), bar_w, bar_h, transform=ax.transAxes,
                                       facecolor='#4a9eff', edgecolor='none', zorder=11)
        ax.add_patch(mass_bar_bg)
        ax.add_patch(mass_bar_fill)
        ax.text(bar_x + bar_w/2, bar_y + bar_h + 0.01, "Donor Mass", 
                transform=ax.transAxes, ha='center', va='bottom', color='white', fontsize=9)

        handles = [
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#4a9eff', markersize=8, lw=0, label='Donor'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff6b4a', markersize=8, lw=0, label='Accretor'),
            Line2D([0],[0], color='#00ffff', linestyle='--', lw=1.5, label='Roche Lobe'),
        ]
        if self.is_hydro:
             handles.append(plt.Rectangle((0,0), 1, 1, facecolor='orange', edgecolor='yellow', alpha=0.7, label='Mass Stream'))
             handles.append(Line2D([0],[0], marker='o', color='white', lw=0, markersize=5, label='Flow Tracer'))
        ax.legend(handles=handles, loc="upper right", fontsize=9)

        x_cm_roche = 1.0 / (1.0 + q_init)
        curr_lim = [a[0] * 2.0]

        def init():
            star1.center = (0,0); star1.set_radius(0)
            star2.center = (0,0); star2.set_radius(0)
            roche_lines.set_segments([])
            stream_poly.set_xy([[0,0]])
            self.flow_particles.set_offsets(np.zeros((0,2)))
            return star1, star2, roche_lines, stream_poly, self.flow_particles

        def update(frame):
            curr_a = a[frame]
            curr_m1 = m1[frame]
            curr_m2 = m2[frame]
            dt = t[frame] - t[max(0, frame-1)]
            da = (a[frame] - a[max(0, frame-1)]) / dt if dt > 0 else 0.0
            dm1 = (m1[frame] - m1[max(0, frame-1)]) / dt if dt > 0 else 0.0
            
            cx1, cy1 = x1[frame], y1[frame]
            cx2, cy2 = x2[frame], y2[frame]
            
            # Dynamic Radius Clamping
            rl1_phys = rl1_norm * curr_a
            rl2_phys = rl2_norm * curr_a
            
            R1_phys = self.config.hydro.R_donor if self.is_hydro else 1.5e4
            R1_vis = min(R1_phys, rl1_phys * 0.98)
            R2_vis = rl2_phys * 0.6
            
            # Update star positions and sizes
            star1.center = (cx1, cy1); star1.set_radius(R1_vis)
            star2.center = (cx2, cy2); star2.set_radius(R2_vis)
            
            # Fade donor color based on mass remaining
            mass_frac = curr_m1 / m1_init
            donor_color = (0.29 + 0.1*(1-mass_frac), 0.62 * mass_frac, 1.0 * mass_frac)  # Fade to red/orange
            star1.set_facecolor(donor_color)
            
            # Zoom
            target = curr_a * 1.35 
            curr_lim[0] = curr_lim[0]*0.9 + target*0.1
            ax.set_xlim(-curr_lim[0], curr_lim[0])
            ax.set_ylim(-curr_lim[0]*9/16, curr_lim[0]*9/16)
            
            # Roche Lobe
            if self.roche_segs:
                segs = []
                for s in self.roche_segs:
                    phys = np.column_stack(((s[:,0]-x_cm_roche)*curr_a, s[:,1]*curr_a))
                    segs.append(phys)
                roche_lines.set_segments(segs)
            
            # 2D Stream (Filled Polygon with width)
            if self.is_hydro:
                try:
                    curr_q = curr_m1 / curr_m2
                    pts, widths, densities = self._compute_stream_geometry(curr_m1, curr_m2, curr_a, curr_q)
                    
                    if pts is not None and len(pts) > 2:
                        curr_x_cm = 1.0 / (1.0 + curr_q)
                        # Scale pts and widths to physical
                        x_phys = (pts[:,0] - curr_x_cm) * curr_a
                        w_phys = widths * curr_a * 3.0  # Amplify width for visibility
                        
                        # Build polygon: upper edge, then lower edge reversed
                        upper_y = w_phys / 2.0
                        lower_y = -w_phys / 2.0
                        
                        poly_x = np.concatenate([x_phys, x_phys[::-1]])
                        poly_y = np.concatenate([upper_y, lower_y[::-1]])
                        poly_pts = np.column_stack((poly_x, poly_y))
                        
                        stream_poly.set_xy(poly_pts)
                        
                        # Color intensity based on density/mass rate
                        # Normalize to make changes visible
                        max_rho = max(densities) if len(densities) > 0 else 1.0
                        avg_rho = np.mean(densities)
                        intensity = min(1.0, avg_rho / (self.config.hydro.rho_ph * 0.5))
                        stream_poly.set_facecolor((1.0, 0.5 + 0.5*intensity, 0.0, 0.5 + 0.4*intensity))
                        
                        # Flow Particles (proper injection + integration)
                        if len(x_phys) > 2:
                            self._update_particles(x_phys, w_phys)
                    else:
                        stream_poly.set_xy([[0,0]])
                        self.flow_particles.set_offsets(np.zeros((0, 2)))
                
                except Exception as e:
                    stream_poly.set_xy([[0,0]])
                    self.flow_particles.set_offsets(np.zeros((0, 2)))
            
            # Update Mass Bar
            mass_frac = curr_m1 / m1_init
            mass_bar_fill.set_width(bar_w * mass_frac)
            
            # Update Labels (Clear, separate lines)
            time_txt.set_text(f"Time: {t[frame]:.2e} s")
            sep_txt.set_text(f"Separation: {curr_a:.3e} m")
            da_txt.set_text(f"da/dt: {da:.2e} m/s")
            dm_txt.set_text(f"dM₁/dt: {dm1:.2e} M☉/s")
            
            return star1, star2, roche_lines, stream_poly, self.flow_particles, mass_bar_fill

        anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=1000/self.config.fps)
        writer = FFMpegWriter(fps=self.config.fps, bitrate=5000)
        with tqdm.tqdm(total=n_frames, desc="Rendering") as pbar:
            anim.save(video_path, writer=writer, progress_callback=lambda i, n: pbar.update(1) if i > 0 else None)
        plt.close(fig)
        print(f"Saved: {video_path}")

