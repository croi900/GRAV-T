from config import M_SUN


class ETPreprocessor:

    def __init__(self, config):
        self.config = config
        self.h5path = f"{config.name}/{config.name}.h5"

    def preprocess():
        with h5py.File(self.h5path, "r") as f:
            a_ds = np.array(f[f"{run_name}/a"])
            e_ds = np.array(f[f"{run_name}/e"])
            t_ds = np.array(f[f"{run_name}/times"])
            m1_ds = np.array(f[f"{run_name}/m1"])
            m2_ds = np.array(f[f"{run_name}/m2"])
            final_idx = -1
        a_final_au = a_data[final_idx]
        m1_final_msun = m1_data[final_idx]
        m2_final_msun = m2_data[final_idx]
        e_final = e_data[final_idx]
        M_tot_kg = m1_final_kg + m2_final_kg
        omega_rad_per_sec = np.sqrt(G * M_tot_kg / a_final_m**3)
        time_unit_geom = G * M_SUN / c**3
        omega_geom = omega_rad_per_sec * time_unit_geom
        m1_final_msun = m1_final_kg / M_SUN
        m2_final_msun = m2_final_kg / M_SUN
        a_final_km = a_final_m / 1000.0
        print(
            f"GRAV-T Final State -> M1: {m1_final_msun:.3f}, M2: {m2_final_msun:.3f}, a: {a_final_km:.2f} km"
        )
        print(f"Target Omega (Geom): {omega_geom:.8e}")
        ini_content = f'\n\n\n\n\n[EquationOfState]\nType = Cold_PWPoly\nEosFile = "sly_piecewise.eos" \nh_cut = 0.0\n\n[BinaryParameters]\n\nTargetOmega = {omega_geom:.8e}\n\nInitialSeparationGuess_km = {a_final_km:.4f}\n\n[Star1]\nTargetMass = {m1_final_msun:.6f}\n\nSpin = 0.0 \n\n[Star2]\nTargetMass = {m2_final_msun:.6f}\nSpin = 0.0\n\n[KadathGrid]\n\nResolution_Radial = 33\nResolution_Theta = 21\nResolution_Phi = 20\n\n[Output]\n\nFilename = FUKA_InitialData.h5\nFormat = HDF5\n'
        with open(output_file, "w") as f_ini:
            f_ini.write(ini_content)
        print(f"FUKA configuration successfully written to '{output_file}'!")
