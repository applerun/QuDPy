def make_absorption_runner(
    *,
    number_density_m3: float,
    window: str | None = "hann",
    subtract_mean: bool = True,
    rel_threshold: float = 1e-6,
    zero_padding_factor: int = 4,
):
    def absorption_runner(*, delay_case, readout_result, readout_window):
        # 1. 从 readout_result 截取 readout_window
        t_fs = np.asarray(readout_result.times_fs, dtype=float)
        mask = (
            (t_fs >= float(readout_window.start_fs))
            & (t_fs <= float(readout_window.end_fs))
        )

        t_readout = t_fs[mask]

        # 2. 取 readout 对应的输入场
        physical = readout_result.physical_params
        if physical is None:
            raise ValueError("readout_result.physical_params is required.")

        E_readout = np.asarray(physical.field(t_readout), dtype=float)

        # 3. 从 density matrix 计算 polarization
        P_readout = polarization_C_per_m2(
            readout_result.density_array()[mask],
            physical.dipole_matrix_D,
            float(number_density_m3),
        )

        # 4. 取一个 coherence，具体 pair 你可以改成参数
        rhoij_readout = readout_result.matrix_element(0, 1)[mask]

        # 5. 调谱学函数
        return lab_frame_absorption_response(
            t_fs=t_readout,
            E_MV_per_cm=E_readout,
            P_C_per_m2=P_readout,
            rhoij=rhoij_readout,
            window=window,
            subtract_mean=subtract_mean,
            rel_threshold=rel_threshold,
            zero_padding_factor=zero_padding_factor,
        )

    return absorption_runner