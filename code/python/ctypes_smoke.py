from gfe_ctypes import fit_soe_kernel, load_gfe_library


def main() -> None:
    lib = load_gfe_library()
    t = [0.1 * i for i in range(81)]
    y = [0.65 * (2.718281828459045 ** (-0.8 * x)) + 0.35 * (2.718281828459045 ** (-0.12 * x)) for x in t]
    result = fit_soe_kernel(lib, t, y)
    print("python ctypes smoke ok")
    print(f"modes={len(result['gamma_fit'])} fit_len={len(result['fit'])}")


if __name__ == "__main__":
    main()
