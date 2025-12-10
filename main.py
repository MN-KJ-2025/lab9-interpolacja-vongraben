# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np


def chebyshev_nodes(n: int = 10) -> np.ndarray | None:
    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    i sortująca wynik od najmniejszego do największego węzła.

    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n < 2:
        return None

    k = np.arange(n, dtype=float)
    nodes = np.cos(np.pi * k / (n - 1))
    return nodes


def bar_cheb_weights(n: int = 10) -> np.ndarray | None:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n < 2:
        return None

    weights = (-1.0) ** np.arange(n, dtype=float)
    weights[0] = 0.5
    weights[-1] = 0.5 * ((-1.0) ** (n - 1))
    return weights


def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> np.ndarray | None:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        xi = np.asarray(xi, dtype=float)
        yi = np.asarray(yi, dtype=float)
        wi = np.asarray(wi, dtype=float)
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None

    if (
        xi.ndim != 1
        or yi.ndim != 1
        or wi.ndim != 1
        or x.ndim != 1
        or len(xi) != len(yi)
        or len(xi) != len(wi)
        or len(xi) < 2
    ):
        return None

    result = np.empty_like(x, dtype=float)

    for idx, xv in enumerate(x):
        diff = xv - xi
        zero_mask = np.isclose(diff, 0.0)

        if np.any(zero_mask):
            result[idx] = yi[zero_mask][0]
            continue

        tmp = wi / diff
        result[idx] = np.sum(tmp * yi) / np.sum(tmp)

    return result


def L_inf(
    xr: int | float | list | np.ndarray, x: int | float | list | np.ndarray
) -> float | None:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        xr_arr = np.asarray(xr, dtype=float)
        x_arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None

    if xr_arr.shape != x_arr.shape:
        return None

    diff = np.abs(xr_arr - x_arr)
    return float(np.max(diff))
