#!/usr/bin/env python3
"""
ATHENA Wavelet-Enhanced Feature Engineering

Modification 2 vs ARTEMIS: Adds wavelet decomposition features on top of
the existing 98+ hand-crafted technical indicators. Wavelet decomposition
separates price signals into trend, cyclical, and noise components at
multiple scales, capturing multi-scale dynamics that traditional
indicators miss.

Reference: Ramsey & Lampart (1998), "The decomposition of economic
relationships by time scale using wavelets"
"""

import numpy as np
import pandas as pd

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


class WaveletFeatureExtractor:
    """
    Extracts wavelet-based features from price and return series.

    Uses the Daubechies-4 (db4) wavelet at 3 decomposition levels to produce:
      - Approximation (trend) and detail (cyclical) coefficients
      - Per-scale energy ratios
      - Cross-scale correlation features
      - Wavelet-derived volatility estimates
    """

    def __init__(self, wavelet='db4', levels=3):
        self.wavelet = wavelet
        self.levels = levels

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add wavelet features to the dataframe in-place style (returns copy).

        Expects columns: Close, Returns (at minimum).
        """
        if not HAS_PYWT:
            return self._fallback_features(df)

        result = df.copy()

        close = df['Close'].values.astype(np.float64)
        returns = df['Returns'].fillna(0).values.astype(np.float64)

        result = self._add_wavelet_coefficients(result, close, 'Close')
        result = self._add_wavelet_coefficients(result, returns, 'Returns')
        result = self._add_energy_ratios(result, close, 'Close')
        result = self._add_cross_scale_features(result, close, 'Close')
        result = self._add_wavelet_volatility(result, returns)

        return result

    def _add_wavelet_coefficients(self, df, signal, prefix):
        """Rolling wavelet decomposition → last coefficient per level."""
        window = 2 ** (self.levels + 1)
        n = len(signal)
        approx = np.full(n, np.nan)
        details = [np.full(n, np.nan) for _ in range(self.levels)]

        for i in range(window, n):
            segment = signal[i - window:i]
            try:
                coeffs = pywt.wavedec(segment, self.wavelet, level=self.levels)
            except Exception:
                continue
            approx[i] = coeffs[0][-1]
            for lvl in range(self.levels):
                details[lvl][i] = coeffs[lvl + 1][-1]

        df[f'ATHENA_Wavelet_{prefix}_Trend'] = approx
        for lvl in range(self.levels):
            df[f'ATHENA_Wavelet_{prefix}_D{lvl + 1}'] = details[lvl]

        return df

    def _add_energy_ratios(self, df, signal, prefix):
        """Fraction of total energy at each decomposition scale."""
        window = 2 ** (self.levels + 1)
        n = len(signal)
        energies = [np.full(n, np.nan) for _ in range(self.levels + 1)]

        for i in range(window, n):
            segment = signal[i - window:i]
            try:
                coeffs = pywt.wavedec(segment, self.wavelet, level=self.levels)
            except Exception:
                continue
            total_energy = sum(np.sum(c ** 2) for c in coeffs) + 1e-12
            for lvl, c in enumerate(coeffs):
                energies[lvl][i] = np.sum(c ** 2) / total_energy

        df[f'ATHENA_Wavelet_{prefix}_Energy_Approx'] = energies[0]
        for lvl in range(self.levels):
            df[f'ATHENA_Wavelet_{prefix}_Energy_D{lvl + 1}'] = energies[lvl + 1]

        return df

    def _add_cross_scale_features(self, df, signal, prefix):
        """Correlation between adjacent wavelet scales over a rolling window."""
        window = 2 ** (self.levels + 1)
        corr_window = 20
        n = len(signal)

        detail_series = [np.full(n, 0.0) for _ in range(self.levels)]

        for i in range(window, n):
            segment = signal[i - window:i]
            try:
                coeffs = pywt.wavedec(segment, self.wavelet, level=self.levels)
            except Exception:
                continue
            for lvl in range(self.levels):
                detail_series[lvl][i] = coeffs[lvl + 1][-1]

        for lvl in range(self.levels - 1):
            s1 = pd.Series(detail_series[lvl])
            s2 = pd.Series(detail_series[lvl + 1])
            df[f'ATHENA_Wavelet_{prefix}_CrossCorr_D{lvl + 1}_D{lvl + 2}'] = (
                s1.rolling(corr_window).corr(s2)
            )

        return df

    def _add_wavelet_volatility(self, df, returns):
        """Volatility estimated from detail coefficients of returns."""
        window = 2 ** (self.levels + 1)
        n = len(returns)
        wavelet_vol = np.full(n, np.nan)

        for i in range(window, n):
            segment = returns[i - window:i]
            try:
                coeffs = pywt.wavedec(segment, self.wavelet, level=self.levels)
            except Exception:
                continue
            detail_energy = sum(np.sum(c ** 2) for c in coeffs[1:])
            wavelet_vol[i] = np.sqrt(detail_energy / window)

        df['ATHENA_Wavelet_Volatility'] = wavelet_vol
        return df

    def _fallback_features(self, df):
        """When PyWavelets is not installed, use simple rolling proxies."""
        result = df.copy()
        close = df['Close']
        returns = df['Returns'].fillna(0)

        for w in [4, 8, 16]:
            result[f'ATHENA_Wavelet_Close_Scale{w}'] = (
                close.rolling(w).mean() - close.rolling(w * 2).mean()
            )
            result[f'ATHENA_Wavelet_Returns_Scale{w}'] = (
                returns.rolling(w).std()
            )

        result['ATHENA_Wavelet_Volatility'] = returns.rolling(16).std()
        return result
