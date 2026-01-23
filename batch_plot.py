#!/usr/bin/env python3
"""
SM_2D Batch Result Plotter
배치 시뮬레이션 결과를 시각화합니다.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# 한글 폰트 설정 (선택사항)
try:
    rcParams['font.family'] = 'DejaVu Sans'
except:
    pass


class BatchPlotter:
    """배치 결과 시각화"""

    def __init__(self, result_dirs: Optional[List[Path]] = None):
        """
        Args:
            result_dirs: 결과 디렉토리 리스트 (None이면 자동 탐색)
        """
        if result_dirs is None:
            result_dirs = self._find_result_dirs()

        self.result_dirs = result_dirs
        self.results = self._load_results()

    def _find_result_dirs(self, base_path: str = "results/batch") -> List[Path]:
        """결과 디렉토리 자동 탐색"""
        base = Path(base_path)
        if not base.exists():
            return []

        return [d for d in base.iterdir() if d.is_dir() and d.name.startswith("batch_")]

    def _load_results(self) -> List[Dict]:
        """모든 결과 로드"""
        results = []

        for result_dir in self.result_dirs:
            pdd_file = result_dir / "pdd.txt"

            if pdd_file.exists():
                try:
                    # PDD 파일 로드
                    data = np.loadtxt(pdd_file)

                    pdd_result = {
                        'dir': result_dir,
                        'name': result_dir.name,
                        'z_mm': data[:, 0],
                        'dose_Gy': data[:, 1],
                        'dose_norm': data[:, 2] if data.shape[1] > 2 else None,
                        'bragg_peak_z': None,
                        'max_dose': None
                    }

                    # Bragg peak 위치 찾기
                    max_idx = np.argmax(pdd_result['dose_Gy'])
                    pdd_result['bragg_peak_z'] = pdd_result['z_mm'][max_idx]
                    pdd_result['max_dose'] = pdd_result['dose_Gy'][max_idx]

                    results.append(pdd_result)
                except Exception as e:
                    print(f"[WARNING] {pdd_file} 로드 실패: {e}")

        return results

    def _parse_params(self, name: str) -> Dict[str, float]:
        """이름에서 파라미터 파싱 (예: batch_MeV150_x0-10 -> {MeV: 150, x0: -10})"""
        import re
        params = {}
        parts = name.replace('batch_', '').split('_')

        for part in parts:
            # 정규식으로 파싱: 키(알파벳) + 값(숫자, 음수 포함)
            match = re.match(r'([a-zA-Z]+)(-?\d+\.?\d*)', part)
            if match:
                key = match.group(1)
                value = float(match.group(2))
                params[key] = value

        return params

    def plot_pdd_comparison(self, figsize=(12, 6)) -> None:
        """PDD 곡선 비교 플롯"""
        if not self.results:
            print("PDD 결과가 없습니다.")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 원본 dose
        for result in self.results:
            params = self._parse_params(result['name'])
            label = result['name'].replace('batch_', '')
            axes[0].plot(result['z_mm'], result['dose_Gy'], label=label, alpha=0.7)

        axes[0].set_xlabel('Depth (mm)')
        axes[0].set_ylabel('Dose (Gy)')
        axes[0].set_title('PDD Comparison')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)

        # 정규화된 dose (있는 경우)
        has_norm = any(r['dose_norm'] is not None for r in self.results)
        if has_norm:
            for result in self.results:
                if result['dose_norm'] is not None:
                    params = self._parse_params(result['name'])
                    label = result['name'].replace('batch_', '')
                    axes[1].plot(result['z_mm'], result['dose_norm'],
                                label=label, alpha=0.7)

            axes[1].set_xlabel('Depth (mm)')
            axes[1].set_ylabel('Normalized Dose')
            axes[1].set_title('Normalized PDD Comparison')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=8)
        else:
            # 정규화 없으면 Bragg peak 위치 비교
            energies = []
            bragg_z = []
            for result in sorted(self.results, key=lambda r: r['name']):
                params = self._parse_params(result['name'])
                if 'MeV' in params:
                    energies.append(params['MeV'])
                    bragg_z.append(result['bragg_peak_z'])

            if energies:
                axes[1].plot(energies, bragg_z, 'o-')
                axes[1].set_xlabel('Energy (MeV)')
                axes[1].set_ylabel('Bragg Peak Depth (mm)')
                axes[1].set_title('Bragg Peak Depth vs Energy')
                axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_2d_comparison(self, figsize=(15, 5)) -> None:
        """2D dose map 비교 (선택된 결과만)"""
        if not self.results:
            print("2D dose 결과가 없습니다.")
            return

        # 최대 6개까지 표시
        n_plot = min(len(self.results), 6)
        fig, axes = plt.subplots(1, n_plot, figsize=figsize)

        if n_plot == 1:
            axes = [axes]

        for i in range(n_plot):
            result = self.results[i]
            dose_file = result['dir'] / "dose_2d.txt"

            if dose_file.exists():
                try:
                    data = np.loadtxt(dose_file)
                    x = data[:, 0]
                    z = data[:, 1]
                    dose = data[:, 2]

                    # 2D 그리드로 변환
                    x_unique = np.unique(x)
                    z_unique = np.unique(z)
                    dose_grid = dose.reshape(len(z_unique), len(x_unique))

                    # X-Z 평면이므로 transpose
                    X, Z = np.meshgrid(x_unique, z_unique)

                    im = axes[i].pcolormesh(X, Z, dose_grid, shading='auto',
                                           cmap='hot')
                    axes[i].set_xlabel('X (mm)')
                    axes[i].set_ylabel('Z (mm)')
                    axes[i].set_title(result['name'].replace('batch_', ''), fontsize=10)
                    axes[i].set_aspect('equal')
                    plt.colorbar(im, ax=axes[i], label='Dose (Gy)')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Load failed\n{e}',
                               ha='center', va='center')
                    axes[i].set_title(result['name'])

        plt.tight_layout()
        return fig

    def plot_bragg_peak_summary(self, figsize=(10, 6)) -> None:
        """Bragg peak 요약 플롯"""
        if not self.results:
            print("결과가 없습니다.")
            return

        # 파라미터별로 그룹화
        param_groups = {}
        for result in self.results:
            params = self._parse_params(result['name'])

            # 주요 파라미터 (MeV 또는 x) 기준 그룹화
            key = params.get('MeV', params.get('x', 0))
            param_groups[key] = result

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 에너지별 Bragg peak 깊이
        energies = []
        bragg_depths = []
        for result in sorted(self.results, key=lambda r: r['name']):
            params = self._parse_params(result['name'])
            if 'MeV' in params:
                energies.append(params['MeV'])
                bragg_depths.append(result['bragg_peak_z'])

        if energies:
            axes[0].plot(energies, bragg_depths, 'o-', linewidth=2, markersize=8)
            axes[0].set_xlabel('Energy (MeV)', fontsize=12)
            axes[0].set_ylabel('Bragg Peak Depth (mm)', fontsize=12)
            axes[0].set_title('Bragg Peak Depth vs Energy', fontsize=14)
            axes[0].grid(True, alpha=0.3)

        # 최대 dose 비교
        max_doses = [r['max_dose'] for r in self.results]
        names = [r['name'].replace('batch_', '') for r in self.results]

        axes[1].bar(range(len(names)), max_doses)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('Max Dose (Gy)', fontsize=12)
        axes[1].set_title('Maximum Dose Comparison', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_all(self, output_path: str = "results/batch_summary.png") -> None:
        """모든 플롯 생성 및 저장"""
        fig1 = self.plot_pdd_comparison()
        if fig1:
            fig1.savefig(output_path.replace('.png', '_pdd.png'), dpi=150, bbox_inches='tight')
            plt.close(fig1)
            print(f"PDD 플롯 저장: {output_path.replace('.png', '_pdd.png')}")

        fig2 = self.plot_2d_comparison()
        if fig2:
            fig2.savefig(output_path.replace('.png', '_2d.png'), dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"2D 플롯 저장: {output_path.replace('.png', '_2d.png')}")

        fig3 = self.plot_bragg_peak_summary()
        if fig3:
            fig3.savefig(output_path.replace('.png', '_bragg.png'), dpi=150, bbox_inches='tight')
            plt.close(fig3)
            print(f"Bragg peak 요약 저장: {output_path.replace('.png', '_bragg.png')}")

        # 통합 플롯
        self._plot_combined(output_path)

    def _plot_combined(self, output_path: str) -> None:
        """통합 요약 플롯"""
        if not self.results:
            return

        fig = plt.figure(figsize=(16, 10))

        # PDD 비교
        ax1 = fig.add_subplot(2, 2, 1)
        for result in self.results:
            label = result['name'].replace('batch_', '')
            ax1.plot(result['z_mm'], result['dose_Gy'], label=label, alpha=0.7)
        ax1.set_xlabel('Depth (mm)')
        ax1.set_ylabel('Dose (Gy)')
        ax1.set_title('PDD Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        # Bragg peak depth
        ax2 = fig.add_subplot(2, 2, 2)
        energies = []
        bragg_z = []
        for result in sorted(self.results, key=lambda r: r['name']):
            params = self._parse_params(result['name'])
            if 'MeV' in params:
                energies.append(params['MeV'])
                bragg_z.append(result['bragg_peak_z'])
        if energies:
            ax2.plot(energies, bragg_z, 'o-', linewidth=2, markersize=8)
            ax2.set_xlabel('Energy (MeV)', fontsize=12)
            ax2.set_ylabel('Bragg Peak Depth (mm)', fontsize=12)
            ax2.set_title('Bragg Peak Depth vs Energy', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # Max dose comparison
        ax3 = fig.add_subplot(2, 2, 3)
        max_doses = [r['max_dose'] for r in self.results]
        names = [r['name'].replace('batch_', '') for r in self.results]
        ax3.bar(range(len(names)), max_doses, color='steelblue')
        ax3.set_xticks(range(len(names)))
        ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Max Dose (Gy)', fontsize=12)
        ax3.set_title('Maximum Dose by Configuration', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Statistics
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        stats_text = "=== Batch Simulation Summary ===\n\n"
        stats_text += f"Total simulations: {len(self.results)}\n\n"

        if energies:
            stats_text += "Energy range: {:.1f} - {:.1f} MeV\n".format(min(energies), max(energies))
            stats_text += "Bragg peak range: {:.1f} - {:.1f} mm\n\n".format(min(bragg_z), max(bragg_z))

        stats_text += "Results:\n"
        for result in self.results:
            params = self._parse_params(result['name'])
            stats_text += f"  {result['name']}: Bragg peak @ {result['bragg_peak_z']:.1f} mm\n"

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.suptitle('SM_2D Batch Simulation Results', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"통합 요약 플롯 저장: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SM_2D Batch Result Plotter")
    parser.add_argument(
        '-d', '--dirs',
        nargs='+',
        help='결과 디렉토리 (지정하지 않으면 자동 탐색)'
    )
    parser.add_argument(
        '-o', '--output',
        default='results/batch_summary.png',
        help='출력 파일 경로'
    )
    parser.add_argument(
        '--pdd-only',
        action='store_true',
        help='PDD 플롯만 생성'
    )
    parser.add_argument(
        '--2d-only',
        action='store_true',
        help='2D 플롯만 생성'
    )

    args = parser.parse_args()

    result_dirs = [Path(d) for d in args.dirs] if args.dirs else None
    plotter = BatchPlotter(result_dirs)

    if args.pdd_only:
        fig = plotter.plot_pdd_comparison()
        if fig:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"PDD 플롯 저장: {args.output}")
    elif args._2d_only:
        fig = plotter.plot_2d_comparison()
        if fig:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"2D 플롯 저장: {args.output}")
    else:
        plotter.plot_all(args.output)


if __name__ == "__main__":
    main()
