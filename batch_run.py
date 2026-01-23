#!/usr/bin/env python3
"""
SM_2D Batch Simulation Runner
sim.ini 템플릿을 기반으로 파라미터 스윕 시뮬레이션을 실행합니다.
"""

import os
import sys
import copy
import shutil
import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Any
import yaml


def _run_wrapper(args):
    """multiprocessing용 래퍼 함수"""
    runner, run = args
    return runner.run_single(run)


class ConfigManager:
    """INI 설정 파일 관리"""

    def __init__(self, template_path: str):
        self.template_path = Path(template_path)
        self.config = self._parse_ini()

    def _parse_ini(self) -> Dict[str, Dict[str, str]]:
        """INI 파일 파싱"""
        config = {}
        current_section = None

        with open(self.template_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # 빈 줄 또는 주석 무시
                if not line or line.startswith('#'):
                    continue

                # 섹션 파싱
                if line.startswith('[') and line.endswith(']'):
                    current_section = line
                    config[current_section] = {}
                # 키-값 파싱
                elif '=' in line and current_section:
                    key, value = line.split('=', 1)
                    config[current_section][key.strip()] = value.strip()

        return config

    def modify(self, section: str, key: str, value: Any) -> None:
        """설정 값 수정"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = str(value)

    def save(self, output_path: str) -> None:
        """설정을 INI 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for section, params in self.config.items():
                f.write(f"{section}\n")
                for key, value in params.items():
                    f.write(f"{key} = {value}\n")
                f.write("\n")


class ParameterSweep:
    """파라미터 스윕 조합 생성"""

    @staticmethod
    def generate_combinations(sweep_config: Dict) -> List[Dict]:
        """
        스윕 설정에서 모든 파라미터 조합 생성 (Cartesian product)

        Args:
            sweep_config: {
                'energy': {
                    'section': '[energy]',
                    'mean_MeV': [150, 160, ...]
                },
                'position': {
                    'section': '[spatial]',
                    'x0_mm': [-10, 0, 10]
                }
            }

        Returns:
            파라미터 조합 리스트, 각 조합은 여러 섹션의 수정사항 포함
        """
        # 각 그룹별 가능한 값의 리스트
        group_values = []

        for group_name, group_config in sweep_config.items():
            section = group_config.get('section', '')
            params = {k: v for k, v in group_config.items() if k != 'section'}

            if not params:
                continue

            # 그룹 내 파라미터 조합 생성
            keys = list(params.keys())
            values = [params[k] if isinstance(params[k], list) else [params[k]]
                      for k in keys]

            group_combos = []
            for combo in product(*values):
                group_combos.append({
                    'section': section,
                    'params': dict(zip(keys, combo))
                })

            group_values.append(group_combos)

        # 모든 그룹 간 Cartesian product
        combinations = []
        if group_values:
            for multi_combo in product(*group_values):
                # 실행 이름 생성
                name_parts = []
                for mod in multi_combo:
                    for key, value in mod['params'].items():
                        param_name = key.replace('_mm', '').replace('_MeV', '')
                        name_parts.append(f"{param_name}{value}")

                combo_dict = {
                    'name': '_'.join(name_parts),
                    'modifications': list(multi_combo)
                }
                combinations.append(combo_dict)

        return combinations


class BatchRunner:
    """배치 시뮬레이션 실행기"""

    def __init__(self, config_file: str = "batch_config.yaml"):
        self.config_file = Path(config_file)
        self.batch_config = self._load_batch_config()
        self.template_config = ConfigManager(self.batch_config['template'])

        self.runs = []
        self.completed = []
        self.failed = []

    def _load_batch_config(self) -> Dict:
        """배치 설정 로드"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def prepare_runs(self) -> List[Dict]:
        """실행할 시뮬레이션 준비"""
        sweep_config = self.batch_config.get('sweep', {})
        output_base = self.batch_config.get('output_base', 'results/batch')

        combinations = ParameterSweep.generate_combinations(sweep_config)

        runs = []
        for idx, combo in enumerate(combinations):
            # 실행 이름 (이미 generate_combinations에서 생성됨)
            run_name = f"batch_{combo['name']}"
            run_dir = Path(output_base) / run_name

            # 수정된 설정 생성
            config_copy = copy.deepcopy(self.template_config)

            for mod in combo['modifications']:
                section = mod['section']
                for key, value in mod['params'].items():
                    config_copy.modify(section, key, value)

            # output_dir 수정 - 절대 경로로 설정하여 중복 방지
            config_copy.modify('[output]', 'output_dir', str(run_dir.resolve()))

            runs.append({
                'index': idx,
                'name': run_name,
                'dir': run_dir,
                'config': config_copy,
                'params': combo
            })

        self.runs = runs
        return runs

    def run_single(self, run: Dict, executable: str = "./run_simulation") -> bool:
        """단일 시뮬레이션 실행"""
        run_dir = run['dir']
        config = run['config']

        try:
            # 출력 디렉토리 생성
            run_dir.mkdir(parents=True, exist_ok=True)

            # 임시 config 파일 저장
            config_path = run_dir / "sim_run.ini"
            config.save(str(config_path))

            # 시뮬레이션 실행
            result = subprocess.run(
                [executable, str(config_path)],
                capture_output=True,
                text=True,
                timeout=self.batch_config.get('execution', {}).get('timeout', 300)
            )

            if result.returncode == 0:
                print(f"[OK] {run['name']}")
                return True
            else:
                print(f"[FAIL] {run['name']}: {result.stderr[:100]}")
                return False

        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {run['name']}")
            return False
        except Exception as e:
            print(f"[ERROR] {run['name']}: {e}")
            return False

    def run_all(self, parallel: int = None) -> None:
        """모든 시뮬레이션 실행"""
        if not self.runs:
            self.prepare_runs()

        if parallel is None:
            parallel = self.batch_config.get('execution', {}).get('parallel', 1)

        # parallel: 0이면 사용 가능한 모든 코어 사용
        if parallel == 0:
            parallel = mp.cpu_count()

        n_runs = len(self.runs)

        # GPU 시뮬레이션에서 병렬 실행은 권장하지 않음
        if parallel > 1:
            print(f"\n[WARNING] GPU 시뮬레이션에서 병렬 실행은 CUDA 메모리 문제를 일으킬 수 있습니다.")
            print(f"[WARNING] 병렬 프로세스가 각각 GPU를 사용하려고 하므로 메모리 부족이 발생할 수 있습니다.")
            response = input(f"\n총 {n_runs}개 시뮬레이션을 병렬 {parallel}프로세스로 실행하시겠습니까? (y/n): ")
            if response.lower() != 'y':
                print("순차 실행으로 변경합니다.")
                parallel = 1

        print(f"\n총 {n_runs}개 시뮬레이션 실행 (병렬: {parallel})\n")

        if parallel > 1:
            # 병렬 실행 - GPU와 호환되도록 spawn 방식 사용
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=parallel) as pool:
                # run_single은 인스턴스 메서드이므로 래퍼 함수 필요
                results = pool.map(_run_wrapper, [(self, run) for run in self.runs])

            for run, success in zip(self.runs, results):
                if success:
                    self.completed.append(run)
                else:
                    self.failed.append(run)
        else:
            # 순차 실행 (GPU 권장)
            for run in self.runs:
                success = self.run_single(run)
                if success:
                    self.completed.append(run)
                else:
                    self.failed.append(run)

        # 결과 요약
        print(f"\n완료: {len(self.completed)}/{n_runs}")
        if self.failed:
            print(f"실패: {len(self.failed)}")
            for run in self.failed:
                print(f"  - {run['name']}")

    def get_completed_dirs(self) -> List[Path]:
        """완료된 실행의 결과 디렉토리 반환"""
        return [run['dir'] for run in self.completed]


def main():
    parser = argparse.ArgumentParser(description="SM_2D Batch Simulation Runner")
    parser.add_argument(
        '-c', '--config',
        default='batch_config.yaml',
        help='배치 설정 파일 (기본값: batch_config.yaml)'
    )
    parser.add_argument(
        '-p', '--parallel',
        type=int,
        help='병렬 프로세스 수 (설정 파일 무시)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실행할 조합만 보여주고 실제로는 실행하지 않음'
    )
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='결과 플로팅만 수행'
    )

    args = parser.parse_args()

    runner = BatchRunner(args.config)

    if args.plot_only:
        # 플로팅만 실행
        from batch_plot import BatchPlotter
        plotter = BatchPlotter()
        plotter.plot_all()
        return

    # 실행 준비
    runs = runner.prepare_runs()

    if args.dry_run:
        print("실행될 시뮬레이션:")
        for run in runs:
            print(f"  {run['name']}")
            for mod in run['params']['modifications']:
                print(f"    {mod['params']}")
        return

    # 실행
    runner.run_all(parallel=args.parallel)

    # 자동 플로팅
    if runner.batch_config.get('plot', {}).get('auto_plot', False):
        if runner.completed:
            print("\n결과 시각화 중...")
            from batch_plot import BatchPlotter
            plotter = BatchPlotter(runner.get_completed_dirs())
            plotter.plot_all()


if __name__ == "__main__":
    main()
