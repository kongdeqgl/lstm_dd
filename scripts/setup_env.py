"""用 Python 3.12 创建项目独立环境；离线注册 src，不修改系统环境。"""

import json
import subprocess
import sys
import venv
from pathlib import Path


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("请使用 Python 3.12 运行本脚本，例如 py -3.12 scripts/setup_env.py")
    root = Path(__file__).resolve().parents[1]
    environment = root / ".venv"
    executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not environment.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    elif not (environment / "pyvenv.cfg").is_file() or not executable.is_file():
        raise SystemExit(".venv 已存在但不是完整虚拟环境，请保留并检查它后重试")
    probe = subprocess.run(
        [str(executable), "-c", (
            "import json,site,sys; "
            "print(json.dumps({'version': list(sys.version_info[:2]), "
            "'site': site.getsitepackages()[-1]}))"
        )],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(probe.stdout)
    if info["version"] != [3, 12]:
        raise SystemExit("现有 .venv 不是 Python 3.12，未修改该环境")
    packages = Path(info["site"]).resolve()
    if not packages.is_relative_to(environment.resolve()) or not packages.is_dir():
        raise SystemExit("虚拟环境 site-packages 路径异常，未写入任何注册文件")
    # 开发模式：只把当前项目 src 注册到当前项目环境，无需下载构建依赖。
    # 相对虚拟环境定位项目：整个文件夹移动后仍可找到 src。
    registration = (
        "import sys, pathlib; "
        "sys.path.insert(0, str(pathlib.Path(sys.prefix).parent / 'src'))\n"
    )
    (packages / "sinr_predictor_dev.pth").write_text(registration, encoding="ascii")
    print("Python 3.12 项目环境已就绪：")
    print(executable)
    print("请在 VS Code 中选择该解释器，再运行示例或测试。")


if __name__ == "__main__":
    main()
