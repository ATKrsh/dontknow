import os
import re
import sys
import shutil
import subprocess

def get_next_version_name(output_dir: str, prefix: str = "dontknow_v") -> str:
    """Find existing versioned build folders/executables and determine next version name (e.g., dontknow_v3.exe)."""
    os.makedirs(output_dir, exist_ok=True)
    existing_versions = []
    pattern = re.compile(rf"^{prefix}(\d+)(\.exe)?$", re.IGNORECASE)

    for fname in os.listdir(output_dir):
        match = pattern.match(fname)
        if match:
            existing_versions.append(int(match.group(1)))

    next_ver = max(existing_versions) + 1 if existing_versions else 1
    return f"{prefix}{next_ver}.exe"

def build():
    print("==================================================")
    print("   Building Standalone Executable for 'dontknow'")
    print("==================================================")

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(workspace_dir, "dist")
    build_dir = os.path.join(workspace_dir, "build")

    target_exe_name = get_next_version_name(dist_dir, prefix="dontknow_v")
    print(f"[BUILD] Target Executable Version: {target_exe_name}")

    # PyInstaller arguments
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", os.path.splitext(target_exe_name)[0],
        "--collect-all", "sumy",
        "--collect-all", "PIL",
        os.path.join(workspace_dir, "app.py")
    ]

    print(f"Executing build command: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=workspace_dir)

    if res.returncode == 0:
        dist_exe_path = os.path.join(dist_dir, os.path.splitext(target_exe_name)[0], f"{os.path.splitext(target_exe_name)[0]}.exe")
        print("--------------------------------------------------")
        print(f"[SUCCESS] Build completed!")
        print(f"Executable path: {dist_exe_path}")
        print("--------------------------------------------------")
    else:
        print("[ERROR] Build failed with exit code:", res.returncode)
        sys.exit(res.returncode)

if __name__ == "__main__":
    build()
