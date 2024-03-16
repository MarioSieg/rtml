import os
import platform
import shutil
from pathlib import Path
from typing import Any

import setuptools
from setuptools.command import build_ext

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

# hardcoded SABI-related options. Requires that each Python interpreter
# (hermetic or not) participating is of the same major-minor version.
version_tuple = tuple(int(i) for i in platform.python_version_tuple())
py_limited_api = version_tuple >= (3, 12)
options = {"bdist_wheel": {"py_limited_api": "cp312"}} if py_limited_api else {}


class BazelExtension(setuptools.Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, name: str, bazel_target: str, **kwargs: Any):
        super().__init__(name=name, sources=[], **kwargs)

        self.bazel_target = bazel_target
        stripped_target = bazel_target.split("//")[-1]
        self.relpath, self.target_name = stripped_target.split(":")


class BuildBazelExtension(build_ext.build_ext):
    """A command that runs Bazel to build a C/C++ extension."""

    def run(self):
        for ext in self.extensions:
            self.bazel_build(ext)
        super().run()
        # explicitly call `bazel shutdown` for graceful exit
        self.spawn(["bazel", "shutdown"])

    def copy_extensions_to_source(self):
        """
        Copy generated extensions into the source tree.
        This is done in the ``bazel_build`` method, so it's not necessary to
        do again in the `build_ext` base class.
        """
        pass

    def bazel_build(self, ext: BazelExtension) -> None:
        """Runs the bazel build to create the package."""
        temp_path = Path(self.build_temp)
        # omit the patch version to avoid build errors if the toolchain is not
        # yet registered in the current @rules_python version.
        # patch version differences should be fine.
        python_version = ".".join(platform.python_version_tuple()[:2])

        bazel_argv = [
            "bazel",
            "build",
            ext.bazel_target,
            f"--symlink_prefix={temp_path / 'bazel-'}",
            f"--compilation_mode={'dbg' if self.debug else 'opt'}",
            # C++17 is required by nanobind
            f"--cxxopt={'/std:c++17' if IS_WINDOWS else '-std=c++17'}",
            f"--@rules_python//python/config_settings:python_version={python_version}",
        ]

        if ext.py_limited_api:
            bazel_argv += ["--@nanobind_bazel//:py-limited-api=cp312"]

        if IS_WINDOWS:
            # Link with python*.lib.
            for library_dir in self.library_dirs:
                bazel_argv.append("--linkopt=/LIBPATH:" + library_dir)
        elif IS_MAC:
            if platform.machine() == "x86_64":
                # C++17 needs macOS 10.14 at minimum
                bazel_argv.append("--macos_minimum_os=10.14")

                # cross-compilation for Mac ARM64 on GitHub Mac x86 runners.
                # ARCHFLAGS is set by cibuildwheel before macOS wheel builds.
                archflags = os.getenv("ARCHFLAGS", "")
                if "arm64" in archflags:
                    bazel_argv.append("--cpu=darwin_arm64")
                    bazel_argv.append("--macos_cpus=arm64")

            elif platform.machine() == "arm64":
                bazel_argv.append("--macos_minimum_os=11.0")

        self.spawn(bazel_argv)

        if IS_WINDOWS:
            suffix = ".pyd"
        else:
            suffix = ".abi3.so" if ext.py_limited_api else ".so"

        ext_name = ext.target_name + suffix
        ext_bazel_bin_path = temp_path / "bazel-bin" / ext.relpath / ext_name
        ext_dest_path = Path(self.get_ext_fullpath(ext.name)).with_name(
            ext_name
        )
        shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


setuptools.setup(
    cmdclass=dict(build_ext=BuildBazelExtension),
    ext_modules=[
        BazelExtension(
            name="google_benchmark._benchmark",
            bazel_target="//bindings/python/google_benchmark:_benchmark",
            py_limited_api=py_limited_api,
        )
    ],
    options=options,
)