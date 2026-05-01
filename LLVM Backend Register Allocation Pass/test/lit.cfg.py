import os

import lit.formats

config.name = "custom-ra"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = getattr(config, "test_exec_root", config.test_source_root)

tools_dir = getattr(config, "llvm_tools_dir", "")


def tool(name):
    return os.path.join(tools_dir, name) if tools_dir else name


config.substitutions.append(("%llc", tool("llc")))
config.substitutions.append(("%FileCheck", tool("FileCheck")))
config.substitutions.append(("%custom_ra", getattr(config, "custom_ra_plugin", "")))
