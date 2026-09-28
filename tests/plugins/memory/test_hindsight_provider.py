"""Hindsight provider tests — retired upstream.

The bundled hindsight memory provider moved to the plugin catalog upstream, so
its test suite was removed with it (upstream 4cbf862abe + 090b06c345, whose
generic cases were retargeted into the shared memory-provider tests).  The path
is kept as an empty module because the carried-port harness byte-compiles every
path that was reported as a merge conflict.
"""
