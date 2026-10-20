#!/usr/bin/env python3
"""Audit an executable-only MTP install without loading the model or GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def audit(prefix, cap):
    prefix = prefix.resolve(strict=True)
    data = prefix / 'share/Qwen3xOrinMTP'
    manifest = json.loads((data / f'cap{cap}/manifest.json').read_text())
    profile = f'q3x.sm87.service.mtp-gdn-owner-input-cap{cap}.v1'
    require(manifest['schema'] == 'q3x-mtp-executable-package-v1', 'manifest schema')
    require(manifest['draft_cap'] == cap and manifest['profile'] == profile, 'compiled profile')
    require(manifest['model_revision'] == '0893e1606ff3d5f97a441f405d5fc541a6bdf404',
            'model revision')
    require(manifest['BUILD_TESTING'] is False and manifest['exports_cpp_abi'] is False,
            'package must exclude test seams and C++ ABI')
    require(re.fullmatch(r'[0-9a-f]{40}', manifest['source_commit']), 'source identity')
    require(type(manifest['source_dirty']) is bool, 'source cleanliness identity')
    relative = Path(manifest['executable'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'executable path')
    binary = (prefix / relative).resolve(strict=True)
    require(binary.is_relative_to(prefix), 'executable escapes install')
    require(binary.name == f'qwen3x-mtp-server-cap{cap}', 'fixed-cap executable name')
    require(hashlib.sha256(binary.read_bytes()).hexdigest() == manifest['binary_sha256'],
            'installed executable hash mismatch')
    for path in prefix.rglob('*'):
        require(not (path.suffix in ('.h', '.hpp', '.a', '.so') or '.so.' in path.name
                     or path.name.startswith('Qwen3xOrinConfig')
                     or path.name == 'qwen3x-eval-server'), 'foreign ABI/service artifact: ' + str(path))
    help_text = subprocess.check_output([str(binary), '--help'], text=True)
    for value in (profile, f'MTP draft cap: {cap} (compiled)',
                  'independent service package; qualification is artifact-specific'):
        require(value in help_text, 'help identity: ' + value)
    for value in ('--candidate-profile', '--production-profile', '--development-route',
                  '--prefill-projection-tactic', '--projection-backend', 'Q3X_MTP_DRAFT_LENGTH'):
        require(value not in help_text, 'exposed route selector: ' + value)
    dependencies = subprocess.check_output(['readelf', '-d', str(binary)], text=True)
    symbols = subprocess.check_output(['nm', '-C', str(binary)], text=True)
    require('cublas' not in dependencies.lower(), 'forbidden cuBLAS dependency')
    # Native CUDA stubs encode historical source basenames containing "cublas";
    # those names are not vendor API symbols. Check actual symbol entry names,
    # and reject cuBLASLt contexts/wrappers anywhere in the demangled name.
    require(not re.search(r'^\s*(?:[0-9a-f]+\s+)?[a-z?]\s+cublas\w*(?:@|\s|\(|$)',
                          symbols, re.I | re.M), 'forbidden cuBLAS API symbol')
    for value in ('cublasLt', 'EngineAccess', 'set_gdn_owner_constructor_failure_for_test'):
        require(value.lower() not in symbols.lower(), 'forbidden symbol: ' + value)
    for name in ('LICENSE', 'NOTICE', 'LICENSE.cutlass.txt', 'LICENSE.fla.txt'):
        require((data / 'licenses' / name).is_file(), 'missing license: ' + name)
    require(any((data / 'licenses/flashinfer').rglob('LICENSE*')), 'missing FlashInfer licenses')
    require((data / 'systemd/qwen3x-mtp@.service').is_file(), 'missing supervisor template')
    return dict(passed=True, cap=cap, profile=profile, binary_sha256=manifest['binary_sha256'],
                source_commit=manifest['source_commit'],
                source_dirty=manifest['source_dirty'],
                scope='Installed host identity/ABI/dependency audit; no numerical or performance claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--cap', type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prefix, args.cap), indent=2))


if __name__ == '__main__':
    main()
