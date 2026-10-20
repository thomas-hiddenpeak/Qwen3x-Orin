"""Host negative controls for independently installed MTP artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mtp_install', ROOT / 'tools/evaluation/audit_mtp_install.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InstallTest(unittest.TestCase):
    def setUp(self):
        work = ROOT / '.q3x-work/host-tests'
        work.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=work)
        self.addCleanup(self.tmp.cleanup)
        self.prefix = Path(self.tmp.name)
        self.data = self.prefix / 'share/Qwen3xOrinMTP'
        self.binary = self.prefix / 'bin/qwen3x-mtp-server-cap2'
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b'host-fixture-not-an-executable')
        self.manifest = self.data / 'cap2/manifest.json'
        self.manifest.parent.mkdir(parents=True)
        self.value = dict(schema='q3x-mtp-executable-package-v1', draft_cap=2,
            profile='q3x.sm87.service.mtp-gdn-owner-input-cap2.v1', BUILD_TESTING=False,
            exports_cpp_abi=False, source_commit='a'*40, source_dirty=False,
            executable='bin/qwen3x-mtp-server-cap2',
            binary_sha256=hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            model_revision='0893e1606ff3d5f97a441f405d5fc541a6bdf404')
        self.write_manifest()
        for name in ('licenses/LICENSE', 'licenses/NOTICE', 'licenses/LICENSE.cutlass.txt',
                     'licenses/LICENSE.fla.txt', 'licenses/flashinfer/LICENSE', 'systemd/qwen3x-mtp@.service'):
            p = self.data / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('host fixture')
        self.help = ('q3x.sm87.service.mtp-gdn-owner-input-cap2.v1\nMTP draft cap: 2 (compiled)\n'
                     'independent service package; qualification is artifact-specific')
        self.dependencies = 'NEEDED libcudart.so.13'
        self.symbols = 'ordinary_service'

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.value))

    def run_audit(self):
        def command(args, **kwargs):
            return self.dependencies if args[0] == 'readelf' else self.symbols if args[0] == 'nm' else self.help
        with patch.object(module.subprocess, 'check_output', side_effect=command):
            return module.audit(self.prefix, 2)

    def test_valid(self):
        self.assertTrue(self.run_audit()['passed'])

    def test_changed_binary(self):
        self.binary.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'): self.run_audit()

    def test_wrong_cap(self):
        self.value['draft_cap'] = 3
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, 'compiled profile'): self.run_audit()

    def test_cpp_abi_export(self):
        (self.prefix / 'leaked.h').write_text('header')
        with self.assertRaisesRegex(RuntimeError, 'foreign ABI'): self.run_audit()

    def test_test_seam(self):
        self.symbols = 'q3x::runtime::EngineAccess::runner()'
        with self.assertRaisesRegex(RuntimeError, 'forbidden symbol'): self.run_audit()

    def test_cublas_dependency(self):
        self.dependencies += ' NEEDED libcublasLt.so.13'
        with self.assertRaisesRegex(RuntimeError, 'cuBLAS dependency'): self.run_audit()

    def test_cublas_api_symbol(self):
        self.symbols = '                 U cublasGemmEx@libcublas.so.13'
        with self.assertRaisesRegex(RuntimeError, 'cuBLAS API symbol'): self.run_audit()

    def test_native_source_basename_is_not_vendor_api(self):
        self.symbols = '0000000 t __device_stub__fp8_dequant_cublas_projection_cu_native_kernel()'
        self.assertTrue(self.run_audit()['passed'])

    def test_escaping_path(self):
        self.value['executable'] = '../outside'
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, 'executable path'): self.run_audit()

    def test_missing_license(self):
        (self.data / 'licenses/LICENSE.fla.txt').unlink()
        with self.assertRaisesRegex(RuntimeError, 'missing license'): self.run_audit()

    def test_wrong_help_cap(self):
        self.help = self.help.replace('cap: 2', 'cap: 3')
        with self.assertRaisesRegex(RuntimeError, 'help identity'): self.run_audit()


if __name__ == '__main__':
    unittest.main()
