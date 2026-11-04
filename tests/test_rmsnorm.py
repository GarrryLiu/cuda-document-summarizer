"""Public opt-in integration: lifecycle, fallback and GPU dispatch."""
import unittest
from cuda_summary.rmsnorm import rmsnorm_backend


class APIWithoutGPU(unittest.TestCase):
    def test_native_needs_no_model_or_dependencies(self):
        with rmsnorm_backend(None) as stats:
            self.assertEqual(stats['modules'], 0)

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            with rmsnorm_backend(None, 'typo'):
                pass


try:
    import torch
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'GPU extra not installed')
class Integration(unittest.TestCase):
    def test_cpu_grad_training_and_exception_restore(self):
        model = Qwen2RMSNorm(2048).eval()
        x = torch.randn(1, 3, 2048, requires_grad=True)
        expected = model(x)
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with rmsnorm_backend(model, 'cuda') as stats:
                model(x).sum().backward()
                self.assertIsNotNone(x.grad)
                with torch.inference_mode():
                    torch.testing.assert_close(model(x), expected)
                    model.train()
                    model(x)
                self.assertEqual(stats['replaced'], 0)
                self.assertEqual(stats['fallback'], 3)
                raise RuntimeError('sentinel')
        self.assertNotIn('forward', model.__dict__)

    def test_existing_override_and_nested_context_preserved(self):
        model = Qwen2RMSNorm(2048).eval()
        with rmsnorm_backend(model, 'cuda') as outer:
            installed = model.forward
            with rmsnorm_backend(model, 'compiled') as inner:
                self.assertEqual(inner['modules'], 0)
                self.assertIs(model.forward, installed)
            self.assertIs(model.forward, installed)
        self.assertEqual(outer['modules'], 1)
        self.assertNotIn('forward', model.__dict__)

    def test_cuda_prefill_and_native_decode(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        model = Qwen2RMSNorm(2048).cuda().half().eval()
        x = torch.randn(2, 17, 2048, device='cuda', dtype=torch.float16)
        with torch.inference_mode():
            expected = model(x)
            decode = x[:, :1].contiguous()
            native_decode = model(decode)
            with rmsnorm_backend(model, 'cuda') as stats:
                torch.testing.assert_close(model(x), expected, atol=.008, rtol=.002)
                self.assertTrue(torch.equal(model(decode), native_decode))
                model.train()
                self.assertTrue(torch.equal(model(x), expected))
                self.assertEqual(stats['replaced'], 1)
                self.assertEqual(stats['fallback'], 2)
        self.assertNotIn('forward', model.__dict__)
