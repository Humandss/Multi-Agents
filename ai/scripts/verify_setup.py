"""GPU/CUDA/bitsandbytes 동작 확인."""

import sys

import torch


def check(name, ok, detail=""):
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {name}: {detail}")
    return ok


def main():
    print("환경 점검")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")

    if not check("CUDA", torch.cuda.is_available(), torch.version.cuda or "사용 불가"):
        print("\nCUDA 안 잡힘. 드라이버 + PyTorch CUDA 설치 확인.")
        return 1

    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    vram_gb = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    check("GPU", True, f"{name} ({vram_gb:.1f} GB)")
    check("bf16", torch.cuda.is_bf16_supported())

    try:
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()
        check("matmul", True, f"shape={tuple(y.shape)}")
    except Exception as e:
        check("matmul", False, str(e))
        return 1

    try:
        import bitsandbytes as bnb
        linear = bnb.nn.Linear4bit(
            512, 512, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
        ).cuda()
        out = linear(torch.randn(2, 512, device="cuda", dtype=torch.bfloat16))
        check("bitsandbytes", True, f"4bit out={tuple(out.shape)}")
    except Exception as e:
        check("bitsandbytes", False, str(e))
        return 1

    try:
        import peft
        import transformers
        check("transformers", True, transformers.__version__)
        check("peft", True, peft.__version__)
    except Exception as e:
        check("transformers/peft", False, str(e))
        return 1

    print("\n전부 OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
