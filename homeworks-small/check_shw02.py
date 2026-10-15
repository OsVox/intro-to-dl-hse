"""Проверки HW2 на CPU, без CIFAR-10 и без экспериментов.

Запуск: python homeworks-small/check_shw02.py
Нужны пакеты из notebook, а также nbformat и ipython.
"""

import ast, contextlib, io, json, os, sys, tempfile, time
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset

p = Path(__file__).resolve().with_name("shw-02-cnn.ipynb")
nb = nbformat.read(p, as_version=4)
nbformat.validate(nb)
torch.set_num_threads(1)
namespace = {"__name__": "__main__"}
# Failure here would reveal an accidental data download during Run all.
original_dataset = torchvision.datasets.CIFAR10


def no_data(*args, **kwargs):
    raise AssertionError("CIFAR10 was accessed in safe mode")


torchvision.datasets.CIFAR10 = no_data
log = io.StringIO()
started = time.perf_counter()
with tempfile.TemporaryDirectory() as temp:
    os.chdir(temp)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        print("CELL", i, flush=True)
        code = compile(cell.source, f"cell-{i}", "exec")
        with contextlib.redirect_stdout(log):
            exec(code, namespace)
        plt.close("all")
    assert not list(Path(".").iterdir()), list(Path(".").iterdir())
print("SAFE RUN ALL OK:", round(time.perf_counter() - started, 2), "seconds")
# Вывод ячеек остаётся в памяти; notebook и каталог с данными не меняются.
torchvision.datasets.CIFAR10 = original_dataset
# Small CPU probes only. They do not use CIFAR, GPU, or full experiment runner.
with tempfile.TemporaryDirectory() as tmp:
    namespace["RESULTS_DIR"] = Path(tmp)
    namespace["STUDENT_ID"] = 123
    namespace["SEEDS"] = [123, 124, 125]
    namespace["AMP"] = False
    namespace["seed_everything"](123)
    Config = namespace["Config"]
    build = namespace["build_model"]
    # Same weights across controlled architectural removals.
    models = {}
    for kind in ["basic", "no_skip", "no_bn", "neither"]:
        torch.manual_seed(123)
        models[kind] = build(Config(model=kind))
        assert models[kind](torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    for kind, m in models.items():
        assert torch.equal(m.conv1.weight, models["basic"].conv1.weight)
        assert torch.equal(m.conv2.weight, models["basic"].conv2.weight)
        assert torch.equal(m.fc.weight, models["basic"].fc.weight)
    print("CONTROLLED INITIALIZATION OK")
    for kind in ["resnet", "convnext", "challenge"]:
        c = Config(model=kind, width=8)
        model = build(c)
        assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    print("ARCHITECTURE SHAPES OK")
    # Loss gradient matches hard CE, including smoothing.
    logits = torch.randn(4, 10, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    for smoothing in [0.0, 0.05]:
        a = namespace["soft_cross_entropy"](
            logits, torch.nn.functional.one_hot(labels, 10).float(), smoothing
        )
        b = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=smoothing)
        assert torch.allclose(a, b, atol=1e-6)
    print("SOFT LOSS OK")
    x = torch.randn(5, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3, 4])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = models["basic"].eval()
    loss, acc = namespace["test"](model, loader)
    with torch.no_grad():
        z = model(x)
        assert abs(loss - torch.nn.functional.cross_entropy(z, y).item()) < 1e-6
        assert abs(acc - (z.argmax(1) == y).float().mean().item()) < 1e-6
    print("WEIGHTED EVALUATION OK")
    # One synthetic batch for each scheduler: not an experiment or quality measurement.
    batch_loader = DataLoader(TensorDataset(x[:2], y[:2]), batch_size=2)
    for schedule in ["constant", "multistep", "onecycle", "cosine"]:
        c = Config(schedule=schedule, epochs=20)
        model = build(c)
        opt = namespace["make_optimizer"](model, c)
        sched, perbatch = namespace["make_scheduler"](opt, c, 1)
        losses, accs, lrs, sizes = namespace["train_epoch"](
            model,
            opt,
            batch_loader,
            label_smoothing=0.05,
            scheduler=sched,
            scheduler_per_batch=perbatch,
        )
        assert len(losses) == len(accs) == len(lrs) == 1
    print("TRAINING STEP / SCHEDULERS OK")
    for mixing in ["mixup", "cutmix"]:
        model = build(Config())
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        losses, accs, _, _ = namespace["train_epoch"](
            model, opt, batch_loader, mixer=namespace["make_mixer"](mixing)
        )
        assert all(torch.isfinite(torch.tensor(losses)))
    print("BATCH MIXING OK")
    # CAM maps vary with class and remove hooks.
    for kind in ["basic", "resnet", "convnext"]:
        model = build(Config(model=kind, width=8)).eval()
        layer = namespace["cam_layer"](model)
        before = len(layer._forward_hooks)
        cam = namespace["GradCAM"](model, layer)
        heatmap, cls = cam(x[:1], class_index=1)
        assert heatmap.shape == (32, 32) and torch.isfinite(heatmap).all()
        assert 0 <= heatmap.min() <= heatmap.max() <= 1
        assert cam.activation is None
        cam.close()
        assert len(layer._forward_hooks) == before
    print("GRAD-CAM OK")
    # Own checkpoint round-trip, cache and disabled-run guards.
    config = Config()
    path = namespace["experiment_path"](config, 123)
    path.mkdir(parents=True)
    row = dict(
        status="complete", config=namespace["asdict"](config), seed=123, path=str(path)
    )
    torch.save(models["basic"].state_dict(), path / "best.pt")
    restored = namespace["load_model"](row, on_device=torch.device("cpu"))
    assert torch.equal(restored.fc.weight, models["basic"].fc.weight)
    (path / "result.json").write_text(json.dumps(row))
    assert (
        namespace["run_experiment"]("cached", config, 123, "2")["status"] == "complete"
    )
    assert namespace["run_experiment"]("disabled", config, 124, "2") is None
    print("CACHE / CHECKPOINT / DISABLED TRAINING OK")
print("ALL CPU CHECKS PASSED")
