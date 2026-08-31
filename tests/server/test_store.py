import json

import pytest
import torch

from data.tokenizer import load_tokenizer
from worker import store
from worker.worker import TOKENIZER_PATH


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """Point the store at an isolated temp dir instead of the repo models/."""
    monkeypatch.setattr(store, "MODELS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def tokenizer():
    return load_tokenizer(TOKENIZER_PATH)


def _weights():
    return {"embedding.weight": torch.randn(8, 4), "proj.bias": torch.zeros(4)}


class TestSaveLoad:
    def test_round_trip(self, models_dir, tokenizer):
        w = _weights()
        store.save("m1", tokenizer, w, "hash_abc")

        pkg = store.load("m1")
        assert pkg is not None
        assert pkg.id == "m1"
        assert pkg.structure_hash == "hash_abc"
        assert pkg.tokenizer.get_vocab_size() == tokenizer.get_vocab_size()
        assert pkg.tokenizer.encode("hello").ids == tokenizer.encode("hello").ids
        for k, v in w.items():
            assert torch.equal(pkg.weights[k], v)

    def test_load_unknown_returns_none(self, models_dir):
        assert store.load("nope") is None

    def test_layout_on_disk(self, models_dir, tokenizer):
        store.save("m1", tokenizer, _weights(), "h")
        d = models_dir / "m1"
        assert (d / "tokenizer.json").exists()
        assert (d / "weights.pt").exists()
        assert (d / "meta.json").exists()
        meta = json.loads((d / "meta.json").read_text())
        assert meta == {"structure_hash": "h", "status": "trained"}


class TestExists:
    def test_true_after_save(self, models_dir, tokenizer):
        store.save("m1", tokenizer, _weights(), "h")
        assert store.exists("m1") is True

    def test_false_for_unknown(self, models_dir):
        assert store.exists("ghost") is False

    def test_incomplete_package_reads_as_absent(self, models_dir, tokenizer):
        # a dir with artifacts but no committed meta (crash mid-write) is not valid
        store.save("m1", tokenizer, _weights(), "h")
        (models_dir / "m1" / "meta.json").unlink()
        assert store.exists("m1") is False
        assert store.load("m1") is None

    def test_writing_status_reads_as_absent(self, models_dir, tokenizer):
        store.save("m1", tokenizer, _weights(), "h")
        meta_path = models_dir / "m1" / "meta.json"
        meta_path.write_text(json.dumps({"structure_hash": "h", "status": "writing"}))
        assert store.exists("m1") is False


class TestListIds:
    def test_empty(self, models_dir):
        assert store.list_ids() == []

    def test_lists_committed(self, models_dir, tokenizer):
        store.save("b", tokenizer, _weights(), "h")
        store.save("a", tokenizer, _weights(), "h")
        assert store.list_ids() == ["a", "b"]

    def test_excludes_incomplete(self, models_dir, tokenizer):
        store.save("good", tokenizer, _weights(), "h")
        store.save("bad", tokenizer, _weights(), "h")
        (models_dir / "bad" / "meta.json").unlink()
        assert store.list_ids() == ["good"]


class TestDelete:
    def test_removes_package(self, models_dir, tokenizer):
        store.save("m1", tokenizer, _weights(), "h")
        assert store.delete("m1") is True
        assert store.exists("m1") is False
        assert not (models_dir / "m1").exists()

    def test_unknown_returns_false(self, models_dir):
        assert store.delete("ghost") is False


class TestResave:
    def test_overwrites_with_new_hash(self, models_dir, tokenizer):
        store.save("m1", tokenizer, _weights(), "old_hash")
        new_w = {"embedding.weight": torch.ones(8, 4), "proj.bias": torch.ones(4)}
        store.save("m1", tokenizer, new_w, "new_hash")

        pkg = store.load("m1")
        assert pkg.structure_hash == "new_hash"
        assert torch.equal(pkg.weights["embedding.weight"], torch.ones(8, 4))
