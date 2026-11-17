from b_replay import delivery
import json
import pytest

def test_nested_release_manifests_are_included(tmp_path,monkeypatch):
    monkeypatch.setattr(delivery,"ROOT",tmp_path)
    own=tmp_path/"release_manifest.json";own.write_text("{}")
    nested=tmp_path/"runtime/Kairos/provenance/release_manifest.json"
    nested.parent.mkdir(parents=True);nested.write_text("{}")
    snapshot=tmp_path/"sources/kairos/provenance/release_manifest.json"
    snapshot.parent.mkdir(parents=True);snapshot.write_text("{}")
    files=delivery.release_files()
    assert own not in files and nested in files and snapshot in files


def test_partial_gpu_pilot_evidence_cannot_validate_release(tmp_path, monkeypatch):
    monkeypatch.setattr(delivery, "ROOT", tmp_path)
    path=tmp_path/"verification/gpu_verification.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(status="passed", runs=[dict(programs=1, calls=10,
        output_tokens=2437, strict_complete=True, fixed_inputs_and_output_counts_verified=True)])))
    with pytest.raises(ValueError, match="complete 1/4/14"):
        delivery.gpu_evidence()
