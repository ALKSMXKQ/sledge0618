__all__ = [
    "CompositionalSemanticSceneEditor",
    "HazardSemanticSpec",
    "load_spec",
    "load_specs_from_dir",
]


def __getattr__(name):
    if name == "CompositionalSemanticSceneEditor":
        from sledge.semantic_control.compositional_editor import CompositionalSemanticSceneEditor

        return CompositionalSemanticSceneEditor
    if name == "HazardSemanticSpec":
        from sledge.semantic_control.hazard_spec import HazardSemanticSpec

        return HazardSemanticSpec
    if name == "load_spec":
        from sledge.semantic_control.spec_io import load_spec

        return load_spec
    if name == "load_specs_from_dir":
        from sledge.semantic_control.spec_io import load_specs_from_dir

        return load_specs_from_dir
    raise AttributeError(name)
