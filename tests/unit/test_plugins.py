from calibrex.graph.factors import (
    FactorDescriptor,
    FactorPlugin,
    get_factor_plugin,
    register_factor,
)


def test_builtin_factor_registry() -> None:
    plugin = get_factor_plugin("radar_doppler_motion")()
    descriptors = plugin.build({"sensor": "radar0"})
    assert descriptors[0].residual == "doppler_velocity_residual_mps"


def test_register_custom_factor() -> None:
    @register_factor("unit_test_custom_factor")
    class CustomFactor(FactorPlugin):
        def build(self, context: dict[str, object]) -> list[FactorDescriptor]:
            return [FactorDescriptor(name="unit_test_custom_factor", variables=[], residual="unit")]

    assert get_factor_plugin("unit_test_custom_factor") is CustomFactor
