from common.files_utils import load_json
from common.settings import PathSettings
from vp_selection_algo import (
    VPSelectionDNS,
    VPSelectionMultiIteration,
    VPSelectionDistance,
)

path_settings = PathSettings()

if __name__ == "__main__":
    vp_selection_dns = True
    vp_selection_distance = False
    vp_selection_multi = True

    targets = load_json(path_settings.TARGETS)
    vps = load_json(path_settings.VPS)

    if vp_selection_dns:
        VPSelectionDNS().main(
            targets=targets,
            vps=vps,
            output_path="ecs_dns",
            target_selection=True,
            subnet_selection=False,
        )

    if vp_selection_distance:
        VPSelectionDistance().main(
            targets=targets,
            vps=vps,
            output_path="distance",
            target_selection=True,
            subnet_selection=True,
        )

    if vp_selection_multi:
        VPSelectionMultiIteration().main(
            targets=targets,
            vps=vps,
            output_path="multi_iteration",
            target_selection=True,
            subnet_selection=False,
        )
