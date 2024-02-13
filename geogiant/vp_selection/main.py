from common.files_utils import load_json
from common.settings import PathSettings
from geogiant.vp_selection import VPSelectionDNS, VPSelectionMultiIteration

path_settings = PathSettings()

if __name__ == "__main__":
    vp_selection_dns = True
    vp_selection_multi = True

    targets = load_json(path_settings.OLD_TARGETS)
    vps = load_json(path_settings.OLD_VPS)

    if vp_selection_dns:
        VPSelectionDNS().main(
            targets=targets,
            vps=vps,
            output_path="ecs_dns",
            target_selection=True,
            subnet_selection=False,
        )

    if vp_selection_multi:
        VPSelectionMultiIteration().main(
            targets=targets,
            vps=vps,
            output_path="multi_iteration",
            target_selection=True,
            subnet_selection=False,
        )
