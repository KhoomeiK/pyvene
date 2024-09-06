import pyvene as pv
from PIL import Image

config, processor, blip = pv.create_blip_itm()
pv_config = pv.IntervenableConfig(
    {"layer": 0, "component": "mlp_output"}, intervention_types=pv.VanillaIntervention
)
pv_blip = pv.IntervenableModel(pv_config, model=blip)

# load images into PIL
image_1 = Image.open("tests/umbrella1.jpeg")
image_2 = Image.open("tests/umbrella2.jpeg")

# run an interchange intervention
intervened_outputs = pv_blip(
    base=processor(image_1, "someone holding a yellow umbrella wearing a white dress", return_tensors="pt"),
    sources=processor(image_1, "someone holding a white umbrella wearing a yellow dress", return_tensors="pt"),
    # the location to intervene at (3rd token)
    unit_locations={"sources->base": 3},
    # the individual dimensions targeted
    subspaces=[10, 11, 12],
)

intervened_outputs = pv_blip(
    base=processor(image_2, "someone holding a white umbrella wearing a yellow dress", return_tensors="pt"),
    sources=processor(image_2, "someone holding a yellow umbrella wearing a white dress", return_tensors="pt"),
    # the location to intervene at (3rd token)
    unit_locations={"sources->base": 3},
    # the individual dimensions targeted
    subspaces=[10, 11, 12],
)