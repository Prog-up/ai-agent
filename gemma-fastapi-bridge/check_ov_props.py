import openvino as ov
core = ov.Core()
print(core.get_property("CPU", "SUPPORTED_PROPERTIES"))
