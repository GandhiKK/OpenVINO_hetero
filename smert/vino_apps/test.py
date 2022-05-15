from openvino.inference_engine import IECore

ie = IECore()

devices = ie.available_devices
for device in devices:
    device_name = ie.get_metric(device_name=device, metric_name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
    
    
classification_model_xml = "smert\diploma\model\horizontal-text-detection-0001.xml"
net = ie.read_network(model=classification_model_xml)
exec_net = ie.load_network(network=net, device_name="CPU")

input_layer = next(iter(net.input_info))
print(input_layer)