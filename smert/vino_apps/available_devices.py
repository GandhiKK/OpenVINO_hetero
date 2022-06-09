from openvino.inference_engine import IECore
import numpy as np

ie = IECore()

devices = ie.available_devices
for device in devices:
    device_name = ie.get_metric(device_name=device, metric_name="FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")
    
    
classification_model_xml = "D:\\Extra\\pyDev\\smert\\vino_apps\\model\\horizontal-text-detection-0001.xml"
net = ie.read_network(model=classification_model_xml)
exec_net = ie.load_network(network=net, device_name="CPU")

input_layer = next(iter(net.input_info))
print(input_layer)

# d = {'CPU': [], 'GPU': [], 'HETERO': []}
# d['CPU'].append(1)
# d['CPU'].append(2)
# d['GPU'].append(3)
# d['HETERO'].append(0)

# print(d)

# for i in d:
#     print(f'{i}: {str(np.mean(np.array(d[i])))}')

# # a = str(np.mean(np.array(d['CPU'])))
# # print(a)
# # print(np.mean(np.array(d['GPU'])))
# # print(np.mean(np.array(d['HETERO'])))

# print(str(np.round(34.141313513, 2)))