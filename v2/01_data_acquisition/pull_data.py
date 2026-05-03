from roboflow import Roboflow
rf = Roboflow(api_key="8XQTfGjudP24VUAJUVkA")
project = rf.workspace("cico-siefo").project("durian-k51j3")
version = project.version(1)
dataset = version.download("yolov11")