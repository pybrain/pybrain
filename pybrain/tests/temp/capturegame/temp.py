from pybrain.tools.xml import NetworkReader

fn = 'e10-CaptureGameNetwork-s4-h1--851152'
net = NetworkReader.readFrom(fn)
print net
print net.params
print net.args['RUNRES']
