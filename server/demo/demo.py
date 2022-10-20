import os
import sys
import time
from prometheus_client import start_http_server, Gauge
from prometheus_client.core import GaugeMetricFamily, REGISTRY

pipeline_name = 'KerasCompFullPipeline'
util_path = os.path.join(os.path.dirname(__file__), 'util')
sys.path.append(util_path)
train_path = os.path.join(os.path.dirname(__file__), 'train')
sys.path.append(train_path)
prom_path = os.path.join(os.path.dirname(__file__), 'prom')
sys.path.append(prom_path)
sys.path.append("..")

from prom.query import PrometheusClient, NODE_STAT_QUERY, PROM_QUERY_INTERVAL
from util.config import getConfig

SAMPLING_INTERVAL = PROM_QUERY_INTERVAL
SAMPLING_INTERVAL = getConfig('SAMPLING_INTERVAL', SAMPLING_INTERVAL)
SAMPLING_INTERVAL = int(SAMPLING_INTERVAL)

#measured_core = Gauge("current_rapl_core_joules", "Measured Core joules of Kepler Workload")
#measured_dram = Gauge("current_rapl_dram_joules", "Measured Dram joules of Kepler Workload")
#predicted_core = Gauge("current_predicted_core_joules", "Predicted Core joules of Kepler Workload")
#predicted_dram = Gauge("current_predicted_dram_joules", "Measured Dram joules of Kepler Workload")


class Demo_Metric(object):
    def __init__(self):
        pass
    def collect(self):
        gauge = GaugeMetricFamily("energy_metric", "Contains predicted/measured core and predicted/measured dram in joules", labels=["predicted_core", "predicted_dram", "measured_core", "measured_dram"])
         # retrieve measured node metrics and performance counters
        node_data = prom_client.get_data(NODE_STAT_QUERY, None)
        # all in order
        measured_core_joule_list = node_data['energy_in_core_joule'].tolist()
        measured_dram_joule_list = node_data['energy_in_dram_joule'].tolist()

        # retrieve predicted core and dram energy metrics
        # all in order
        predicted_results = pipeline.predict(node_data)
        predicted_core_joule_list = [float(x[0]) for x in predicted_results['core']]
        predicted_dram_joule_list = [float(x[0]) for x in predicted_results['dram']]

        for index in range(len(measured_core_joule_list)):
            #gauge.add_metric(['predicted_core'], float(predicted_core_joule_list[index]))
            #gauge.add_metric(['predicted_dram'], float(predicted_dram_joule_list[index]))
            #gauge.add_metric(['measured_core'], float(measured_core_joule_list[index]))
            #gauge.add_metric(['measured_dram'], float(measured_dram_joule_list[index]))
            gauge.add_metric(['predicted_core'], 10)
            gauge.add_metric(['predicted_dram'], 20)
            gauge.add_metric(['measured_core'], 30)
            gauge.add_metric(['measured_dram'], 40)
        yield gauge


import importlib
pipeline_module = importlib.import_module('train.pipelines.{}.pipe'.format(pipeline_name))
pipeline = getattr(pipeline_module, pipeline_name)()

if __name__ == "__main__":
    start_http_server(8000)
    prom_client = PrometheusClient()
    counter = 0
    REGISTRY.register(Demo_Metric())
    while True:
        prom_client.query()
        if counter == 10:
            # retraining occurs after every 10 iterations (call before querying prometheus again)
            pipeline.train(prom_client)
            counter = 0
        # retrieve measured node metrics and performance counters
        #node_data = prom_client.get_data(NODE_STAT_QUERY, None)
        # all in order
        #measured_core_joule_list = node_data['energy_in_core_joule'].tolist()
        #measured_dram_joule_list = node_data['energy_in_dram_joule'].tolist()

        # retrieve predicted core and dram energy metrics
        # all in order
        #predicted_results = pipeline.predict(node_data)
        #predicted_core_joule_list = [float(x[0]) for x in predicted_results['core']]
        #predicted_dram_joule_list = [float(x[0]) for x in predicted_results['dram']]

        #for index in range(len(measured_core_joule_list)):
        #    measured_core.set(float(measured_core_joule_list[index]))
        #    measured_dram.set(float(measured_dram_joule_list[index]))
        #    predicted_core.set(float(predicted_core_joule_list[index]))
        #    predicted_dram.set(float(predicted_dram_joule_list[index]))
        time.sleep(SAMPLING_INTERVAL)
        counter += 1
