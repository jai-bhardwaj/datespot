import sys
import tensorhub

def load_dataset(cdl_file):
    try:
        return tensorhub.CreateCDLFromJSON(cdl_file)
    except Exception as e:
        print(f"**** error: {sys.argv[0]} could not parse CDL file {cdl_file}")
        sys.exit()

def create_network(cdl, dataset_list):
    cdl_mode = tensorhub.GetCDLMode(cdl)
    network_filename = tensorhub.GetCDLNetworkFileName(cdl)
    batch_size = tensorhub.GetCDLBatch(cdl)

    if cdl_mode == "Prediction":
        return tensorhub.LoadNeuralNetworkNetCDF(network_filename, batch_size)
    else:
        return tensorhub.LoadNeuralNetworkJSON(network_filename, batch_size, dataset_list)

def train_network(network, cdl):
    checkpoint_file_name, checkpoint_interval = tensorhub.GetCheckpoint(network)
    print(f"**** checkpoint filename = {checkpoint_file_name}")
    print(f"**** checkpoint interval = {checkpoint_interval}")

    optimizer = tensorhub.GetCDLOptimizer(cdl)
    alpha = tensorhub.GetCDLAlpha(cdl)
    lambda1 = 0.0
    mu1 = 0.0
    epochs = 0

    while epochs < tensorhub.GetCDLEpochs(cdl):
        alpha_interval = tensorhub.GetCDLAlphaInterval(cdl)
        tensorhub.Train(network, alpha_interval, alpha, tensorhub.GetCDLLambda(cdl), lambda1, tensorhub.GetCDLMu(cdl), mu1)
        alpha *= tensorhub.GetCDLAlphaMultiplier(cdl)
        epochs += alpha_interval

    results_filename = tensorhub.GetCDLResultsFileName(cdl)
    print(f"**** saving training results to file: {results_filename}")
    tensorhub.SaveNetCDF(network, results_filename)

def predict_output(network, dataset_list, cdl, K=10):
    output = tensorhub.PredictOutput(network, dataset_list, cdl, K)
    print(f"**** top {K} results follow:")
    for index, value in enumerate(output, 1):
        print(format(index, '3d'), end=' ')
        for i, x in enumerate(value):
            print(format(x, '.3f'), end=' ')
        print()

def main():
    tensorhub.Startup(sys.argv)
    if len(sys.argv) != 2:
        print(f"**** error: you must provide a CDL file to {sys.argv[0]} (typically either train.cdl or predict.cdl)")
        sys.exit()

    cdl_file = sys.argv[1]
    CDL = load_dataset(cdl_file)
    random_seed = tensorhub.GetCDLRandomSeed(CDL)
    tensorhub.SetRandomSeed(random_seed)

    dataset_list = tensorhub.LoadNetCDF(tensorhub.GetCDLDataFileName(CDL))
    network = create_network(CDL, dataset_list)

    gpu_memory_usage, cpu_memory_usage = tensorhub.GetMemoryUsage()
    print(f"**** GPU memory usage: {gpu_memory_usage} KB")
    print(f"**** CPU memory usage: {cpu_memory_usage} KB")

    tensorhub.LoadDataSets(network, dataset_list)

    cdl_mode = tensorhub.GetCDLMode(CDL)
    if cdl_mode == "Training":
        train_network(network, CDL)
    else:
        predict_output(network, dataset_list, CDL)

    tensorhub.DeleteCDL(CDL)
    for dataset in dataset_list:
        tensorhub.DeleteDataSet(dataset)
    tensorhub.DeleteNetwork(network)
    tensorhub.Shutdown()

if __name__ == "__main__":
    main()
