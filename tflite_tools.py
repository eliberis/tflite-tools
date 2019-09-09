import argparse

from tflite_tools import TFLiteModel


def main():
    parser = argparse.ArgumentParser(description='TFLite model analyser & memory optimizer')

    parser.add_argument("-i", type=str, dest="input_path", help="input model file (.tflite)")
    parser.add_argument("-o", type=str, dest="output_path", default=None, help="output model file (.tflite)")
    parser.add_argument("--clusters", type=int, default=0,
                        help="cluster weights into n-many values (simulate code-book quantization)")
    parser.add_argument("--optimize", action="store_true", default=False, help="optimize peak working set size")
    args = parser.parse_args()

    # Example API usage:
    # Can also use `TFLiteModel.create_from_protobuf`, which will invoke TOCO.

    model = TFLiteModel.load_from_file(args.input_path)

    if args.optimize:
        print("Optimizing peak memory usage...")
        model.optimize_memory()

    model.print_model_analysis()

    if args.clusters > 0:
        model.cluster_weights(args.clusters)

    # Example API usage:
    # `model.evaluate(<data_iterator>)`
    # where data iterator returns `img`, `label`.
    # `img` will be cast to uint8 and `label` is assumed to be a one-hot vector

    if args.output_path:
        print(f"Saving the model to {args.output_path}...")
        model.write_to_file(args.output_path)


if __name__ == "__main__":
    main()
