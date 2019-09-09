import sys
import numpy as np
from collections import namedtuple
import functools

from .tflite import Model
from .tflite.BuiltinOperator import BuiltinOperator
from .tflite.TensorType import TensorType
from flatbuffers.number_types import UOffsetTFlags
import tensorflow.lite as tf_lite
from tqdm import tqdm
from prettytable import PrettyTable

from sklearn import cluster


def cluster_weights(weights, n_clusters):
    kmeans = cluster.KMeans(n_clusters=n_clusters).fit(weights.reshape((-1, 1)))
    return kmeans.labels_.reshape(weights.shape), np.around(kmeans.cluster_centers_).astype(np.int32)


# Flatbuffers provide a per-byte view on data, so we need to cast the underlying buffer to the correct datatype
def get_buffer_as_numpy(tensor, buffer):
    if tensor.Type() == TensorType.UINT8:
        arr = buffer.DataAsNumpy()
    elif tensor.Type() == TensorType.INT16:
        arr = np.frombuffer(buffer.DataAsNumpy(), dtype=np.dtype(np.int16).newbyteorder("<"))
    elif tensor.Type() == TensorType.INT32:
        arr = np.frombuffer(buffer.DataAsNumpy(), dtype=np.dtype(np.int32).newbyteorder("<"))
    elif tensor.Type() == TensorType.INT64:
        arr = np.frombuffer(buffer.DataAsNumpy(), dtype=np.dtype(np.int64).newbyteorder("<"))
    else:
        raise NotImplementedError()
    return arr.reshape(tensor.ShapeAsNumpy())


def get_buffer_element_size(t):
    sizes = {
        TensorType.UINT8: 1,
        TensorType.INT16: 2,
        TensorType.INT32: 4,
        TensorType.INT64: 8,
    }
    return sizes[t]


class TFLiteTensor:
    def __init__(self, id=None, shape=None, name=None, is_constant=False, producer=None,
                 consumers=None, predecessors=None, type=None):
        self.id = id
        self.shape = shape
        self.name = name
        self.is_constant = is_constant
        self.producer = producer
        self.consumers = consumers if consumers is not None else []
        self.predecessors = predecessors
        self.type = type

    @property
    def size(self):
        return 0 if self.is_constant else np.prod(self.shape) * get_buffer_element_size(self.type)

    def __hash__(self):
        return hash(self.id)


class TFLiteOperator:
    def __init__(self, id=None, output=None, inputs=None):
        self.id = id
        self.output = output
        self.inputs = inputs if inputs is not None else []

    def __hash__(self):
        return hash(self.id)


TFLiteGraph = namedtuple("TFLiteGraph", ["tensors", "operators", "inputs", "outputs"])


class TFLiteModel:
    def __init__(self, model_bytes):
        self.model_bytes = model_bytes
        self.model_graph = None
        self.peak_usage = None

    @classmethod
    def create_from_protobuf(cls, protobuf_file, inputs, outputs, input_shapes):
        converter = tf_lite.TFLiteConverter.from_frozen_graph(protobuf_file, input_arrays=inputs,
                                                              output_arrays=outputs, input_shapes=input_shapes)
        from tensorflow.lite.python import lite_constants
        converter.inference_type = lite_constants.QUANTIZED_UINT8
        converter.inference_input_type = lite_constants.QUANTIZED_UINT8
        # converter.optimizations = [tf_lite.Optimize.DEFAULT]
        input_arrays = converter.get_input_arrays()
        converter.quantized_input_stats = {input_arrays[0]: (0, 1)}  # mean, std_dev
        return cls(bytearray(converter.convert()))

    @classmethod
    def load_from_file(cls, model_path):
        with open(model_path, 'rb') as f:
            return cls(bytearray(f.read()))

    def write_to_file(self, output_path):
        with open(output_path, "wb") as f:
            f.write(self.model_bytes)

    def cluster_weights(self, weight_clusters):
        print(f"Clustering weights into {weight_clusters} clusters...")
        weights = self._discover_tflite_weights()
        for b_index, weight in weights:
            assignments, centroids = cluster_weights(weight, weight_clusters)
            self._overwrite_flatbuffers_buffer(b_index, np.squeeze(centroids[assignments], axis=-1))

    def _overwrite_flatbuffers_buffer(self, buffer_idx, new_contents):
        model = Model.Model.GetRootAsModel(self.model_bytes, 0)
        orig_buffer = model.Buffers(buffer_idx)
        # NB. Update this to directly manipulate `serialized_model` if this view becomes unwriteable
        orig_buffer.DataAsNumpy()[:] = new_contents.astype(np.uint8).flatten()

    def _discover_tflite_weights(self):
        model = Model.Model.GetRootAsModel(self.model_bytes, 0)
        subgraph = model.Subgraphs(0)

        weights = []
        for o in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(o)
            opcode = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            inputs = op.InputsAsNumpy()

            parametrised_opcodes = [BuiltinOperator.CONV_2D, BuiltinOperator.FULLY_CONNECTED, BuiltinOperator.DEPTHWISE_CONV_2D]
            if opcode not in parametrised_opcodes:
                continue

            weight_tensor = subgraph.Tensors(inputs[1])
            buffer_idx = weight_tensor.Buffer()
            buffer = model.Buffers(buffer_idx)
            # Return a buffer index and contents as an ndarray
            weights.append((buffer_idx, get_buffer_as_numpy(weight_tensor, buffer)))

        return weights

    def _build_graph(self):
        model = Model.Model.GetRootAsModel(self.model_bytes, 0)
        subgraph = model.Subgraphs(0)

        tensors = []
        operators = []

        for i in range(subgraph.TensorsLength()):
            t = subgraph.Tensors(i)
            tensors.append(TFLiteTensor(id=i, shape=t.ShapeAsNumpy(), name=t.Name().decode("ascii"),
                                        producer=None, consumers=[], type=t.Type()))

        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            assert op.OutputsLength() <= 1
            has_output = op.OutputsLength() == 1
            inputs = [tensors[j] for j in op.InputsAsNumpy()]
            assert len(inputs) > 0

            tflite_op = TFLiteOperator(id=i, output=tensors[op.Outputs(0)] if has_output else None, inputs=inputs)
            tflite_op.output.producer = tflite_op
            for t in inputs:
                t.consumers.append(tflite_op)
            operators.append(tflite_op)

        inputs = [tensors[j] for j in subgraph.InputsAsNumpy()]
        outputs = [tensors[j] for j in subgraph.OutputsAsNumpy()]

        for t in tensors:
            t.is_constant = (t.producer is None) and (t not in inputs)

        # Can turn into an iterative function if this ever causes performance / stack overflow issues
        def _compute_predecessors(tensor):
            if tensor.predecessors is not None:
                return tensor.predecessors

            if tensor.producer is None:
                tensor.predecessors = set()
            else:
                op_inputs = tensor.producer.inputs
                tensor.predecessors = set(op_inputs)
                for i in op_inputs:
                    tensor.predecessors |= _compute_predecessors(i)
            return tensor.predecessors

        for o in outputs:
            _compute_predecessors(o)  # Will recursively compute predecessors for all nodes leading up to output nodes

        self.model_graph = TFLiteGraph(tensors, operators, inputs, outputs)

    @staticmethod
    def _cum_tensor_sizes(tensors):
        return sum(t.size for t in tensors)

    def peak_mem_usage(self):
        if self.peak_usage is not None:
            return self.peak_usage

        if not self.model_graph:
            self._build_graph()
        g = self.model_graph

        # Can turn into an iterative function if this ever causes performance / stack overflow issues
        @functools.lru_cache(maxsize=None)
        def mem(tensors):
            # Computes the peak memory usage of a runtime system that computes all tensors in a set `tensors`.
            constants = [t for t in tensors if t.producer is None]
            if constants:
                upstream_mem_use, schedule = mem(frozenset(t for t in tensors if t.producer is not None))
                return TFLiteModel._cum_tensor_sizes(constants) + upstream_mem_use, schedule
            if not tensors:
                return 0, []

            min_use = sys.maxsize  # A reasonably large integer
            schedule = []
            # For each of tensors in our working set, we try to unapply the operator that produced it
            for t in tensors:
                rest = tensors - {t}
                # We constrain the search to never consider evaluating an operator (`t.producer`) more than once ---
                # so we prevent cases where we consider unapplying `t.producer` but it's actually necessary for other
                # tensors in the working set.
                if any(t in r.predecessors for r in rest):
                    continue
                inputs = frozenset(t.producer.inputs)
                new_set = rest | inputs
                upstream_mem_use, operators = mem(new_set)

                tensors_in_memory = new_set | {t}
                mem_use = max(upstream_mem_use, TFLiteModel._cum_tensor_sizes(tensors_in_memory))
                if mem_use < min_use:
                    min_use = mem_use
                    schedule = operators + [t.producer]
            return min_use, schedule

        self.peak_usage = mem(frozenset(g.outputs))
        return self.peak_usage

    def evaluate(self, test_data):
        interpreter = tf_lite.Interpreter(model_content=bytes(self.model_bytes))
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        input_index = input_info["index"]
        scale, offset = input_info["quantization"]
        output_index = interpreter.get_output_details()[0]["index"]

        correct = 0
        test_data.reset_state()
        for img, label in tqdm(test_data):
            interpreter.set_tensor(input_index, np.expand_dims(img.astype(np.uint8), axis=0))
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_index)
            if predictions.argmax() == label.argmax():
                correct += 1
        print(f"{correct} classified correctly out of {len(test_data)} ({correct / len(test_data) * 100:.2f}%)")

    def _print_execution_schedule(self):
        if not self.model_graph:
            self._build_graph()
        g = self.model_graph

        # Compute tensor lifetimes
        num_operators = len(g.operators)
        first_used_at = {t: t.producer.id if t.producer is not None else 0 for t in g.tensors}
        last_used_at = {t: max(op.id for op in t.consumers) if t.consumers else num_operators for t in g.tensors}

        x = PrettyTable()
        x.field_names = ["Operator (output name)", "Tensors in memory (IDs)", "Memory use (B)"]
        x.align["Memory use (B)"] = "r"

        peak_mem_use = 0
        print("Operator execution schedule:")
        for op in g.operators:
            tensors = {t for t in g.tensors if first_used_at[t] <= op.id <= last_used_at[t]}
            mem_use = TFLiteModel._cum_tensor_sizes(tensors)
            peak_mem_use = max(mem_use, peak_mem_use)
            x.add_row([op.output.name, f"[{', '.join(str(t.id) for t in tensors if t.size != 0)}]", f"{mem_use:,}"])

        print(x)
        print(f"Current peak memory usage: {peak_mem_use:,} B")
        print()

    def _print_tensor_details(self):
        if not self.model_graph:
            self._build_graph()

        x = PrettyTable()
        x.field_names = ["Id", "Tensor", "Shape", "Size in RAM (B)"]
        x.align["Id"] = "r"
        x.align["Size in RAM (B)"] = "r"

        for t in self.model_graph.tensors:
            if t.size != 0:
                x.add_row([t.id, t.name, tuple(t.shape), f"{t.size:,}"])

        print("Tensor information (weights excluded):")
        print(x)
        print()

    def print_model_analysis(self):
        self._print_tensor_details()
        self._print_execution_schedule()

    def optimize_memory(self):
        _, schedule = self.peak_mem_usage()
        num_operators = len(self.model_graph.operators)
        correctly_ordered = all(i == schedule[i].id for i in range(num_operators))
        if correctly_ordered:
            print("The model already has optimal operator arrangement.")
            return

        # Proceed reordering the operators by changing the indirection table
        model = Model.Model.GetRootAsModel(self.model_bytes, 0)
        subgraph = model.Subgraphs(0)
        indirection_table_offset = UOffsetTFlags.py_type(subgraph._tab.Offset(10))
        indirection_table = subgraph._tab.GetVectorAsNumpy(UOffsetTFlags, indirection_table_offset)
        old_indirection_table = indirection_table.copy()

        for i in range(num_operators):
            # Operator #op_id should go into position i
            op_id = schedule[i].id
            indirection_table[i] = old_indirection_table[op_id] + 4 * (op_id - i)
            schedule[i].id = i

        # Patch up model_graph instead of rebuilding it
        self.model_graph.operators.sort(key=lambda op: op.id)
