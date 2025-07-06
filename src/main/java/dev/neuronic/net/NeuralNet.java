package dev.neuronic.net;

import dev.neuronic.net.common.Utils;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.math.Parallelization;
import dev.neuronic.net.serialization.ModelSerializer;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationService;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;

/**
 * Neural network with clean separation between hidden layers and output layers.
 * <p>
 * Output layers handle their own loss computation, eliminating configuration errors
 * and providing mathematically optimal implementations.
 * <p>
 * Usage:
 * NeuralNet net = NeuralNet.newBuilder()
 * .input(784)
 * .layer(Layers.hiddenDenseRelu(256, optimizer))
 * .layer(Layers.hiddenDenseRelu(128, optimizer))
 * .output(Layers.outputSoftmaxCrossEntropy(10, optimizer))
 * .build();
 */
public class NeuralNet implements Serializable {

    public static NeuralNetBuilder newBuilder() {
        return new NeuralNetBuilder();
    }

    private final Layer[] layers;
    private final ThreadLocal<Layer.LayerContext[]> contextBuffers;
    private final ExecutorService executor;
    private final float globalGradientClipNorm;
    private final FastRandom random;
    private final Long seed; // Store seed for serialization/reproducibility

    private final ThreadLocal<float[][]> singleInputBuffer = ThreadLocal.withInitial(() -> new float[1][]);
    private final ThreadLocal<float[][]> singleTargetBuffer = ThreadLocal.withInitial(() -> new float[1][]);

    private final ThreadLocal<GradientBuffers> gradientBuffers = new ThreadLocal<>();

    NeuralNet(Layer[] layers) {
        this(layers, null, 10.0f, new FastRandom());
    }

    NeuralNet(Layer[] layers, ExecutorService executor) {
        this(layers, executor, 10.0f, new FastRandom());
    }

    NeuralNet(Layer[] layers, ExecutorService executor, float globalGradientClipNorm) {
        this(layers, executor, globalGradientClipNorm, new FastRandom());
    }
    
    NeuralNet(Layer[] layers, ExecutorService executor, float globalGradientClipNorm, FastRandom random) {
        this(layers, executor, globalGradientClipNorm, random, null);
    }
    
    NeuralNet(Layer[] layers, ExecutorService executor, float globalGradientClipNorm, FastRandom random, Long seed) {
        this.layers = Arrays.copyOf(layers, layers.length);
        this.contextBuffers = ThreadLocal.withInitial(() -> new Layer.LayerContext[layers.length]);
        this.executor = executor;
        this.globalGradientClipNorm = globalGradientClipNorm;
        this.random = random;
        this.seed = seed;
    }

    private Layer.LayerContext[] predictStack(float[] input, boolean isTraining) {
        Layer.LayerContext[] contexts = contextBuffers.get();

        Layer.LayerContext output = null;
        for (int x = 0; x < layers.length; x++) {
            float[] currentInput = (output != null) ? output.outputs() : input;
            if (executor != null) {
                output = layers[x].forward(currentInput, isTraining, executor);
            } else {
                output = layers[x].forward(currentInput, isTraining);
            }
            // Store the context directly - layers are responsible for their own buffer management
            contexts[x] = output;
        }

        return contexts;
    }

    private float[] getFinalLayerOutputs(Layer.LayerContext[] stack) {
        return stack[stack.length - 1].outputs();
    }

    public float[] predict(float[] input) {
        return getFinalLayerOutputs(predictStack(input, false));
    }

    /**
     * Predict the most likely class (argmax of output probabilities).
     *
     * @param input input data
     * @return the index of the class with highest probability
     */
    public float predictArgmax(float[] input) {
        return Utils.argmax(predict(input));
    }

    /**
     * Get the indices of the top K most likely classes.
     *
     * @param input input data
     * @param k     number of top classes to return
     * @return array of K class indices, sorted by probability (highest first)
     */
    public float[] predictTopK(float[] input, int k) {
        int[] indices = Utils.topKIndices(predict(input), k);

        float[] result = new float[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = indices[i];
        }

        return result;
    }

    /**
     * Sample a class using temperature-based sampling.
     * Higher temperature = more random, lower temperature = more deterministic.
     *
     * @param input       input data
     * @param temperature sampling temperature (typically 0.1 to 2.0)
     * @return sampled class index
     */
    public float predictWithTemperature(float[] input, float temperature) {
        return SamplingStrategies.sampleWithTemperature(predict(input), temperature, random);
    }

    /**
     * Sample a class from the top K classes using temperature.
     * Combines top-K filtering with temperature-based sampling.
     *
     * @param input       input data
     * @param k           number of top classes to consider
     * @param temperature sampling temperature
     * @return sampled class index
     */
    public float predictSampleTopK(float[] input, int k, float temperature) {
        return SamplingStrategies.sampleTopK(predict(input), k, temperature, random);
    }

    /**
     * Sample a class using nucleus (top-P) sampling with temperature.
     * Dynamically selects from the smallest set of classes whose cumulative probability exceeds P.
     *
     * @param input       input data
     * @param p           cumulative probability threshold (typically 0.9 to 0.95)
     * @param temperature sampling temperature
     * @return sampled class index
     */
    public float predictSampleTopP(float[] input, float p, float temperature) {
        return SamplingStrategies.sampleTopP(predict(input), p, temperature, random);
    }
    
    /**
     * Get the random number generator used by this network.
     * 
     * <p>This FastRandom instance is shared across all layers in the network to ensure
     * consistent randomness behavior. It is used for:
     * <ul>
     *   <li>Weight initialization during network construction</li>
     *   <li>Dropout masks during training</li>
     *   <li>Sampling strategies for language model generation</li>
     *   <li>Any other stochastic operations within layers</li>
     * </ul>
     * 
     * <p>When a seed is specified via {@link NeuralNetBuilder#withSeed(long)}, this
     * random generator will produce deterministic sequences, enabling reproducible
     * training runs and consistent model initialization.
     * 
     * @return the FastRandom instance used by this network
     */
    public FastRandom getRandom() {
        return random;
    }
    
    /**
     * Get the seed used to initialize this network's random number generator.
     * 
     * <p>The seed determines the initial state of the random number generator,
     * which affects:
     * <ul>
     *   <li>Initial weight values for all layers</li>
     *   <li>Initial embedding values for embedding layers</li>
     *   <li>Dropout patterns during training (though these vary per forward pass)</li>
     *   <li>Any other random operations during network construction</li>
     * </ul>
     * 
     * <p>This seed is preserved during serialization, ensuring that models can be
     * saved and loaded while maintaining their exact state. This is particularly
     * important for "always online training" scenarios where training may be
     * interrupted and resumed.
     * 
     * <p>Note: Even with the same seed, training results may vary slightly due to:
     * <ul>
     *   <li>Floating-point rounding differences across platforms</li>
     *   <li>Parallel execution with different thread scheduling</li>
     *   <li>Different batch ordering in training data</li>
     * </ul>
     * 
     * @return the seed used for initialization, or null if no seed was specified
     *         (indicating random initialization based on system time)
     */
    public Long getSeed() {
        return seed;
    }

    /**
     * Train the network on a single input-target pair.
     * Thread-safe - multiple threads can call this simultaneously.
     * Each layer uses ThreadLocal buffers and thread-safe optimizers.
     */
    public void train(float[] input, float[] targets) {
        float[][] inputBatch = singleInputBuffer.get();
        float[][] targetBatch = singleTargetBuffer.get();

        inputBatch[0] = input;
        targetBatch[0] = targets;

        // Use batch training with batch size 1 to ensure gradient accumulation works properly
        trainBatch(inputBatch, targetBatch);
    }

    /**
     * Train the network on a single batch of samples with gradient accumulation.
     * Thread-safe: multiple threads can call this concurrently on the same network.
     * <p>
     * Uses local gradient accumulation to eliminate all synchronization during
     * forward/backward passes, with only a single brief lock when applying gradients.
     *
     * @param batchInputs  inputs for each sample [batchSize][inputSize]
     * @param batchTargets targets for each sample [batchSize][outputSize]
     */
    public void trainBatch(float[][] batchInputs, float[][] batchTargets) {
        if (batchInputs.length != batchTargets.length)
            throw new IllegalArgumentException("Batch inputs and targets must have same batch size");

        if (batchInputs.length == 0)
            return;

        // Get ThreadLocal gradient buffers (reused across batches)
        GradientBuffers buffers = gradientBuffers.get();
        if (buffers == null) {
            buffers = new GradientBuffers(layers);
            gradientBuffers.set(buffers);
        }

        buffers.reset();

        List<float[][]> weightBufs = buffers.weightGradients;
        List<float[]> biasBufs = buffers.biasGradients;

        for (int i = 0; i < batchInputs.length; i++) {
            Layer.LayerContext[] stack = predictStack(batchInputs[i], true);

            // Start with output layer - compute loss gradient
            int outputIdx = layers.length - 1;
            float[] gradient = layers[outputIdx].computeGradientWithTargets(
                    stack, outputIdx, batchTargets[i],
                    (idx, wG, bG) -> accumulateGradients(idx, wG, bG, weightBufs, biasBufs));

            // Continue with hidden layers using normal gradient propagation
            for (int layerIndex = layers.length - 2; layerIndex >= 0; layerIndex--) {
                gradient = layers[layerIndex].computeGradient(stack, layerIndex, gradient,
                        (idx, wG, bG) -> accumulateGradients(idx, wG, bG, weightBufs, biasBufs));
            }
        }

        // Apply gradients with single lock
        synchronized (this) {
            float scale = 1.0f / batchInputs.length;
            applyGradients(weightBufs, biasBufs, scale);
        }
    }


    /**
     * Helper to accumulate gradients into buffers.
     */
    private static void accumulateGradients(int idx, float[][] wG, float[] bG,
                                           List<float[][]> weightBufs, List<float[]> biasBufs) {
        if (wG != null && weightBufs.get(idx) != null) {
            float[][] accum = weightBufs.get(idx);
            for (int row = 0; row < wG.length; row++) {
                NetMath.elementwiseAdd(accum[row], wG[row], accum[row]);
            }
        }

        if (bG != null && biasBufs.get(idx) != null) {
            NetMath.elementwiseAdd(biasBufs.get(idx), bG, biasBufs.get(idx));
        }
    }

    private void applyGradients(List<float[][]> weightBufs, List<float[]> biasBufs, float scale) {
        // --- Step 1: Scale all regular gradients by 1/batchSize ---
        for (int i = 0; i < layers.length; i++) {
            if (weightBufs.get(i) != null) {
                NetMath.scaleMatrixInPlace(weightBufs.get(i), scale);
                NetMath.elementwiseScaleInPlace(biasBufs.get(i), scale);
            }
        }

        // --- Step 2: Compute global norm and apply clipping if needed ---
        if (globalGradientClipNorm > 0) {
            double totalNormSq = 0.0;

            // Calculate norm for regular layers
            for (int i = 0; i < layers.length; i++) {
                if (weightBufs.get(i) != null) {
                    for (float[] row : weightBufs.get(i)) {
                        for (float val : row) {
                            totalNormSq += val * val;
                        }
                    }
                    for (float val : biasBufs.get(i)) {
                        totalNormSq += val * val;
                    }
                }
            }

            // Calculate norm for GradientProvider layers
            for (Layer layer : layers) {
                if (layer instanceof dev.neuronic.net.layers.GradientProvider) {
                    dev.neuronic.net.layers.GradientProvider provider = (dev.neuronic.net.layers.GradientProvider) layer;
                    List<float[][]> providerGradients = provider.getGradients();
                    for (float[][] gradMatrix : providerGradients) {
                        for (float[] gradRow : gradMatrix) {
                            for (float val : gradRow) {
                                totalNormSq += val * val;
                            }
                        }
                    }
                }
            }

            float globalNorm = (float) Math.sqrt(totalNormSq);

            if (globalNorm > globalGradientClipNorm) {
                if (globalNorm > globalGradientClipNorm * 10) {
                    System.err.printf("Warning: Large gradient norm %.2f clipped to %.2f\n", globalNorm, globalGradientClipNorm);
                }
                float clipScale = globalGradientClipNorm / globalNorm;

                // Apply clipping scale to all gradients
                for (int i = 0; i < layers.length; i++) {
                    if (layers[i] instanceof dev.neuronic.net.layers.GradientProvider) {
                        ((dev.neuronic.net.layers.GradientProvider) layers[i]).applyClippingScale(clipScale);
                    } else if (weightBufs.get(i) != null) {
                        NetMath.scaleMatrixInPlace(weightBufs.get(i), clipScale);
                        NetMath.elementwiseScaleInPlace(biasBufs.get(i), clipScale);
                    }
                }
            }
        }

        // --- Step 3: Apply the final (potentially clipped) gradients ---
        for (int i = 0; i < layers.length; i++) {
            if (weightBufs.get(i) != null) {
                layers[i].applyGradients(weightBufs.get(i), biasBufs.get(i));
            } else if (layers[i].getGradientDimensions() == null) {
                // This handles layers like MixedFeatureInputLayer which apply their own gradients
                layers[i].applyGradients(null, null);
            }
        }
    }

    /**
     * Predict outputs for a batch of samples.
     *
     * @param batchInputs inputs for each sample [batchSize][inputSize]
     * @return predictions for each sample [batchSize][outputSize]
     */
    public float[][] predictBatch(float[][] batchInputs) {
        float[][] outputs = new float[batchInputs.length][];
        
        // Use parallelization if executor is available and batch is large enough
        if (executor != null && Parallelization.shouldParallelize(batchInputs.length, executor)) {
            int numThreads = Parallelization.calculateOptimalThreads(batchInputs.length, executor);
            Parallelization.WorkRange[] ranges = Parallelization.splitWork(batchInputs.length, numThreads);
            
            Runnable[] tasks = new Runnable[numThreads];
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                final Parallelization.WorkRange range = ranges[threadId];
                tasks[t] = () -> {
                    for (int i = range.start; i < range.end; i++) {
                        outputs[i] = predict(batchInputs[i]);
                    }
                };
            }
            
            Parallelization.executeParallel(executor, tasks);
        } else {
            // Sequential execution for small batches or no executor
            for (int i = 0; i < batchInputs.length; i++) {
                outputs[i] = predict(batchInputs[i]);
            }
        }

        return outputs;
    }

    /**
     * Get the input layer (first layer) of the network.
     * Useful for introspecting feature configuration in mixed feature models.
     */
    public Layer getInputLayer() {
        return layers[0];
    }

    /**
     * Get the output layer (last layer) of the network.
     * Useful for SimpleNet to determine output type and validation.
     */
    public Layer getOutputLayer() {
        return layers[layers.length - 1];
    }

    /**
     * Get all layers in this network.
     * Useful for accessing optimizers during training callbacks.
     */
    public Layer[] getLayers() {
        return layers;
    }


    // ===============================
    // SAVE/LOAD METHODS
    // ===============================

    /**
     * Save this neural network to a file with compression.
     *
     * @param path file path to save to
     * @throws IOException if save fails
     */
    public void save(java.nio.file.Path path) throws IOException {
        ModelSerializer.save(this, path);
    }

    /**
     * Save this neural network to a file with compression and progress tracking.
     *
     * @param path             file path to save to
     * @param progressCallback optional progress callback (0.0 to 1.0)
     * @throws IOException if save fails
     */
    public void save(java.nio.file.Path path, java.util.function.Consumer<Double> progressCallback) throws IOException {
        ModelSerializer.save(this, path, progressCallback);
    }

    /**
     * Load a neural network from a file.
     *
     * @param path file path to load from
     * @return loaded neural network
     * @throws IOException if load fails
     */
    public static NeuralNet load(java.nio.file.Path path) throws IOException {
        return ModelSerializer.load(path);
    }

    // Serialization implementation

    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        // Write network metadata
        out.writeInt(layers.length);
        
        // Write seed (null seeds are written as -1)
        out.writeLong(seed != null ? seed : -1L);

        // Write each layer
        for (Layer layer : layers) {
            Serializable serializableLayer = (Serializable) layer;
            out.writeInt(serializableLayer.getTypeId());
            serializableLayer.writeTo(out, version);
        }
    }

    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use readFrom(DataInputStream, int) static method instead");
    }

    /**
     * Static method to deserialize a NeuralNet from stream.
     * Required because we need to construct the layers array first.
     */
    public static NeuralNet deserialize(DataInputStream in, int version) throws IOException {
        // Read network metadata
        int layerCount = in.readInt();
        
        // Read seed
        long seedValue = in.readLong();
        Long seed = seedValue == -1L ? null : seedValue;
        FastRandom random = seed != null ? new FastRandom(seed) : new FastRandom();

        // Read each layer
        Layer[] layers = new Layer[layerCount];
        for (int i = 0; i < layerCount; i++) {
            int typeId = in.readInt();
            layers[i] = deserializeLayer(in, typeId, version, random);
        }

        return new NeuralNet(layers, null, 10.0f, random, seed);
    }

    private static Layer deserializeLayer(DataInputStream in, int typeId, int version, FastRandom random) throws IOException {
        // Use centralized serialization service - eliminates tight coupling
        return SerializationService.deserializeLayer(in, typeId, version, random);
    }

    @Override
    public int getSerializedSize(int version) {
        int size = 8; // layerCount + placeholder for old argMaxCount
        for (Layer layer : layers) {
            Serializable serializableLayer = (Serializable) layer;
            size += 4; // typeId
            size += serializableLayer.getSerializedSize(version);
        }
        return size;
    }

    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_NEURAL_NET;
    }

    /**
     * ThreadLocal gradient accumulation buffers to reduce GC pressure.
     */
    private static class GradientBuffers {
        final List<float[][]> weightGradients;
        final List<float[]> biasGradients;

        GradientBuffers(Layer[] layers) {
            weightGradients = new ArrayList<>(layers.length);
            biasGradients = new ArrayList<>(layers.length);

            // Pre-allocate buffers based on layer dimensions
            for (Layer layer : layers) {
                Layer.GradientDimensions dims = layer.getGradientDimensions();
                if (dims != null) {
                    weightGradients.add(new float[dims.weightRows()][dims.weightCols()]);
                    biasGradients.add(new float[dims.biasSize()]);
                } else {
                    weightGradients.add(null);
                    biasGradients.add(null);
                }
            }
        }

        void reset() {
            // Zero out all gradient buffers
            for (int i = 0; i < weightGradients.size(); i++) {
                if (weightGradients.get(i) != null) {
                    NetMath.matrixInit(weightGradients.get(i), 0.0f);
                    NetMath.biasInit(biasGradients.get(i), 0.0f);
                }
            }
        }
    }
}
